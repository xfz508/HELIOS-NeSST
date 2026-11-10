import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import NeSST as nst
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_mat_channels(mat, I_E, rhoL_func, A1s_mean):
    """
    DD-optimized transfer spectra extractor.
    Directly evaluates elastic and (n,p) loss, bypassing redundant total integrations.
    """
    # 1. Evaluate required channels directly
    mat.calc_station_elastic_dNdE(I_E, rhoL_func)
    # 2. Pre-allocate output matrix (3 channels x N_energy)
    rx_array = np.zeros((2, len(I_E)), dtype=np.float64)
    rx_array[0] = A1s_mean * mat.elastic_dNdE
    # 3. Evaluate (n,p) loss if supported by material
    if getattr(mat, 'l_np', False):
        mat.calc_np_dNdE(I_E, rhoL_func)
        if mat.np_dNdE is not None:
            rx_array[1] = A1s_mean * mat.np_dNdE

    return rx_array


class Scattered_Spectra:
    """
    Structured data container for 7D output tensor with query slicing.
    Tensor Shape: (N_times, N_zones, max_scatter_order, 3_src_regions, N_species, 4_reactions, N_E)
    """

    def __init__(self, spectra_tensor, unscattered, valid_species):
        self.spectra_tensor = spectra_tensor
        self.src_regions = ['primary', 'fuel', 'shell']
        self.reactions = ['elastic', 'np']
        self.unscattered = unscattered
        self.valid_species = valid_species

        self.sp_map = {sp: i for i, sp in enumerate(self.valid_species)}
        self.src_map = {src: i for i, src in enumerate(self.src_regions)}
        self.rx_map = {rx: i for i, rx in enumerate(self.reactions)}

    def _resolve_indices(self, key, index_map):
        """Helper to resolve strings, integers, lists, or None into integer index lists."""
        if key is None:
            return list(index_map.values())
        if isinstance(key, (int, str)):
            keys = [key]
        else:
            keys = list(key)
        return [index_map[k] if isinstance(k, str) else k for k in keys]

    def get_spectrum(self, time=None, zone=None, order=None, src_region=None, species=None, reaction=None):
        """Extracts and sums component spectra based on filter criteria."""
        t_idx = range(self.spectra_tensor.shape[0]) if time is None else ([time] if isinstance(time, int) else time)
        k_idx = range(self.spectra_tensor.shape[1]) if zone is None else ([zone] if isinstance(zone, int) else zone)

        if order is None:
            o_idx = range(self.spectra_tensor.shape[2])
        else:
            orders = [order] if isinstance(order, int) else order
            o_idx = [o - 1 for o in orders]

        src_idx = self._resolve_indices(src_region, self.src_map)
        sp_idx = self._resolve_indices(species, self.sp_map)

        if reaction is None or reaction == 'elastic':
            rx_idx = [0]
        elif reaction in ('losses', 'np_loss', 'np'):
            rx_idx = [1]
        else:
            rx_idx = self._resolve_indices(reaction, self.rx_map)

        sub_tensor = self.spectra_tensor[np.ix_(t_idx, k_idx, o_idx, src_idx, sp_idx, rx_idx)]
        return np.sum(sub_tensor, axis=tuple(range(sub_tensor.ndim - 1)))

def process_zone(k, t, dt, nesst_inputs, mat_data, I_prim,
                 valid_species, f_species, s_species, sp_map, f_sp_idx, s_sp_idx,
                 fuel_zones_only, N_zones, shell_1st_forward_only, shell_regions,
                 max_scatter_order, source_selection_dict, default_src,
                 SRC_PRIM, SRC_FUEL, SRC_SHELL):
    """
    Worker function: Computes multi-order scattering cascade for a single zone k at time t.
    Modifies spectra_tensor[t, k, ...] in-place and returns unscattered flux delta.
    """
    yield_kt = dt * nesst_inputs['neutron_rate'][k, t]
    if yield_kt <= 0:
        return None

    zone_data = nesst_inputs['path_data'][t][k]
    if zone_data is None or (fuel_zones_only and k >= N_zones // 2):
        return None

    mus = zone_data['mus']
    rhoL_func_4pi = {}
    rhoL_func_fwd = {}

    A1s_mean_dict = {}

    # Build angular rhoL functions per species
    for reg, species_dict in zone_data['rhoL'].items():
        for sp, rhoL_array in species_dict.items():
            if sp not in mat_data or np.all(rhoL_array == 0):
                continue

            mat = mat_data[sp]
            rhoL_mean = np.trapezoid(rhoL_array, x=mus) / 2.0
            if rhoL_mean <= 1e-12:
                continue
            A1s_mean_dict[sp] = mat.rhoR_2_A1s(rhoL_mean)
            shape_array = rhoL_array / rhoL_mean

            rhoL_func_4pi[sp] = lambda mu, m=mus, s=shape_array: np.interp(mu, m, s)
            #if shell_1st_forward_only and reg in shell_regions:
                #rhoL_func_fwd[sp] = lambda mu, m=mus, s=shape_array: np.where(mu >= 0, np.interp(mu, m, s), 0.0)
            #else:
            rhoL_func_fwd[sp] = rhoL_func_4pi[sp]

    # Pre-allocate local numpy array buffer for this zone (~240 KB in CPU L1/L2 cache)
    local_tensor = np.zeros(
        (max_scatter_order, 3, len(valid_species), 2, len(I_prim)),
        dtype=np.float64
    )

    # -----------------------------------------------------------------
    # Order 1 (n = 1): Primary -> 1st Scatter
    # -----------------------------------------------------------------
    I_scaled = I_prim * yield_kt  # Physical neutrons / eV

    # Retrieve energy grid and ray weights
    E_in = mat_data[valid_species[0]].Ein
    weights = zone_data['weights']

    # Pre-allocate optical depth tensor: (n_rays, N_energy)
    tau_rays = np.zeros((len(mus), len(E_in)))

    # Compute exact optical depth (tau) along each ray using NeSST's cross-sections
    for sp in valid_species:
        if sp in rhoL_func_4pi:
            rhoL_func_u = rhoL_func_4pi[sp](mus)

            A1s_per_ray = A1s_mean_dict[sp] * rhoL_func_u
            tot_xsec = mat_data[sp].sigma_tot(E_in)

            # Broadcast & add to total tau: (n_rays, 1) * (1, N_energy) -> (n_rays, N_energy)
            tau_rays += A1s_per_ray[:, None] * tot_xsec[None, :]

    # Transmit primary beam via exponential attenuation, weighted across all solid angles
    delta_unscattered = I_scaled * np.sum(weights[:, None] * np.exp(-tau_rays), axis=0)

    for sp in f_species:
        if sp in rhoL_func_4pi:
            rx_mat = extract_mat_channels(mat_data[sp], I_scaled, rhoL_func_4pi[sp], A1s_mean=A1s_mean_dict[sp])
            local_tensor[0, SRC_PRIM, sp_map[sp], :, :] += rx_mat

    for sp in s_species:
        if sp in rhoL_func_fwd:
            rx_mat = extract_mat_channels(mat_data[sp], I_scaled, rhoL_func_fwd[sp], A1s_mean=A1s_mean_dict[sp])
            local_tensor[0, SRC_PRIM, sp_map[sp], :, :] += rx_mat

    # Sum emitted neutron channels (0:1 = elastic) across species
    S_fuel_1 = np.sum(local_tensor[0, SRC_PRIM, f_sp_idx, 0:1, :],
                      axis=(0, 1)) if f_sp_idx else np.zeros_like(I_prim)
    S_shell_1 = np.sum(local_tensor[0, SRC_PRIM, s_sp_idx, 0:1, :],
                       axis=(0, 1)) if s_sp_idx else np.zeros_like(I_prim)

    S_local = {1: {'fuel': S_fuel_1, 'shell': S_shell_1}}

    # -----------------------------------------------------------------
    # Orders n >= 2: Multi-Order Cascades
    # -----------------------------------------------------------------
    for n in range(2, max_scatter_order + 1):
        o_idx = n - 1
        active_sources = [
            (SRC_FUEL, S_local[n - 1]['fuel']),
            (SRC_SHELL, S_local[n - 1]['shell'])
        ]

        for src_id, S_prev in active_sources:
            if np.sum(S_prev) > 0:
                for sp in valid_species:
                    if sp in rhoL_func_4pi:
                        rx_mat = extract_mat_channels(mat_data[sp], S_prev, rhoL_func_4pi[sp], A1s_mean=A1s_mean_dict[sp])
                        local_tensor[o_idx, src_id, sp_map[sp], :, :] += rx_mat

        # Extract regional scattered totals for routing into order n+1
        ff_n = np.sum(local_tensor[o_idx, SRC_FUEL, f_sp_idx, 0:1, :],
                      axis=(0, 1)) if f_sp_idx else np.zeros_like(I_prim)
        fs_n = np.sum(local_tensor[o_idx, SRC_FUEL, s_sp_idx, 0:1, :],
                      axis=(0, 1)) if s_sp_idx else np.zeros_like(I_prim)
        sf_n = np.sum(local_tensor[o_idx, SRC_SHELL, f_sp_idx, 0:1, :],
                      axis=(0, 1)) if f_sp_idx else np.zeros_like(I_prim)
        ss_n = np.sum(local_tensor[o_idx, SRC_SHELL, s_sp_idx, 0:1, :],
                      axis=(0, 1)) if s_sp_idx else np.zeros_like(I_prim)

        src_choice = source_selection_dict.get(n, default_src)
        if src_choice == 'fuel':
            S_local[n] = {'fuel': ff_n, 'shell': fs_n}
        elif src_choice == 'shell':
            S_local[n] = {'fuel': sf_n, 'shell': ss_n}
        else:  # 'total'
            S_local[n] = {'fuel': ff_n + sf_n, 'shell': fs_n + ss_n}

    return t, k, local_tensor,delta_unscattered

def calculate_nesst_n_scatter(
        nesst_inputs,
        E_spec,
        I_prim,
        species_list=('H2', 'He3', 'C12', 'H1', 'Al27'),
        fuel_species=('H2', 'He3', 'D'),
        shell_species=('C12', 'H1', 'Al27'),
        shell_regions=('CH', 'Al'),
        max_scatter_order=1,
        shell_1st_forward_only=True,
        fuel_zones_only=True,
        source_selection=None
):

    store = {}
    # Parse source selection toggles
    source_selection_dict = source_selection if isinstance(source_selection, dict) else {}
    default_src = source_selection if isinstance(source_selection, str) else 'total'

    # Initialize materials
    mat_data = {}
    for sp in species_list:
        nst.initialise_material_data(sp)
        if sp in nst.mat_dict:
            mat_data[sp] = nst.init_mat_scatter(E_spec, E_spec, sp)
            mat_data[sp].l_n2n = False
            mat_data[sp].l_inelastic = False

    valid_species = [sp for sp in species_list if sp in mat_data]
    f_species = [sp for sp in valid_species if sp in fuel_species]
    s_species = [sp for sp in valid_species if sp in shell_species]

    sp_map = {sp: i for i, sp in enumerate(valid_species)}
    f_sp_idx = [sp_map[sp] for sp in f_species]
    s_sp_idx = [sp_map[sp] for sp in s_species]

    N_times = len(nesst_inputs['times'])
    N_zones = nesst_inputs['neutron_rate'].shape[0]
    dt_array = np.gradient(np.asarray(nesst_inputs['times']))

    # Pre-allocate master 7D output tensor
    spectra_tensor = zarr.zeros(
        (N_times, N_zones, max_scatter_order, 3, len(valid_species), 2, len(I_prim)),
        chunks=(1, 1, max_scatter_order, 3, len(valid_species), 2, len(I_prim)),
        dtype=np.float64,
        overwrite=True,
        store=store
    )
    unscattered = np.zeros_like(I_prim)
    SRC_PRIM, SRC_FUEL, SRC_SHELL = 0, 1, 2

    # =========================================================================
    # Pointwise Spatio-Temporal Cascade Loop
    # =========================================================================
    num_workers = 8  # Adjust to match available CPU cores
    completed_zones = 0
    total_zones = N_times * N_zones
    print("Calculating spectra...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_zone, k, t, dt_array[t], nesst_inputs, mat_data, I_prim,
                valid_species, f_species, s_species, sp_map,
                f_sp_idx, s_sp_idx, fuel_zones_only, N_zones, shell_1st_forward_only,
                shell_regions, max_scatter_order, source_selection_dict, default_src,
                SRC_PRIM, SRC_FUEL, SRC_SHELL
            )
            for t in range(N_times)
            for k in range(N_zones)
        ]

        # Collect results and update global unscattered array safely
        for future in as_completed(futures):
            res = future.result()
            completed_zones += 1

            if res is not None:
                t, k, local_tensor, delta_unscattered = res
                # Single write per zone into its own independent Zarr chunk
                spectra_tensor[t, k] = local_tensor
                unscattered += delta_unscattered

            # Single self-updating progress counter in stdout
            if completed_zones % N_zones == 0:
                t_done = completed_zones // N_zones
                print(f"\rCompleted timestep {t_done}/{N_times} ({t_done / N_times * 100:.1f}%)", end="", flush=True)

    print("\nSpectra calculation complete.")
    return Scattered_Spectra(spectra_tensor, unscattered, valid_species)