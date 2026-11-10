import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from helios_parser import get_helios_2d_arrays, get_initial_fuel_fraction
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_chord_lengths(r0, mus, radii):
    """
    Calculates analytical chord lengths Delta s_j in each spherical shell j
    for a ray starting at radius r0 with direction cosine mu = cos(theta).

    radii: array of shell boundaries [R_0=0, R_1, R_2, ..., R_N] at current time step
    """
    p2 = (r0 ** 2) * (1.0 - mus[:, None] ** 2)
    R2 = radii[None, :] ** 2

    # Mask shells where impact parameter p < R
    valid = R2 > p2
    disc = np.sqrt(np.maximum(0.0, R2 - p2))
    s_minus = np.maximum(0.0, -r0 * mus[:, None] - disc)
    s_plus = np.maximum(0.0, -r0 * mus[:, None] + disc)
    s_len = np.where(valid, s_plus - s_minus, 0.0)
    return np.diff(s_len, axis=1)


def get_split_domain_rays(k_index, a=2, b=4):
    """
    Generates dynamic ray directions (mus) and weights (weights) for source zone k_index.
    Uses 'a * k' inward rays for inner boundary crossings and 'b' fixed outward rays.
    """
    k = k_index + 1  # 1-based zone index
    n_in = a * k
    n_out = b

    # Inward Quadrature on [-1, 0]
    x_in, w_in = np.polynomial.legendre.leggauss(n_in)
    mus_in = 0.5 * (x_in - 1.0)
    weights_in = 0.5 * w_in

    # Outward Quadrature on [0, 1]
    x_out, w_out = np.polynomial.legendre.leggauss(n_out)
    mus_out = 0.5 * (x_out + 1.0)
    weights_out = 0.5 * w_out

    mus = np.concatenate((mus_in, mus_out))
    weights = 0.5 * np.concatenate((weights_in, weights_out))  # Includes 1/2 solid angle norm

    return mus, weights


def calculate_staged_rhoL_single_timestep(
        radii,  # 1D boundary radii array [0, R_1, ..., R_N] (cm)
        species,  # Dict of species 1D density arrays per zone (10 x g/cm^3)
        region_tags,  # List/1D array of region strings per zone (e.g. ['fuel', 'fuel', 'ablator', 'coating'])
        precomputed_rays=None,
        a=2, b=4
):
    """
    Computes species path areal density <rhoL> for every source zone k,
    broken down by destination material region (fuel, ablator, coating, etc.).

    Returns: dict[source_zone_k][region_tag][species] = scalar <rhoL> (kg/m^2)
    """
    N_zones = len(radii) - 1
    r_mid = 0.5 * (radii[:-1] + radii[1:])
    region_tags_arr = np.asarray(region_tags)
    unique_regions = np.unique(region_tags_arr)
    reg_masks = {reg: (region_tags_arr == reg).astype(float) for reg in unique_regions}

    if precomputed_rays is None:
        precomputed_rays = [get_split_domain_rays(k, a=a, b=b) for k in range(N_zones)]

    sp_names = list(species.keys())
    sp_matrix = np.array([species[sp] for sp in sp_names]) # Shape: (N_species, N_zones)

    staged_rhoL = {}

    for k in range(N_zones):
        r0 = r_mid[k]
        mus, weights = precomputed_rays[k]

        delta_s = compute_chord_lengths(r0, mus, radii)
        rhoL_dict = {reg: {} for reg in unique_regions}

        for reg, reg_mask in reg_masks.items():
            delta_s_reg = delta_s * reg_mask[None, :]
            rhoL_mat = np.dot(delta_s_reg, sp_matrix.T)

            for idx, sp in enumerate(sp_names):
                rhoL_dict[reg][sp] = rhoL_mat[:, idx]

        staged_rhoL[k] = {
            'mus': mus,
            'weights': weights,
            'rhoL': rhoL_dict
        }

    return staged_rhoL

def _process_single_timestep_rays(t, neutron_rate_t, zone_widths_t, rho_species_t, region_tags, all_zone_rays, min_rate_threshold):
    if np.max(neutron_rate_t) < min_rate_threshold:
        return t, None

    radii_t = np.concatenate(([0.0], np.cumsum(zone_widths_t)))
    staged_rhoL_t = calculate_staged_rhoL_single_timestep(
        radii_t, rho_species_t, region_tags, precomputed_rays=all_zone_rays
    )
    print(t)
    return t, staged_rhoL_t

def extract_helios_rhoL(
        file,
        min_rate_threshold=1.0e4
):
    """
    Master controller that loops over HELIOS output arrays and accumulates
    yield-weighted <rhoL> and effective source parameters across time and space.
    """

    # --- 1. Unpack & Parse HELIOS Output Dictionary ---
    helios_data = get_helios_2d_arrays(file)
    f_D_atom = get_initial_fuel_fraction(file)[0]
    f_He3_atom = get_initial_fuel_fraction(file)[1]
    times = helios_data['times']
    neutron_rate = helios_data['neutron_rate']  # (N_zones, N_times) [neutrons/s]
    density = 10.0 * helios_data['density']  # (N_zones, N_times) [10 x g/cm^3 so rhoR in kg/m2]
    zone_widths = helios_data['dr']  # (N_zones, N_times) [cm]
    materials = helios_data['materials']  # Dict (range: mat) or List of mat names

    dt = np.gradient(times)  # Calculate time step intervals [s]
    N_zones, N_times = neutron_rate.shape

    # Map materials to per-zone names & set region tags ('fuel' vs shell material)
    zone_materials = [''] * N_zones
    for row in materials:
        mat = row['material']
        z_min_idx = row['z_min'] - 1  # Convert 1-based log index to 0-based Python index
        z_max_idx = row['z_max']  # Exclusive end for Python slicing
        for z in range(z_min_idx, z_max_idx):
            zone_materials[z] = mat

    # --- 1. Map Zone Materials & Region Tags ---
    def get_region_tag(mat):
        if mat in ('H2', 'He3'):
            return 'fuel'
        elif mat in ('C12', 'H1'):
            return 'CH'
        return mat

    region_tags = np.array([get_region_tag(m) for m in zone_materials])

    # --- 2. Calculate Species Mass Fractions ---
    m_D, m_He3 = 2.014, 3.016
    tot_fuel_m = (f_D_atom * m_D) + (f_He3_atom * m_He3)
    mass_fractions = {
        'H2': (f_D_atom * m_D) / tot_fuel_m,
        'He3': (f_He3_atom * m_He3) / tot_fuel_m,
        'C12': 12.0 / 13.008,
        'H1': 1.008 / 13.008,
        'Al27': 1.0
    }
    # Pre-allocate species density 2D arrays
    rho_species = {}
    for frac in mass_fractions:
        rho_species[frac] = np.zeros((N_zones, N_times))

    # --- 3. Fill Species Densities (Vectorized by Region) ---
    for row in materials:
        mat = row['material']
        z_min = row['z_min'] - 1
        z_max = row['z_max']

        # Normalize species key (e.g., 'Al' -> 'Al27')
        sp = 'Al27' if 'Al' in mat else mat

        if sp in rho_species:
            rho_species[sp][z_min:z_max, :] = mass_fractions[sp] * density[z_min:z_max, :]

    # Stores path data indexed by [t][k]
    path_data = [[None for _ in range(N_zones)] for _ in range(N_times)]

    # Store yield-weighted inputs ready for NeSST
    nesst_inputs = {
        'times': times,
        'dt': dt,
        'neutron_rate': neutron_rate,
        'path_data': path_data  # path_data[t][k] holds ray directions, weights, & rhoL per path
        }

    all_zone_rays = [get_split_domain_rays(k) for k in range(N_zones)]

    print(N_times, N_zones)
    print("Calculating rays...")

    tasks = [
        (
            t,
            neutron_rate[:, t],
            zone_widths[:, t],
            {sp: arr[:, t] for sp, arr in rho_species.items()},
            region_tags,
            all_zone_rays,
            min_rate_threshold
        )
        for t in range(N_times)
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_single_timestep_rays, *task) for task in tasks]
        for future in as_completed(futures):
            t, staged_rhoL_t = future.result()
            if staged_rhoL_t is not None:
                for k in range(N_zones):
                    path_data[t][k] = staged_rhoL_t[k]

    return nesst_inputs