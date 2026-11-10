import numpy as np

def get_initial_fuel_fraction(log_file_path):
    """
    Reads initial D/He3 fuel mass fractions from the PROPACEOS filepath string in the header.
    Expects naming format 'X_YDHe3...' (e.g., '0_5DHe3' -> D=0.5, He3=0.5).
    """
    filename = log_file_path.replace("\\", "/").split("/")[-1]

    idx = filename.find("DHe3")
    if idx == -1:
        raise ValueError("Could not find 'DHe3' substring in the log file.")

    us_idx = filename.rfind("_", 0, idx)
    if us_idx == -1:
        raise ValueError("Could not find preceding '_' before 'DHe3'.")

    former = filename[us_idx - 1]
    latter = filename[us_idx + 1:idx]

    d_fraction = float(former) if latter == "0" else float(f"{former}.{latter}")
    return d_fraction, 1.0 - d_fraction

def _parse_materials(lines):
    """
    Parses the material table / EOS files section from the log file header.
    Returns a dict mapping cleaned material name -> (min_zone, max_zone) tuple.
    """
    materials = []
    mat_index_to_name = {}
    i = 0
    num_lines = len(lines)

    # 1. Map Material Index -> Cleaned Name from PRP Filepath
    while i < num_lines:
        line = lines[i]
        if "EOS Model" in line and "Filepath" in line:
            i += 2
            while i < num_lines:
                row = lines[i].split()
                if not row or not row[0].isdigit():
                    break
                m_idx = int(row[0])
                filepath = row[-1]
                filename = filepath.replace("\\", "/").split("/")[-1]
                if filename.lower().endswith(".prp"):
                    filename = filename[:-4]

                # If DHe3 pattern is anywhere in the filename, reduce to "DHe3"
                fn_upper = filename.upper()
                if "DHE3" in fn_upper:
                    mat_name = ["H2", "He3"]
                elif "CH" in fn_upper:
                    mat_name = ["C12", "H1"]
                else:
                    mat_name = [filename]
                # Either add code to infer isotope info,
                # or manually add other materials here
                mat_index_to_name[m_idx] = mat_name
                i += 1
            break
        i += 1

    # 2. Map Material Name -> Zone Range from Region Table
    i = 0
    while i < num_lines:
        line = lines[i]
        if "Region" in line and "Min" in line:
            i += 3 # Skip divider
            while i < num_lines:
                row = lines[i].split()
                if not row or not row[0].isdigit():
                    break
                region_idx = int(row[0])
                z_min = int(row[1])
                z_max = int(row[2])
                mat_idx = abs(int(row[3]))  # Strips negative sign (-1, -2)

                material = mat_index_to_name.get(mat_idx, [f"Material_{mat_idx}"])
                for mat in material:
                    materials.append((region_idx, mat, z_min, z_max))
                i += 1
            break
        i += 1

    dtype = [
        ('region', 'i4'),  # 32-bit integer
        ('material', 'U16'),  # String (up to 16 characters)
        ('z_min', 'i4'),  # 32-bit integer
        ('z_max', 'i4')  # 32-bit integer
    ]

    # Step C: Convert the list of tuples into the structured NumPy array
    return np.array(materials, dtype=dtype)

def get_helios_2d_arrays(log_file_path, zone_range=None, time_range=None):
    """
    Extracts 2D spatiotemporal arrays (zones x time) for neutron production rate,
    ion temperature, zone density and width (rho, dr).

    Parameters:
    -----------
    log_file_path : str
        Path to the HELIOS log file.
    zone_range : tuple or list of int, optional
        (min_zone, max_zone) inclusive. Defaults to full zone range (1 to N).
    time_range : tuple or list of float, optional
        (min_time, max_time) in seconds. Defaults to full simulation time range.

    Returns:
    --------
    dict containing:
        - "times": 1D array of shape (num_times,)
        - "zones": 1D array of shape (num_zones,)
        - "Tion": 2D array of shape (num_zones, num_times) in eV
        - "density": 2D array of shape (num_zones, num_times) in g/cm^3
        - "dr": 2D array of shape (num_zones, num_times) in cm
        - "neutron_rate": 2D array of shape (num_zones, num_times) in neutrons/s
    """
    with open(log_file_path, 'r') as f:
        lines = [line.strip() for line in f]

    material_ranges = _parse_materials(lines)

    all_times = []
    raw_time_data = []

    i = 0
    num_lines = len(lines)
    while i < num_lines:
        line = lines[i]

        if line.startswith("Cycle") and "Time (s)" in line:
            try:
                sim_time = float(lines[i + 1].split()[1])
                i += 2
            except (IndexError, ValueError):
                i += 1
                continue

            dr_dict, density_dict, ti_dict, dd_rates_dict, vol_dict = {}, {}, {}, {}, {}

            while i < num_lines:
                inner = lines[i]
                if inner.startswith("Cycle") and "Time (s)" in inner:
                    i -= 1
                    break

                # Hydro Table (Table 1)
                if "#" in inner and "Radius" in inner and "Ion temp" in inner:
                    i += 3
                    while i < num_lines:
                        row = lines[i].split()
                        if not row or not row[0].isdigit():
                            break
                        z_id = int(row[0])
                        r_outer = float(row[1])
                        z_dr = float(row[2])

                        # Calculate volume shell directly from this row's outer radius & width
                        d_vol = (4.0 / 3.0) * np.pi * (r_outer ** 3 - (r_outer - z_dr) ** 3)

                        dr_dict[z_id] = z_dr
                        density_dict[z_id] = float(row[3])
                        ti_dict[z_id] = float(row[6])
                        vol_dict[z_id] = d_vol
                        i += 1

                # Reaction Rates Table (Table 3)
                elif "#" in inner and "D+D>n+He3" in inner:
                    i += 3
                    while i < num_lines:
                        row = lines[i].split()
                        if not row or not row[0].isdigit():
                            break
                        dd_rates_dict[int(row[0])] = float(row[2])
                        i += 1

                i += 1

            if dr_dict:
                all_times.append(sim_time)
                raw_time_data.append({
                    "dr": dr_dict,
                    "density": density_dict,
                    "Tion": ti_dict,
                    "dd_rates": dd_rates_dict,
                    "vol": vol_dict
                })

        i += 1

    if not all_times:
        raise ValueError("No valid cycle data found in log file.")

    all_times = np.array(all_times)

    # Apply time slice filtering
    if time_range is not None:
        t_min, t_max = time_range[0], time_range[1]
        time_mask = (all_times >= t_min) & (all_times <= t_max)
        if not np.any(time_mask):
            raise ValueError(f"No simulation cycles found within time range {time_range}.")
        filtered_times = all_times[time_mask]
        filtered_raw_data = [data for idx, data in enumerate(raw_time_data) if time_mask[idx]]
    else:
        filtered_times = all_times
        filtered_raw_data = raw_time_data

    # Determine spatial zones (excluding Zone 0 origin node)
    sample_radius_dict = filtered_raw_data[0]["dr"]
    available_zones = np.array(sorted([z for z in sample_radius_dict.keys() if z > 0]))

    if zone_range is not None:
        z_min, z_max = zone_range[0], zone_range[1]
        selected_zones = available_zones[(available_zones >= z_min) & (available_zones <= z_max)]
        if len(selected_zones) == 0:
            raise ValueError(f"No spatial zones found within zone range {zone_range}.")
    else:
        selected_zones = available_zones

    num_zones = len(selected_zones)
    num_times = len(filtered_times)

    # Initialize 2D output arrays (zones x time)
    tion_2d = np.zeros((num_zones, num_times))
    density_2d = np.zeros((num_zones, num_times))
    dr_2d = np.zeros((num_zones, num_times))
    neutron_rate_2d = np.zeros((num_zones, num_times))

    # Fill 2D arrays
    for t_idx, cycle in enumerate(filtered_raw_data):
        dr = cycle["dr"]
        rho = cycle["density"]
        ti = cycle["Tion"]
        rates = cycle["dd_rates"]
        vol = cycle["vol"]

        for z_idx, z in enumerate(selected_zones):
            z_rho = rho.get(z, 0.0)
            z_rate = rates.get(z, 0.0)
            d_vol = vol.get(z, 0.0)

            tion_2d[z_idx, t_idx] = ti.get(z, 0.0)
            density_2d[z_idx, t_idx] = z_rho
            dr_2d[z_idx, t_idx] = dr.get(z, 0.0)
            neutron_rate_2d[z_idx, t_idx] = z_rate * z_rho * d_vol

    return {
        "times": filtered_times,
        "zones": selected_zones,
        "materials": material_ranges,
        "Tion": tion_2d,
        "density": density_2d,
        "dr": dr_2d,
        "neutron_rate": neutron_rate_2d
    }

def I_prim_integrator(Tion, neutron_rate, E_pspec):
    # Initialize primary source spectrum matching NeSST grid
    I_prim_raw = np.zeros_like(E_pspec)

    E_0 = 2.45e6

    # Loop over time steps and spatial zones
    for j in range(neutron_rate.shape[0]):  # time steps
        for i in range(neutron_rate.shape[1]):  # zones
            yield_ij = neutron_rate[j, i]
            T_ij = Tion[j, i]

            # Skip cells with no reactions or zero/negative temperature
            if yield_ij <= 0:
                continue
            if T_ij <= 0:
                continue

            # Doppler broadening width in eV, constant is for Ti in keV
            sigma_eV = 35.03 * np.sqrt(T_ij / 1000) * 1e3

            # Gaussian evaluated in eV
            arg = -0.5 * ((E_pspec - E_0) / sigma_eV) ** 2

            # Avoid underflow/overflow artifacts
            cell_spectrum = yield_ij * np.exp(np.clip(arg, -500, 0)) / (np.sqrt(2 * np.pi) * sigma_eV)

            # Add to total primary spectrum
            I_prim_raw += cell_spectrum

    I_prim = I_prim_raw / np.trapezoid(I_prim_raw, E_pspec)

    return I_prim

