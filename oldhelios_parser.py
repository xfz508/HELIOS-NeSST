import numpy as np


def get_initial_fuel_fraction(log_file_path):
    """
    Reads initial D/He3 fuel mass fractions from the PROPACEOS filepath string in the header.
    Expects naming format 'X_YDHe3...' (e.g., '0_5DHe3' -> D=0.5, He3=0.5).
    """
    with open(log_file_path, 'r') as f:
        content = f.read()

    idx = content.find("DHe3")
    if idx == -1:
        raise ValueError("Could not find 'DHe3' substring in the log file.")

    us_idx = content.rfind("_", 0, idx)
    if us_idx == -1:
        raise ValueError("Could not find preceding '_' before 'DHe3'.")

    former = int(content[us_idx - 1])
    latter = int(content[us_idx + 1:idx])

    d_fraction = float(former) if latter == 0 else float(f"0.{latter}")
    return d_fraction, 1.0 - d_fraction


def _parse_region_map(lines):
    """
    Parses the 'Spatial Region parameters' table in the header to return region mapping metadata.
    Returns: (regions_dict, fuel_region_ids, shell_region_ids)
    """
    fuel_mat_index = None

    # 1. Identify material index assigned to DHe3 in EOS table
    for i, line in enumerate(lines):
        if "material equation of state parameters:" in line.lower():
            for sub_line in lines[i + 1:]:
                if "material opacity parameters:" in sub_line.lower() or "spatial region parameters:" in sub_line.lower():
                    break
                if "dhe3" in sub_line.lower() and ".prp" in sub_line.lower():
                    fuel_mat_index = "-" + sub_line.split()[0]
                    break
            break

    if not fuel_mat_index:
        raise ValueError("Could not find DHe3 material entry in Material EOS table.")

    # 2. Parse Spatial Region parameters table rows into structured metadata
    regions = {}
    fuel_ids, shell_ids = [], []

    for i, line in enumerate(lines):
        if "spatial region parameters:" in line.lower():
            for sub_line in lines[i + 1:]:
                parts = sub_line.split()
                # Row format: [Region Index, Min Index, Max Index, Material Index, ...]
                if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                    reg_id = int(parts[0])
                    is_fuel = (parts[3] == fuel_mat_index)
                    regions[reg_id] = {
                        "min": int(parts[1]),
                        "max": int(parts[2]),
                        "is_fuel": is_fuel
                    }
                    if is_fuel:
                        fuel_ids.append(reg_id)
                    else:
                        shell_ids.append(reg_id)
                elif "cycle" in sub_line.lower() or "edit parameters:" in sub_line.lower() or not sub_line:
                    if regions:
                        break
            break

    if not regions:
        raise ValueError("Could not parse any regions from 'Spatial Region parameters' table.")

    return regions, fuel_ids, shell_ids


def _parse_full_log(log_file_path, shell_regions=None):
    """
    Base engine: Reads log file, identifies spatial bounds for fuel and target shell regions,
    and calculates time-series metrics across simulation cycles.
    """
    with open(log_file_path, 'r') as f:
        lines = [line.strip() for line in f]

    region_map, fuel_ids, all_shell_ids = _parse_region_map(lines)

    if shell_regions is None:
        target_shell_ids = all_shell_ids
    else:
        target_shell_ids = [r for r in shell_regions if r in all_shell_ids]
        if not target_shell_ids:
            raise ValueError(f"None of requested shell regions {shell_regions} match available {all_shell_ids}.")

    trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    times, total_flux = [], []
    fuel_data = {"rhoR": [], "Tion_mean": [], "Tion_stdev": [], "Tion_var": []}
    indiv_shell_data = {r_id: {"rhoR": [], "Tion_mean": [], "Tion_stdev": [], "Tion_var": []} for r_id in
                        target_shell_ids}

    # Only allocate combined data structures if multiple shell regions are targeted
    is_multi_shell = len(target_shell_ids) > 1
    combined_shell_data = {"rhoR": [], "Tion_mean": [], "Tion_stdev": [], "Tion_var": []} if is_multi_shell else None

    i = 0
    num_lines = len(lines)
    while i < num_lines:
        line = lines[i]

        if line.startswith("Cycle") and "Time (s)" in line:
            try:
                sim_time = float(lines[i + 1].split()[1])
                times.append(sim_time)
                i += 2
            except (IndexError, ValueError):
                i += 1
                continue

            radius_dict, density_dict, ti_dict, dd_rates_dict = {}, {}, {}, {}

            while i < num_lines:
                inner = lines[i]
                if inner.startswith("Cycle") and "Time (s)" in inner:
                    i -= 1
                    break

                if "#" in inner and "Radius" in inner and "Ion temp" in inner:
                    i += 3
                    while i < num_lines:
                        row = lines[i].split()
                        if not row or not row[0].isdigit():
                            break
                        z_id = int(row[0])
                        radius_dict[z_id] = float(row[1])
                        density_dict[z_id] = float(row[3])
                        ti_dict[z_id] = float(row[6])
                        i += 1

                elif "#" in inner and "D+D>n+He3" in inner:
                    i += 2
                    while i < num_lines:
                        row = lines[i].split()
                        if not row or not row[0].isdigit():
                            break
                        dd_rates_dict[int(row[0])] = float(row[2])
                        i += 1

                i += 1

            if radius_dict:
                zones = np.array(sorted(radius_dict.keys()))
                r = np.array([radius_dict[z] for z in zones])
                rho = np.array([density_dict.get(z, 0.0) for z in zones])
                Ti = np.array([ti_dict.get(z, 0.0) for z in zones])
                rates = np.array([dd_rates_dict.get(z, 0.0) for z in zones])

                # Spatial masks
                fuel_mask = np.zeros(len(zones), dtype=bool)
                for f_id in fuel_ids:
                    fuel_mask |= (zones >= region_map[f_id]["min"]) & (zones <= region_map[f_id]["max"])

                comb_shell_mask = np.zeros(len(zones), dtype=bool) if is_multi_shell else None
                indiv_masks = {}
                for s_id in target_shell_ids:
                    s_mask = (zones >= region_map[s_id]["min"]) & (zones <= region_map[s_id]["max"])
                    indiv_masks[s_id] = s_mask
                    if is_multi_shell:
                        comb_shell_mask |= s_mask

                # Volume-integrated neutron rate
                rate_vol = rates * rho
                mid_r = (r[1:] + r[:-1]) / 2.0
                dr = np.diff(r)
                mid_rates = (rate_vol[1:] + rate_vol[:-1]) / 2.0
                dV = 4.0 * np.pi * (mid_r ** 2) * dr
                total_flux.append(np.sum(mid_rates * dV))

                def calc_stats(mask):
                    if np.sum(mask) > 0:
                        rhoR = trap_func(rho[mask], r[mask]) if np.sum(mask) > 1 else 0.0
                        return rhoR, float(np.mean(Ti[mask])), float(np.std(Ti[mask])), float(np.var(Ti[mask]))
                    return 0.0, 0.0, 0.0, 0.0

                # Fuel stats
                rR, m, s, v = calc_stats(fuel_mask)
                fuel_data["rhoR"].append(rR);
                fuel_data["Tion_mean"].append(m)
                fuel_data["Tion_stdev"].append(s);
                fuel_data["Tion_var"].append(v)

                # Individual Shell stats (always run)
                for s_id in target_shell_ids:
                    rR, m, s, v = calc_stats(indiv_masks[s_id])
                    indiv_shell_data[s_id]["rhoR"].append(rR)
                    indiv_shell_data[s_id]["Tion_mean"].append(m)
                    indiv_shell_data[s_id]["Tion_stdev"].append(s)
                    indiv_shell_data[s_id]["Tion_var"].append(v)

                # Combined Shell stats (ONLY calculated if multiple shell regions exist)
                if is_multi_shell:
                    rR, m, s, v = calc_stats(comb_shell_mask)
                    combined_shell_data["rhoR"].append(rR);
                    combined_shell_data["Tion_mean"].append(m)
                    combined_shell_data["Tion_stdev"].append(s);
                    combined_shell_data["Tion_var"].append(v)

        i += 1

    # Format output dictionaries
    formatted_indiv = {s_id: {k: np.array(v) for k, v in data.items()} for s_id, data in indiv_shell_data.items()}

    # Assign "shell" directly from the single region's dictionary if len == 1, avoiding duplicate dict copies
    if not is_multi_shell:
        shell_dict = formatted_indiv[target_shell_ids[0]]
    else:
        shell_dict = {k: np.array(v) for k, v in combined_shell_data.items()}

    return {
        "times": np.array(times),
        "total_flux": np.array(total_flux),
        "fuel": {k: np.array(v) for k, v in fuel_data.items()},
        "shell": shell_dict,
        "shell_regions": formatted_indiv
    }


def parse_helios_at_time(log_file_path, target_time, shell_regions=None):
    """
    Evaluates simulation properties at a specific time step (closest match).
    Accepts optional list of 'shell_regions' (e.g., [2, 3]).
    Returns total neutron rate, fuel stats, combined shell stats, and stats per individual shell region.
    """
    data = _parse_full_log(log_file_path, shell_regions=shell_regions)
    times = data["times"]
    if len(times) == 0:
        raise ValueError("No cycle time data found in log file.")

    idx = np.argmin(np.abs(times - target_time))

    res = {
        "time_s": times[idx],
        "total_neutron_rate_n_per_s": data["total_flux"][idx],
        "fuel": {
            "rhoR_g_per_cm2": data["fuel"]["rhoR"][idx],
            "Tion_mean_ev": data["fuel"]["Tion_mean"][idx],
            "Tion_stdev_ev": data["fuel"]["Tion_stdev"][idx],
            "Tion_var_ev": data["fuel"]["Tion_var"][idx]
        },
        "shell": {
            "rhoR_g_per_cm2": data["shell"]["rhoR"][idx],
            "Tion_mean_ev": data["shell"]["Tion_mean"][idx],
            "Tion_stdev_ev": data["shell"]["Tion_stdev"][idx],
            "Tion_var_ev": data["shell"]["Tion_var"][idx]
        },
        "shell_regions": {}
    }

    for s_id, s_data in data["shell_regions"].items():
        res["shell_regions"][s_id] = {
            "rhoR_g_per_cm2": s_data["rhoR"][idx],
            "Tion_mean_ev": s_data["Tion_mean"][idx],
            "Tion_stdev_ev": s_data["Tion_stdev"][idx],
            "Tion_var_ev": s_data["Tion_var"][idx]
        }

    return res


def parse_helios_time_range(log_file_path, start_time=None, end_time=None, shell_regions=None):
    """
    Evaluates simulation properties over a time range.
    Accepts optional list of 'shell_regions' (e.g., [2, 3]).
    Returns integrated neutron count, and time-averaged metrics for fuel, combined shell, and individual shell regions.
    """
    data = _parse_full_log(log_file_path, shell_regions=shell_regions)
    times = data["times"]
    if len(times) == 0:
        raise ValueError("No cycle time data found in log file.")

    t_start = times[0] if start_time is None else start_time
    t_end = times[-1] if end_time is None else end_time

    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        raise ValueError(f"No simulation steps found in specified time range [{t_start}, {t_end}].")

    sub_times = times[mask]
    sub_flux = data["total_flux"][mask]

    trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    total_neutrons = trap_func(sub_flux, sub_times) if len(sub_times) > 1 else 0.0

    res = {
        "time_range_s": (sub_times[0], sub_times[-1]),
        "total_neutron_count": total_neutrons,
        "fuel": {
            "time_avg_rhoR_g_per_cm2": np.mean(data["fuel"]["rhoR"][mask]),
            "time_avg_Tion_ev": np.mean(data["fuel"]["Tion_mean"][mask])
        },
        "shell": {
            "time_avg_rhoR_g_per_cm2": np.mean(data["shell"]["rhoR"][mask]),
            "time_avg_Tion_ev": np.mean(data["shell"]["Tion_mean"][mask])
        },
        "shell_regions": {}
    }

    for s_id, s_data in data["shell_regions"].items():
        res["shell_regions"][s_id] = {
            "time_avg_rhoR_g_per_cm2": np.mean(s_data["rhoR"][mask]),
            "time_avg_Tion_ev": np.mean(s_data["Tion_mean"][mask])
        }

    return res