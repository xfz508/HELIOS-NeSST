import numpy as np
import matplotlib.pyplot as plt

def get_helios_dtev_arrays(log_file_path, zone_range=None, time_range=None):
    with open(log_file_path, 'r') as f:
        lines = [line.strip() for line in f]

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

            r_dict, density_dict, te_dict, vel_dict = {}, {}, {}, {}

            while i < num_lines:
                inner = lines[i]
                if inner.startswith("Cycle") and "Time (s)" in inner:
                    i -= 1
                    break

                # Hydro Table (Table 1)
                if "#" in inner and "Radius" in inner and "Elec temp" in inner:
                    i += 3
                    while i < num_lines:
                        row = lines[i].split()
                        if not row or not row[0].isdigit():
                            break
                        z_id = int(row[0])
                        r_dict[z_id] = float(row[1])
                        density_dict[z_id] = float(row[3])
                        te_dict[z_id] = float(row[7])
                        vel_dict[z_id] = float(row[4])
                        i += 1
                i += 1

            if r_dict:
                all_times.append(sim_time)
                raw_time_data.append({
                    "r": r_dict,
                    "density": density_dict,
                    "Te": te_dict,
                    "vel": vel_dict
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
    sample_radius_dict = filtered_raw_data[0]["r"]
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
    te_2d = np.zeros((num_zones, num_times))
    density_2d = np.zeros((num_zones, num_times))
    r_2d = np.zeros((num_zones, num_times))
    vel_2d = np.zeros((num_zones, num_times))

    # Fill 2D arrays
    for t_idx, cycle in enumerate(filtered_raw_data):
        r = cycle["r"]
        rho = cycle["density"]
        te = cycle["Te"]
        vel = cycle["vel"]
        for z_idx, z in enumerate(selected_zones):
            te_2d[z_idx, t_idx] = te.get(z, 0.0)
            density_2d[z_idx, t_idx] = rho.get(z, 0.0)
            r_2d[z_idx, t_idx] = r.get(z, 0.0)
            vel_2d[z_idx, t_idx] = vel.get(z, 0.0)

    return {
        "times": filtered_times,
        "zones": selected_zones,
        "Te": te_2d,
        "density": density_2d,
        "r": r_2d,
        "velocity": vel_2d
    }

file = "DHe3ab/0_50DHe3ab1.log"
#alldata = get_helios_dtev_arrays(file)
#print(alldata["times"])
time = [5.000e-10, 5.000e-10]
data = get_helios_dtev_arrays(file, time_range=time)
r = data["r"]
z = data["zones"]
rho = data["density"]
te = data["Te"]
v = data["velocity"]

x = r

fig, ax1 = plt.subplots()
color = 'r'
#ax1.set_xlabel('Zone')
ax1.set_xlabel('Radius (cm)')
ax1.set_xlim(x[0]+0.032,x[-1]-0.012)
ax1.set_ylabel('Density (g/cm\u207B\u00B3)', color=color)
ax1.set_ylim(1e-3,1e1)
ax1.semilogy(x,rho, 'r', label="Density")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'g'
ax2.set_ylabel('Electron temperature (eV)', color=color)
#ax2.set_ylim(0,2000)
ax2.plot(x,te, color, label="Te")
ax2.tick_params(axis='y', labelcolor=color)
ax2.spines.right.set_position(("axes", 1.15))

ax3 = ax1.twinx()
color = 'b'
ax3.set_ylabel('Velocity (cm/s)', color=color)
ax3.set_ylim(-1e7,1e8)
ax3.plot(x,v, 'k', label="Velocity")
ax3.tick_params(axis='y', labelcolor=color)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

fig.tight_layout()

plt.show()
