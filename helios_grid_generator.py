import numpy as np


def generate_helios_mesh(
    f_D,
    wall_thickness_cm,
    fill_pressure_atm=15.0,
    temp_eV=0.025,
    rho_ch=1.1,
    interface_mass_jump=1.035,
):
    """Generates a 1D spherical node grid (cm) for HELIOS D-3He target runs.

    Parameters:
    -----------
    f_D : float
        Deuterium atomic fraction (0.0 to 1.0).
    wall_thickness_cm : float
        Inner CH payload wall thickness in cm (e.g., 0.0012 cm = 12 um).
    fill_pressure_atm : float
        Initial gas fill pressure in atmospheres (default 15.0 atm).
    temp_eV : float
        Initial gas temperature in eV (0.025 eV = 290.11 K).
    rho_ch : float
        CH plastic density in g/cm3 (default 1.1 g/cm3).
    interface_mass_jump : float
        Target mass ratio across fuel-wall boundary (default 1.035 = +3.5%).
    """
    # 1. Geometry & Material Densities
    r_fuel = 0.0413  # 413 um fuel core
    r_wall = wall_thickness_cm  # Payload wall thickness
    r_ablator = 0.0008  # 8 um CH ablator
    r_al = 0.000010  # 100 nm Al coating

    rho_al = 2.70  # Aluminum density (g/cm3)

    # 2. Convert eV to Kelvin & Compute Gas Density (Ideal Gas Law)
    EV_TO_K = 11604.525
    temp_K = temp_eV * EV_TO_K  # 0.025 eV -> 290.11 K

    P_pascal = fill_pressure_atm * 101325.0  # 15 atm = 1,519,875 Pa
    molar_mass = 2.014 * f_D + 3.016 * (1.0 - f_D)  # g/mol
    R_GAS = 8.3144626  # J/(mol*K)

    # rho [g/cm3] = (P [Pa] * M [kg/mol] / (R * T [K])) / 1000
    rho_fuel = (P_pascal * (molar_mass * 1e-3) / (R_GAS * temp_K)) * 1e-3

    # 3. Layer Radii & Total Masses
    r1, r2, r3, r4 = (
        r_fuel,
        r_fuel + r_wall,
        r_fuel + r_wall + r_ablator,
        r_fuel + r_wall + r_ablator + r_al,
    )

    V1 = (4.0 / 3.0) * np.pi * r1**3
    V2 = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)
    V3 = (4.0 / 3.0) * np.pi * (r3**3 - r2**3)
    V4 = (4.0 / 3.0) * np.pi * (r4**3 - r3**3)

    M1 = rho_fuel * V1
    M2 = rho_ch * V2
    M3 = rho_ch * V3
    M4 = rho_al * V4

    # 4. Region 1 (Fuel Core): N1 = 150, q1 = 1.045
    N1 = 150
    N1_flat, N1_ramp = 45, 105
    q1 = 1.045
    sum_unit1 = N1_flat + sum(q1**k for k in range(1, N1_ramp + 1))
    m1_0 = M1 / sum_unit1
    m1 = np.concatenate(
        [
            np.full(N1_flat, m1_0),
            m1_0 * (q1 ** np.arange(1, N1_ramp + 1)),
        ]
    )

    # 5. Region 2 (Inner CH Wall): Fixed q2 = 1.040, m2[0] = m1[-1] * interface_mass_jump
    m2_0 = m1[-1] * interface_mass_jump
    q2 = 1.040
    N2 = int(np.round(np.log(1.0 + (M2 / m2_0) * (q2 - 1.0)) / np.log(q2)))
    m2 = m2_0 * (q2 ** np.arange(N2))
    m2 = m2 * (M2 / np.sum(m2))

    # 6. Region 3 (Outer CH Ablator): Fixed q3 = 0.965
    m3_0 = m2[-1] * 1.010
    q3 = 0.965
    N3 = int(np.round(np.log(1.0 - (M3 / m3_0) * (1.0 - q3)) / np.log(q3)))
    m3 = m3_0 * (q3 ** np.arange(N3))
    m3 = m3 * (M3 / np.sum(m3))

    # 7. Region 4 (Al Coating): 10 Uniform Zones
    N4 = 10
    m4 = np.full(N4, M4 / N4)

    # 8. Assemble Masses & Densities
    masses = np.concatenate([m1, m2, m3, m4])
    rhos = [rho_fuel] * N1 + [rho_ch] * (N2 + N3) + [rho_al] * N4
    N_total = len(masses)

    # 9. Integrate Zone Masses to Radial Nodes (cm)
    r_nodes = np.zeros(N_total + 1)
    for k in range(N_total):
        v_k = masses[k] / rhos[k]
        r_nodes[k + 1] = (r_nodes[k] ** 3 + (3.0 * v_k) / (4.0 * np.pi)) ** (
            1.0 / 3.0
        )

    # Verification Output
    dr = np.diff(r_nodes)
    spatial_jump = dr[N1 - 1] / dr[N1]
    q_adj = masses[1 : N1 + N2 + N3] / masses[: N1 + N2 + N3 - 1]

    print("=" * 75)
    print("            HELIOS REGION CONFIGURATION (ENTER INTO UI)            ")
    print("=" * 75)
    print(
        f"Input Temperature     : {temp_eV:.4f} eV  ({temp_K:.2f} K)"
    )
    print(
        f"Gas Density           : {rho_fuel:.6e} g/cm3  (15.0 atm)"
    )
    print(
        f"Spatial Step Drop     : {spatial_jump:.1f}x  (dr_fuel = {dr[N1-1]*1e4:.3f} µm -> dr_wall = {dr[N1]*1e7:.2f} nm)"
    )
    print(f"Boundary Mass Ratio   : {masses[N1]/masses[N1-1]:.4f}x (+3.5%)")
    print("-" * 75)
    print(f"Region 1 (Gas Fuel)   : N1 = {N1} zones")
    print(f"Region 2 (Inner Wall) : N2 = {N2} zones")
    print(f"Region 3 (Ablator)    : N3 = {N3} zones")
    print(f"Region 4 (Al Coating) : N4 = {N4} zones")
    print(f"TOTAL MESH ZONES      : {N_total} zones")
    print("-" * 75)
    print(
        f"Hydro Shell Mass Ratios: Min q = {np.min(q_adj):.4f}, Max q = {np.max(q_adj):.4f}"
    )
    print("=" * 75 + "\n")

    return r_nodes


# --- Example Execution ---
if __name__ == "__main__":
    # Runs at 0.025 eV, 15 atm fill pressure, 1.1 g/cm3 CH density
    nodes = generate_helios_mesh(f_D=0.5, wall_thickness_cm=0.0012)
    np.savetxt("helios_custom_grid_cm.dat", nodes, fmt="%.8e")