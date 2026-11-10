import numpy as np
import matplotlib.pyplot as plt

import NeSST as nst
import helios_parser as hp
import h5py

file = "DHe3ab/0_50DHe3ab1.log"
data = hp.get_helios_2d_arrays(file)

Tion6k = 6400 #2570

tion = data["Tion"]
neutron_rate = data["neutron_rate"]
mask = (neutron_rate > 0) & (tion > 0)
Tion = np.sum(neutron_rate[mask] * tion[mask]) / np.sum(neutron_rate[mask])
print(Tion)

E_pspec = np.linspace(1.0e5,4.0e6,500) # creates 300 points evenly spaced between 1M and 10M

DDmean,DDstddev,DDvar = nst.DDprimspecmoments(Tion)
DDmean6k,DDstddev6k,DDvar6k = nst.DDprimspecmoments(Tion6k)

Y_DD = 1

#Y_HELIOS = np.sum(neutron_rate)

dNdE_DD = Y_DD*nst.QBrysk(E_pspec,DDmean,DDvar) # Brysk shape i.e. Gaussian
dNdE_DD6k = Y_DD*nst.QBrysk(E_pspec,DDmean6k,DDvar6k) # Brysk shape i.e. Gaussian

I_primn = dNdE_DD
I_prim6k = dNdE_DD6k

E_spec = np.linspace(1.0e5, 4.0e6, 500)  # creates 500 points evenly spaced between 1M and 10M
I_primh = hp.I_prim_integrator(tion, neutron_rate, E_spec) # I_prim_born/Y_HELIOS

with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_50DHe3ab1.h5", "r") as f3:
    I_primh6k = f3["I_prim"][()]

plt.figure(dpi=200)

plt.semilogy(E_pspec,I_prim6k, 'c', label="HELIOS 6.4 keV")
plt.semilogy(E_pspec,I_prim6k, 'm--', label="NeSST 6.4 keV")
plt.semilogy(E_pspec,I_primn, 'b', label=f"NeSST {(Tion/1000):.2} keV")
plt.semilogy(E_pspec,I_primh, 'r--', label=f"HELIOS {(Tion/1000):.2} keV")

#plt.title("Primary Spectra from NeSST vs HELIOS")
plt.xlim(E_pspec[0]*20,E_pspec[-1]*0.75)
plt.ylim(1e-6,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.yticks(np.logspace(-6,-5, 2))
plt.yticks(np.linspace(1e-6,1e-5, 10), minor=True, labels=[])
plt.legend()

plt.show()


def NeSST_DHe3_shell(file, E_spec, I_prim):
    # Default NeSST result
    frac_D = hp.get_initial_fuel_fraction(file)[0]
    frac_He3 = hp.get_initial_fuel_fraction(file)[1]
    nst.init_DHe3_scatter(E_spec, E_spec)
    dNdE_scatterf, components = nst.DHe3_sym_scatter_spec(np.asarray(I_prim, dtype=np.float64), frac_D=frac_D,
                                                         frac_He3=frac_He3)
    mat_C12 = nst.init_mat_scatter(E_spec, E_spec, 'C12')
    mat_H1 = nst.init_mat_scatter(E_spec, E_spec, 'H1')
    scatter_C12_total = nst.mat_scatter_spec(mat_C12, I_prim, lambda x: np.ones_like(x))
    scatter_H1_total = nst.mat_scatter_spec(mat_H1, I_prim, lambda x: np.ones_like(x))
    drhoR = data['density'] * data['dr']  # kg/m^2, read from HELIOS
    rhoRf = np.flip(np.cumsum(np.flip(drhoR[:150], axis=0), axis=0), axis=0)
    rhoRf_avg = 10 * np.average(rhoRf, weights=neutron_rate[:150])
    rhoRs = sum(drhoR[150:])
    rhoRs_avg = 10 * np.average(rhoRs, weights=sum(neutron_rate[:150]))
    A_1sf = nst.DHe3rhoR_2_A1s(rhoRf_avg, frac_D=frac_D, frac_He3=frac_He3)
    A_1sc = mat_C12.rhoR_2_A1s(rhoRs_avg * 12.0 / 13.008),
    A_1sh = mat_H1.rhoR_2_A1s(rhoRs_avg * 1.008 / 13.008)
    nC = A_1sc * mat_C12.elastic_dNdE
    nH = A_1sh * mat_H1.elastic_dNdE
    unscatteredf = I_prim * nst.DHe3_transmission(rhoRf_avg, E_spec, lambda x: np.ones_like(x))
    unscatteredc = unscatteredf * nst.mat_transmission(mat_C12, rhoRs_avg * 12.0 / 13.008, E_spec,
                                                           lambda x: np.ones_like(x))
    unscatteredh = unscatteredf * nst.mat_transmission(mat_H1, rhoRs_avg * 1.008 / 13.008, E_spec,
                                                           lambda x: np.ones_like(x))
    #unscattereds = unscatteredc + unscatteredh - unscatteredf
    #dNdE_scatter = (dNdE_scatterf * A_1sf + nC + nH)
    nD, nHe3, He3np = components
    nD *= A_1sf
    nHe3 *= A_1sf
    He3np *= A_1sf

    return nD, nHe3, He3np, nC, nH, unscatteredf, unscatteredc, unscatteredh

nesst = NeSST_DHe3_shell(file, E_spec, I_prim6k)
helios = NeSST_DHe3_shell(file, E_spec, I_primh)
nesst_elastic = nesst[0] + nesst[1] + nesst[3] + nesst[4]
nesst_np = nesst[2]
helios_elastic = helios[0] + helios[1]+ helios[3]+ helios[4]
helios_np = helios[2]

plt.figure(dpi=200)

plt.semilogy(E_pspec,I_prim6k, 'b', label="NeSST 6.4 keV")
plt.semilogy(E_spec,nesst_elastic, 'c', label="NeSST elastic")
plt.semilogy(E_spec,nesst_np, 'g', label="NeSST He3 np")
plt.semilogy(E_pspec,I_primh, 'r--', label=f"HELIOS {(Tion/1000):.2} keV")
plt.semilogy(E_spec,nesst_elastic, 'm--', label="HELIOS elastic")
plt.semilogy(E_spec,nesst_np, 'k--', label="HELIOS He3 np")

plt.xlim(E_spec[0],E_spec[-1]*0.8)
plt.ylim(1e-10,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend()

plt.show()
