import numpy as np
import matplotlib.pyplot as plt

import NeSST as nst
import helios_parser as hp
import rhoR_ray_calculator as rhr
import nesst_n_scatter as nns

file = "0_5DHe3default.log"
data = hp.get_helios_2d_arrays(file)
Tion = data["Tion"] # eV, read from HELIOS
neutron_rate = data["neutron_rate"]
# DD grid
E_spec = np.linspace(1.0e5,4.0e6,500) # creates 500 points evenly spaced between 1M and 10M
I_prim = hp.I_prim_integrator(Tion, neutron_rate, E_spec)

# avg zone width
#dr = hp.get_helios_2d_arrays("0_5DHe3default.log")["dr"]
#print(np.average(dr, weights=neutron_rate))

# 1. Compute physical <rhoL> geometry using the HELIOS adapter
nesst_inputs = rhr.extract_helios_rhoL(file)
spectra = nns.calculate_nesst_n_scatter(nesst_inputs, E_spec, I_prim, max_scatter_order=2)
Y_total = np.sum(nesst_inputs['dt'][:, None] * nesst_inputs['neutron_rate'].T)
#spectrum1 = spectra.get_spectrum(order=1)/Y_total
u_spectrum = spectra.unscattered/Y_total
t1_spectrum = u_spectrum+(spectra.get_spectrum(order=1, reaction='elastic'))/Y_total
e1_spectrum = spectra.get_spectrum(order=1, reaction='elastic')/Y_total
np1_spectrum = spectra.get_spectrum(order=1, reaction='np')/Y_total
d1_spectrum = spectra.get_spectrum(order=1, species='H2', reaction='elastic')/Y_total
sh1_spectrum = spectra.get_spectrum(order=1, species=('H1', 'C12'), reaction='elastic')/Y_total
he31_spectrum = spectra.get_spectrum(order=1, species='He3', reaction='elastic')/Y_total
t2_spectrum = u_spectrum+(spectra.get_spectrum(order=2, reaction='elastic'))/Y_total
e2_spectrum = spectra.get_spectrum(order=2, reaction='elastic')/Y_total
np2_spectrum = spectra.get_spectrum(order=2, reaction='np')/Y_total

# Default NeSST result
frac_D = hp.get_initial_fuel_fraction(file)[0]
frac_He3 = hp.get_initial_fuel_fraction(file)[1]
nst.init_DHe3_scatter(E_spec, E_spec)
dNdE_scatter,components = nst.DHe3_sym_scatter_spec(np.asarray(I_prim, dtype=np.float64),frac_D=frac_D,frac_He3=frac_He3)
mat_C12 = nst.init_mat_scatter(E_spec,E_spec,'C12')
scatter_C12_total = nst.mat_scatter_spec(mat_C12,I_prim,lambda x : np.ones_like(x))
mat_H1 = nst.init_mat_scatter(E_spec,E_spec,'H1')
scatter_H1_total = nst.mat_scatter_spec(mat_H1,I_prim,lambda x : np.ones_like(x))
drhoR = data['density']*data['dr'] # kg/m^2, read from HELIOS
rhoRf = np.flip(np.cumsum(np.flip(drhoR[:50], axis=0), axis=0), axis=0)
rhoRf_avg = 10 * np.average(rhoRf, weights=neutron_rate[:150])
rhoRs = sum(drhoR[150:])
rhoRs_avg = 10 * np.average(rhoRs, weights=sum(neutron_rate[:150]))
A_1sf = nst.DHe3rhoR_2_A1s(rhoRf_avg, frac_D=frac_D,frac_He3=frac_He3)
A_1sc = mat_C12.rhoR_2_A1s(rhoRs_avg*12.0/13.008),
A_1sh = mat_H1.rhoR_2_A1s(rhoRs_avg*1.008/13.008)
n_unscatteredf = I_prim*nst.DHe3_transmission(rhoRf_avg,E_spec,lambda x : np.ones_like(x))
n_unscatteredc = n_unscatteredf*nst.mat_transmission(mat_C12,rhoRs_avg*12.0/13.008,E_spec,lambda x : np.ones_like(x))
n_unscatteredh = n_unscatteredf*nst.mat_transmission(mat_H1,rhoRs_avg*1.008/13.008,E_spec,lambda x : np.ones_like(x))
n_unscattereds = n_unscatteredc+n_unscatteredh-n_unscatteredf
dNdE_scatter = (dNdE_scatter*A_1sf + scatter_C12_total*A_1sc + scatter_H1_total*A_1sh)
nD,nHe3,He3np = components
nesst_elastic = nD*A_1sf + nHe3*A_1sf + A_1sc * mat_C12.elastic_dNdE + A_1sh * mat_H1.elastic_dNdE
nesst_np = He3np*A_1sf
nesst_shell = A_1sc * mat_C12.elastic_dNdE + A_1sh * mat_H1.elastic_dNdE

plt.figure(dpi=200)

plt.semilogy(E_spec,I_prim, 'r', label="primary")
plt.semilogy(E_spec, t1_spectrum, 'k', label='HELIOS 1st total')
plt.semilogy(E_spec, u_spectrum, 'm', label='HELIOS 1st unscattered')
plt.semilogy(E_spec, e1_spectrum, 'g', label='HELIOS 1st elastic')
plt.semilogy(E_spec, d1_spectrum, 'b', label='HELIOS 1st H2 elastic')
plt.semilogy(E_spec, he31_spectrum, 'c', label='HELIOS 1st He3 elastic')
plt.semilogy(E_spec, sh1_spectrum, 'b:', label='HELIOS 1st shell')
plt.semilogy(E_spec, np1_spectrum, 'y', label='HELIOS 1st He3 np')
plt.semilogy(E_spec, t2_spectrum, 'k:', label='HELIOS 2nd total')
plt.semilogy(E_spec, e2_spectrum, 'g:', label='HELIOS 2nd elastic')
plt.semilogy(E_spec, np2_spectrum, 'm:', label='HELIOS 2nd He3 np')
plt.semilogy(E_spec,nesst_elastic+n_unscattereds, 'k--', label="NeSST 1st total")
plt.semilogy(E_spec,n_unscattereds, 'm--', label="NeSST 1st unscattered")
plt.semilogy(E_spec,nesst_elastic, 'g--', label="NeSST 1st elastic")
plt.semilogy(E_spec,nesst_shell, 'r:', label="NeSST 1st shell")
plt.semilogy(E_spec,nD*A_1sf, 'b--', label="NeSST 1st H2")
plt.semilogy(E_spec,nHe3*A_1sf, 'c--', label="NeSST 1st He3 elastic")
plt.semilogy(E_spec,nesst_np, 'y--', label="NeSST 1st He3 np")

plt.xlim(E_spec[0],E_spec[-1])
plt.ylim(1e-11,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend()

plt.show()



