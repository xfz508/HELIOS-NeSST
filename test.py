import numpy as np

import helios_parser as hp
import rhoR_ray_calculator as rc
import nesst_n_scatter as nns

file = "0_5DHe3default.log"
data = hp.get_helios_2d_arrays(file)
Tion = data["Tion"] # eV, read from HELIOS
neutron_rate = data["neutron_rate"]
# DD grid
E_spec = np.linspace(1.0e5,4.0e6,500) # creates 300 points evenly spaced between 1M and 10M
I_prim = hp.I_prim_integrator(Tion, neutron_rate, E_spec)

nesst_inputs = rc.extract_helios_rhoL(file)

spectra = nns.calculate_nesst_n_scatter(nesst_inputs, E_spec, I_prim)

print(spectra.get_spectrum(order=1, species='He3', reaction='np'))