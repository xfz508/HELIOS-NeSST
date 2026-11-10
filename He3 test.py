import numpy as np
import matplotlib.pyplot as plt
from helios_parser import get_helios_2d_arrays
import NeSST as nst

frac_D = 1.0
frac_T = 0.0
frac_He3 = 0.0
Tion = 6.4e3 # eV
"""
file = "DHe3ab/1_00DHe3ab1.log"
data = get_helios_2d_arrays(file)
drhoR = data['density']*data['dr'] # kg/m^2, read from HELIOS
rhoRf = np.flip(np.cumsum(np.flip(drhoR[:150], axis=0), axis=0), axis=0)
"""
rhoL = 0.16

DTmean,DTstddev,DTvar = nst.DTprimspecmoments(Tion)
DDmean,DDstddev,DDvar = nst.DDprimspecmoments(Tion)
Y_DT,Y_DD,Y_TT = nst.yields_normalised(Tion,frac_D=frac_D,frac_T=frac_T)
E_pspec = np.linspace(1.0e5,10.0e6,500)
E_DDspec = np.linspace(1.0e6,4.0e6,200)
E_pspec = np.append(E_pspec,E_DDspec)
dNdE_DD = Y_DD*nst.QBrysk(E_pspec,DDmean,DDvar) # Brysk shape i.e. Gaussian
I_prim = dNdE_DD
E_sspec = np.linspace(1.0e5,4.0e6,500)

A_1s = nst.rhoR_2_A1s(rhoL)
nst.init_DT_scatter(E_sspec,E_pspec)
dNdE_scatter,components = nst.DT_sym_scatter_spec(I_prim,frac_D=frac_D,frac_T=frac_T)
dNdE_scatter *= A_1s

A_1sh = nst.rhoR_2_A1s(rhoL)
nst.init_DHe3_scatter(E_sspec, E_pspec)
dNdE_scatterh,componentsh = nst.DHe3_sym_scatter_spec(np.asarray(I_prim, dtype=np.float64),frac_D=frac_D,frac_He3=frac_He3)
dNdE_scatterh *= A_1sh

plt.figure(dpi=200)
plt.semilogy(E_pspec,dNdE_DD, 'k', label="primary")
plt.semilogy(E_sspec,dNdE_scatter, 'b', label="NeSST DT f\u1D05=1")
plt.semilogy(E_sspec,dNdE_scatterh, 'r--', label="NeSST DHe3 f\u1D05=1")
plt.xlim(E_pspec[0],E_pspec[-1])
plt.ylim(1e-10,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend()

plt.show()
