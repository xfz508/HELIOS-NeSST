import h5py
import matplotlib.pyplot as plt


with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_07DHe3ab1.h5", "r") as f1:
    E_spec = f1["E_spec"][()]

    I_prim_0_07 = f1["I_prim"][()]

with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_27DHe3ab1.h5", "r") as f2:
    I_prim_0_27 = f2["I_prim"][()]

with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_50DHe3ab1.h5", "r") as f3:
    I_prim_0_50 = f3["I_prim"][()]
with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_78DHe3ab1.h5", "r") as f4:
    I_prim_0_78 = f4["I_prim"][()]
with h5py.File("C:/Users/Raphael/Desktop/NeSST output/1_00DHe3ab1.h5", "r") as f5:
    I_prim_1_00 = f5["I_prim"][()]



#f1.close()

plt.figure(dpi=200)

plt.semilogy(E_spec,I_prim_0_07, 'r', label="0.07 primary")
plt.semilogy(E_spec,I_prim_0_27, 'r', label="0.27 primary")
plt.semilogy(E_spec,I_prim_0_50, 'r', label="0.50 primary")
plt.semilogy(E_spec,I_prim_0_78, 'r', label="0.78 primary")
plt.semilogy(E_spec,I_prim_1_00, 'r', label="1.00 primary")

"""
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
"""
plt.xlim(15*E_spec[0],0.75*E_spec[-1])
plt.ylim(1e-11,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend()

plt.show()

