import numpy as np
import matplotlib.pyplot as plt

import NeSST as nst
import helios_parser as hp
from nesst_n_scatter import Scattered_Spectra as nns
import h5py

# Yield-weighted average result
file = "DHe3ab/0_50DHe3ab1.log"
data = hp.get_helios_2d_arrays(file)
Tion = data["Tion"] # eV, read from HELIOS
neutron_rate = data["neutron_rate"]
E_spec = np.linspace(1.0e5,4.0e6,500) # creates 500 points evenly spaced between 1M and 10M
I_prim = hp.I_prim_integrator(Tion, neutron_rate, E_spec)
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
#dNdE_scatter = (dNdE_scatterf * A_1sf + nC + nH)
nD, nHe3, He3np = components
nD *= A_1sf
nHe3 *= A_1sf
He3np *= A_1sf
helios_elastic = nD + nHe3 + nC + nH
helios_np = He3np

class _LazyHDF5TensorProxy:
    """Wraps an h5py.Dataset to support np.ix_ multi-dimensional fancy indexing

    and automatically re-opens the file if the handle was closed.
    """

    def __init__(self, dset):
        self._filename = dset.file.filename
        self._dset_name = dset.name
        self._dset = dset
        self._file_ref = None

    def _get_dset(self):
        """Ensures the dataset handle is active, re-opening the HDF5 file if it was closed."""
        if not self._dset.id.valid:
            self._file_ref = h5py.File(self._filename, "r")
            self._dset = self._file_ref[self._dset_name]
        return self._dset

    @property
    def shape(self):
        return self._get_dset().shape

    @property
    def dtype(self):
        return self._get_dset().dtype

    @property
    def ndim(self):
        return self._get_dset().ndim

    def __getitem__(self, key):
        dset = self._get_dset()

        if isinstance(key, tuple):
            bounding_slices = []
            in_mem_indices = []

            for k in key:
                if isinstance(k, np.ndarray):
                    flat = k.ravel()
                    if flat.size == 0:
                        bounding_slices.append(slice(0, 0))
                        in_mem_indices.append(np.array([], dtype=int))
                    else:
                        min_val, max_val = int(flat.min()), int(flat.max())
                        bounding_slices.append(slice(min_val, max_val + 1))
                        in_mem_indices.append(k - min_val)
                elif isinstance(k, (int, np.integer)):
                    bounding_slices.append(int(k))
                    in_mem_indices.append(0)
                elif isinstance(k, slice):
                    bounding_slices.append(k)
                    in_mem_indices.append(slice(None))
                elif isinstance(k, (list, tuple)):
                    arr = np.asarray(k)
                    min_val, max_val = int(arr.min()), int(arr.max())
                    bounding_slices.append(slice(min_val, max_val + 1))
                    in_mem_indices.append(arr - min_val)
                else:
                    bounding_slices.append(k)
                    in_mem_indices.append(slice(None))

            # Stream ONLY the minimal bounding slice from disk into RAM
            sub_box = dset[tuple(bounding_slices)]

            if isinstance(sub_box, np.ndarray):
                return sub_box[tuple(in_mem_indices)]
            return sub_box

        return dset[key]

class _HDF5ObjectWrapper:
    """Simple wrapper to allow attribute access (.spectra_tensor, .sp_map)."""

    pass


def load_h5_file(h5_item, lazy_threshold_mb=1.0, target_class=None):
    """Recursively converts HDF5 groups into Python dicts/lists or class instances.

    Parameters:
    -----------
    h5_item : h5py.Group or h5py.Dataset
        The target HDF5 node.
    lazy_threshold_mb : float
        Datasets larger than this size in MB are returned as lazy h5py.Dataset handles.
    target_class : type, optional
        If provided (e.g. nns.Scattered_Spectra), instantiates this class and populates
        its attributes, preserving all class methods (_resolve_indices, etc.).
    """
    # 1. Handle Datasets
    if isinstance(h5_item, h5py.Dataset):
        if h5_item.attrs.get("is_none", False):
            return None

        # Check dataset size in MB
        size_mb = (h5_item.size * h5_item.dtype.itemsize) / (1024 * 1024)

        # Leave datasets larger than threshold as lazy h5py.Dataset handles
        if size_mb >= lazy_threshold_mb:
            return _LazyHDF5TensorProxy(h5_item)

        data = h5_item[()]

        # Unpack 0D scalar object array wrappers
        if (
            isinstance(data, np.ndarray)
            and data.shape == ()
            and data.dtype.kind == "O"
        ):
            return data.item()

        return data

    # 2. Handle Groups
    elif isinstance(h5_item, h5py.Group):
        # Recursively process children
        raw_dict = {
            key: load_h5_file(
                child, lazy_threshold_mb=lazy_threshold_mb, target_class=None
            )
            for key, child in h5_item.items()
        }

        if not raw_dict:
            return {}

        # Case A: Reconstruct lists ('item_0', 'item_1', ...)
        if all(
            k.startswith("item_") and k[5:].isdigit() for k in raw_dict.keys()
        ):
            sorted_items = sorted(
                raw_dict.items(), key=lambda x: int(x[0].split("_")[1])
            )
            return [val for _, val in sorted_items]

        # Case B: Instantiate target class and attach attributes + methods
        if target_class is not None:
            obj = target_class.__new__(target_class)
            for k, v in raw_dict.items():
                setattr(obj, k, v)
            return obj

        # Case C: Standard dictionary with restored int keys
        converted_dict = {}
        for k, v in raw_dict.items():
            if k.isdigit() or (k.startswith("-") and k[1:].isdigit()):
                converted_dict[int(k)] = v
            else:
                converted_dict[k] = v

        return converted_dict

with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_50DHe3ab1.h5", "r") as f3:
    dt = f3["nesst_inputs"]["dt"][()]
    neutron_rate = f3["nesst_inputs"]["neutron_rate"][()]
    spectra = load_h5_file(f3["spectra"], target_class=nns)

Y_total = np.sum(dt[:, None] * neutron_rate.T)

e1_spectrum = nns.get_spectrum(spectra, order=1, reaction='elastic')/Y_total
c1_spectrum = nns.get_spectrum(spectra, order=1, species='C12', reaction='elastic')/Y_total
h1_spectrum = nns.get_spectrum(spectra, order=1, species='H1', reaction='elastic')/Y_total
sh1_spectrum = c1_spectrum + h1_spectrum
d1_spectrum = nns.get_spectrum(spectra, order=1, species='H2', reaction='elastic')/Y_total
he1_spectrum = nns.get_spectrum(spectra, order=1, species='He3', reaction='elastic')/Y_total
"""
np1_spectrum = nns.get_spectrum(spectra, order=1, reaction='np')/Y_total
e2_spectrum = spectra.get_spectrum(order=2, reaction='elastic')/Y_total
np2_spectrum = spectra.get_spectrum(order=2, reaction='np')/Y_total

plt.figure(dpi=200)

plt.semilogy(E_spec,I_prim, 'k', label="HELIOS 2.7 keV")
plt.semilogy(E_spec,helios_elastic, 'b', label="Yield avg elastic")
plt.semilogy(E_spec,helios_np, 'c', label="Yield avg He3 np")
plt.semilogy(E_spec,e1_spectrum, 'r--', label="Ray-traced elastic")
plt.semilogy(E_spec,np1_spectrum, 'm--', label="Ray-traced He3 np")

plt.xlim(E_spec[0],E_spec[-1]*0.8)
plt.ylim(1e-10,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend()

plt.show()
"""
plt.figure(dpi=200)

plt.semilogy(E_spec,helios_elastic, 'b', label="Yield avg elastic")
#plt.semilogy(E_spec,nC + nH, 'g', label="Yield avg shell")
plt.semilogy(E_spec,nC, 'g--', label="Yield avg C12")
plt.semilogy(E_spec,nH, 'g:', label="Yield avg H")
plt.semilogy(E_spec,nD, 'b--', label="Yield avg D")
plt.semilogy(E_spec,nHe3, 'c--', label="Yield avg He3")
plt.semilogy(E_spec,e1_spectrum, 'r', label="Ray-traced elastic")
#plt.semilogy(E_spec,sh1_spectrum, 'k', label="Ray-traced shell")
plt.semilogy(E_spec,c1_spectrum, 'k--', label="Ray-traced C12")
plt.semilogy(E_spec,h1_spectrum, 'k:', label="Ray-traced H")
plt.semilogy(E_spec,d1_spectrum, 'r--', label="Ray-traced D")
plt.semilogy(E_spec,he1_spectrum, 'm--', label="Ray-traced He3")

plt.xlim(E_spec[0],E_spec[-1]*0.8)
plt.ylim(1e-10,1e-7)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend(ncol=2,loc=9)

plt.show()
"""
plt.figure(dpi=200)

plt.semilogy(E_spec, e1_spectrum, 'r--', label="Ray-traced 1st elastic")
plt.semilogy(E_spec, np1_spectrum, 'm--', label="Ray-traced 1st He3 np")
plt.semilogy(E_spec, e2_spectrum, 'r:', label='Ray-traced 2nd elastic')
plt.semilogy(E_spec, np2_spectrum, 'm:', label='Ray-traced 2nd He3 np')

plt.xlim(E_spec[0],E_spec[-1]*0.8)
plt.ylim(1e-11,1e-7)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.legend(ncol=2,loc=9)

plt.show()

unscattereds = unscatteredc + unscatteredh - unscatteredf
u_spectrum = spectra.unscattered/Y_total
Tion6k = 6400
DDmean6k,DDstddev6k,DDvar6k = nst.DDprimspecmoments(Tion6k)
dNdE_DD6k = nst.QBrysk(E_spec,DDmean6k,DDvar6k) # Brysk shape i.e. Gaussian
I_prim6k = dNdE_DD6k

plt.figure(dpi=200)

plt.semilogy(E_spec,I_prim6k, 'k', label="NeSST 6.4 keV")
plt.semilogy(E_spec,I_prim, 'g', label="HELIOS 2.7 keV")
plt.semilogy(E_spec,unscattereds, 'b--', label="Yield avg unscattered")
plt.semilogy(E_spec,u_spectrum, 'r--', label="Ray-traced 2nd unscattered")

plt.xlim(E_spec[0]*20,E_spec[-1]*0.8)
plt.ylim(1e-6,1e-5)
plt.xlabel("Energy (eV)")
plt.ylabel("dN/dE (1/eV)")
plt.yticks(np.logspace(-6,-5, 2))
plt.yticks(np.linspace(1e-6,1e-5, 10), minor=True, labels=[])
plt.legend(loc=1)

plt.show()
"""