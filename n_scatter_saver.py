import numpy as np
import helios_parser as hp
import rhoR_ray_calculator as rhr
import nesst_n_scatter as nns
from pathlib import Path
import os
import h5py
import itertools

input_folder = Path("DHe3ab").resolve()
output_folder = Path("C:/Users/Raphael/Desktop/NeSST output").resolve()

def _validate_item(path, value):
    """Pre-validates that all objects in the data structure can be serialized.

    Only inspects shapes, dtypes, and attribute names without loading array
    data into RAM. Fails loudly on invalid types or empty custom objects.
    """
    if value is None:
        return

    # 1. Primitive scalar types
    if isinstance(
        value, (int, float, str, bool, bytes, np.number, np.bool_)
    ):
        return

    # 2. Dictionaries
    if isinstance(value, dict):
        for k, v in value.items():
            _validate_item(f"{path}/{k}", v)
        return

    # 3. Lists and Tuples
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _validate_item(f"{path}[{idx}]", item)
        return

    # 4. Array-like objects (NumPy, Zarr, HDF5 datasets)
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        return

    # 5. Custom class instances
    if hasattr(value, "__dict__"):
        public_attrs = {
            k: v for k, v in vars(value).items() if not k.startswith("_")
        }

        if not public_attrs:
            raise TypeError(
                f"Validation Failed at '{path}': Object of type '{type(value).__name__}' "
                f"has no public attributes to serialize."
            )

        for k, v in public_attrs.items():
            _validate_item(f"{path}.{k}", v)
        return

    # Loud failure on unhandled types
    raise TypeError(
        f"Validation Failed at '{path}': Unsupported data type '{type(value).__module__}.{type(value).__name__}'."
    )


def _write_item(h5_group, key, value):
    """Recursively writes items to HDF5.

    Streams Zarr arrays chunk-by-chunk to avoid high RAM usage.
    """
    string_dt = getattr(h5py, "string_dtype")(encoding="utf-8")

    # 1. Handle None / Null values
    if value is None:
        dset = h5_group.create_dataset(str(key), dtype=string_dt)
        dset.attrs["is_none"] = True
        return

    # 2. Handle NumPy and Zarr / Array-Like Objects
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        shape = value.shape
        dtype = value.dtype
        arr_size = int(np.prod(shape)) if len(shape) > 0 else 1

        # 2a. 0D Scalar Arrays
        if len(shape) == 0:
            val_scalar = value[()]
            if dtype.kind in ("U", "S"):
                h5_group.create_dataset(
                    str(key), data=str(val_scalar), dtype=string_dt
                )
            else:
                h5_group.create_dataset(str(key), data=val_scalar)
            return

        # 2b. Zero-size Arrays
        if arr_size == 0:
            h5_group.create_dataset(
                str(key),
                shape=shape,
                dtype=dtype if dtype.kind != "O" else string_dt,
            )
            return

        # 2c. Direct Write for standard NumPy Arrays (already in RAM)
        if isinstance(value, np.ndarray):
            if dtype.kind in ("U", "S"):
                converted = value.astype(string_dt)
                h5_group.create_dataset(
                    str(key), data=converted, compression="gzip"
                )
            elif dtype.kind == "O":
                is_all_str = all(
                    isinstance(x, (str, bytes)) or x is None for x in value.flat
                )
                if is_all_str:
                    cleaned = np.array(
                        [str(x) if x is not None else "" for x in value.flat],
                        dtype=object,
                    ).reshape(shape)
                    h5_group.create_dataset(
                        str(key),
                        data=cleaned,
                        dtype=string_dt,
                        compression="gzip",
                    )
                else:
                    subgroup = h5_group.require_group(str(key))
                    for idx, item in enumerate(value.flat):
                        _write_item(subgroup, f"item_{idx}", item)
            else:
                try:
                    h5_group.create_dataset(
                        str(key), data=value, compression="gzip"
                    )
                except Exception:
                    h5_group.create_dataset(str(key), data=value)
            return

        # 2d. Chunked Streaming for Zarr Arrays (Streams piece-by-piece to disk)
        raw_chunks = getattr(value, "chunks", None)
        if raw_chunks and isinstance(raw_chunks, tuple):
            h5_chunks = tuple(min(c, s) for c, s in zip(raw_chunks, shape))
        else:
            h5_chunks = True  # Fallback to h5py automatic chunking

        target_dtype = string_dt if dtype.kind in ("U", "S", "O") else dtype

        # Initialize empty dataset on disk
        dset = h5_group.create_dataset(
            str(key),
            shape=shape,
            dtype=target_dtype,
            chunks=h5_chunks,
            compression="gzip",
        )

        # Slice shape configuration matching Zarr's chunking structure
        if raw_chunks and isinstance(raw_chunks, tuple):
            slice_chunks = tuple(min(c, s) for c, s in zip(raw_chunks, shape))
        else:
            step_axis0 = max(1, min(shape[0], 1000))
            slice_chunks = (step_axis0,) + shape[1:]

        dim_slices = [
            [
                slice(start, min(start + c_size, dim_size))
                for start in range(0, dim_size, c_size)
            ]
            for dim_size, c_size in zip(shape, slice_chunks)
        ]

        # Read each chunk from Zarr and write directly into HDF5 slice
        for slice_tuple in itertools.product(*dim_slices):
            chunk_data = np.ascontiguousarray(value[slice_tuple])
            if dtype.kind in ("U", "S", "O"):
                chunk_data = chunk_data.astype(string_dt)
            dset[slice_tuple] = chunk_data
        return

    # 3. Handle Custom Class Instances
    if hasattr(value, "__dict__"):
        subgroup = h5_group.require_group(str(key))
        for attr_name, attr_val in vars(value).items():
            if not attr_name.startswith("_"):
                _write_item(subgroup, attr_name, attr_val)
        return

    # 4. Handle Dictionaries
    if isinstance(value, dict):
        subgroup = h5_group.require_group(str(key))
        for k, v in value.items():
            _write_item(subgroup, k, v)
        return

    # 5. Handle Lists and Tuples
    if isinstance(value, (list, tuple)):
        try:
            arr = np.array(value)
            if arr.dtype.kind in ("U", "S", "i", "f", "b"):
                _write_item(h5_group, key, arr)
                return
        except Exception:
            pass

        subgroup = h5_group.require_group(str(key))
        for idx, item in enumerate(value):
            _write_item(subgroup, f"item_{idx}", item)
        return

    # 6. Handle Primitive Scalars
    if isinstance(value, (int, float, str, bool, bytes, np.number, np.bool_)):
        if isinstance(value, str):
            h5_group.create_dataset(str(key), data=value, dtype=string_dt)
        else:
            h5_group.create_dataset(str(key), data=value)
        return

    raise TypeError(
        f"Failed to write '{key}'. Unsupported data type: {type(value)}"
    )


def save_variables_to_hdf5(input_path, output_dir, **variables):
    """Pre-validates inputs, then writes all variables into an HDF5 file."""
    input_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. PRE-VALIDATE: Check all variables before touching the disk or writing huge datasets
    print("Pre-validating data structures...")
    for var_name, var_value in variables.items():
        _validate_item(var_name, var_value)
    print("Pre-validation successful.")

    # 2. WRITE TO DISK: Execute HDF5 writes using input file stem
    output_filepath = out_dir / f"{input_p.stem}.h5"
    with h5py.File(output_filepath, "w") as h5file:
        for var_name, var_value in variables.items():
            _write_item(h5file, var_name, var_value)

    print(f"Successfully saved HDF5 to: {output_filepath}")

def patch_spectra_in_hdf5(hdf5_filepath, spectra_object):
    """Patches an existing HDF5 file by adding or updating the 'spectra' dataset/group."""
    # 1. Pre-validate the spectra object before touching disk
    print("Pre-validating spectra object...")
    _validate_item("spectra", spectra_object)
    print("Validation passed.")

    # 2. Open file in append/update mode
    with h5py.File(hdf5_filepath, "a") as h5file:
        # Delete the old/stringified node if it exists
        if "spectra" in h5file:
            print("Removing existing 'spectra' entry from HDF5...")
            del h5file["spectra"]

        # Write the new structured object using your updated helper
        print("Writing updated 'spectra' group to HDF5...")
        _write_item(h5file, "spectra", spectra_object)

    print(f"Successfully patched {hdf5_filepath}")

def load_h5_structure(h5_item):
    """Recursively converts an HDF5 group or dataset back into native Python types,
    restoring integer dictionary keys and indexed lists.
    """
    # 1. Handle Datasets
    if isinstance(h5_item, h5py.Dataset):
        if h5_item.attrs.get("is_none", False):
            return None
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
        raw_dict = {key: load_h5_structure(child) for key, child in h5_item.items()}
        if not raw_dict:
            return {}
        # Case A: Check if group represents a reconstructed list ('item_0', 'item_1', ...)
        if all(
            k.startswith("item_") and k[5:].isdigit() for k in raw_dict.keys()
        ):
            sorted_items = sorted(
                raw_dict.items(), key=lambda x: int(x[0].split("_")[1])
            )
            return [val for _, val in sorted_items]
        # Case B: Standard dictionary — convert stringified integer keys back to ints
        converted_dict = {}
        for k, v in raw_dict.items():
            if k.isdigit() or (k.startswith("-") and k[1:].isdigit()):
                converted_dict[int(k)] = v
            else:
                converted_dict[k] = v
        return converted_dict

for file_path in output_folder.glob("*.h5"):
    if file_path.is_file():
        with h5py.File(file_path, "r") as f:
            E_spec = f["E_spec"][()]
            I_prim = f["I_prim"][()]
            nesst_inputs = load_h5_structure(f["nesst_inputs"])
        spectra = nns.calculate_nesst_n_scatter(
            nesst_inputs, E_spec, I_prim, max_scatter_order=2
        )
        # 3. Patch the file (passing file_path, NOT output_folder)
        patch_spectra_in_hdf5(
            hdf5_filepath=file_path,
            spectra_object=spectra,
        )

"""
for file in input_folder.iterdir():
    if file.is_file():
        os.chdir(input_folder)
        try:
            file = file.name
            data = hp.get_helios_2d_arrays(file)
            Tion = data["Tion"]  # eV, read from HELIOS
            neutron_rate = data["neutron_rate"]
            E_spec = np.linspace(1.0e5, 4.0e6, 500)
            I_prim = hp.I_prim_integrator(Tion, neutron_rate, E_spec)
            nesst_inputs = rhr.extract_helios_rhoL(file)
            spectra = nns.calculate_nesst_n_scatter(nesst_inputs, E_spec, I_prim, max_scatter_order=2)

            save_variables_to_hdf5(
                input_path=file,
                output_dir=output_folder,
                data=data,
                E_spec=E_spec,
                I_prim=I_prim,
                nesst_inputs=nesst_inputs,
                spectra=spectra,
            )
        finally:
            os.chdir("..")
"""