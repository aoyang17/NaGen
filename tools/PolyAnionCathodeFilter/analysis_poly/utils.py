import os, sys
import numpy as np
from ast import literal_eval
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import warnings
from contextlib import contextmanager
import ase
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.core.sites import Site
import pandas as pd
from tqdm import tqdm


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_cs_from_row(row) -> set:
    if "elements" in row:
        elements = row["elements"]
    elif "chemical_system" in row:
        elements = row["chemical_system"]
    else:
        return None
    
    if "[" in elements:    # "['Ho', 'Pd']"
        elements = literal_eval(elements)

    if isinstance(elements, str):       # Ho-Pd
        elements = elements.split("-")
    elif isinstance(elements, list):  # ['Ho', 'Pd']
        pass
    else:
        return None
    
    return set(elements)

def load_structures(load_path: str, least_chem_sys: set[str] | None = None) -> tuple[list[str], list[Structure]]:
    '''
    - load_path: Path to the file or directory containing the structures.
      supported formats:
        1. Single CIF file: "path/to/structure.cif"
        2. CSV file with a 'cif' column: "path/to/structures.csv"
        3. POSCAR or CONTCAR file: "path/to/POSCAR"
        4. Directory containing multiple CIF files: "path/to/cif_directory/"
        5. Directory containing multiple XYZ or EXTXYZ files: "path/to/xyz_directory"
    - least_chem_sys: If provided, only structures whose chemical system includes all elements in this set will be loaded.
    '''
    if os.path.isfile(load_path):
        if load_path.endswith("cif"):
            with HiddenPrints():
                    structure = Structure.from_file(load_path)
            structures = [structure]
            struct_ids = [os.path.basename(load_path)]
        elif load_path.endswith("csv"):
            df = pd.read_csv(load_path)
            structures = []
            struct_ids = []
            for idx, row in tqdm(df.iterrows(), total = len(df), desc = "Loading structures", leave = False):
                if least_chem_sys is not None:
                    cs = get_cs_from_row(row)
                    if cs is not None and not (cs >= least_chem_sys):
                        continue
                struct_str = row['cif']
                try:
                    with HiddenPrints():
                        structure = Structure.from_str(struct_str, fmt = "cif")
                    structures.append(structure)
                    struct_ids.append(row.get('material_id', f'structure_{idx}'))
                except Exception as e:
                    warnings.warn(f"Failed to parse structure at index {idx}: {e}")
        else:
            try:
                poscar = Poscar.from_file(load_path)
                structures = [poscar.structure]
                struct_ids = [os.path.basename(load_path)]
            except:
                raise ValueError(f"Unsupported file type: {load_path}")
    else:
        structures = []
        struct_ids = []
        for filename in os.listdir(load_path):
            if filename.endswith(".cif"):
                try:
                    with HiddenPrints():
                        structures.append(Structure.from_file(f"{load_path}/{filename}"))
                        struct_ids.append(filename)
                except ValueError as e:
                    warnings.warning(f"Failed to read {filename} as a CIF file: {e}")
            elif filename.endswith(".extxyz") or filename.endswith(".xyz"):
                ase_atoms = ase.io.read(
                    f"{load_path}/{filename}", index = ":", format = 'extxyz'
                ) 
                with HiddenPrints():
                    for id, ase_atom in enumerate(ase_atoms):
                        structures.append(AseAtomsAdaptor.get_structure(ase_atom))
                        struct_ids.append(f"{filename}_{id}")

    if least_chem_sys is not None:
        ret_structures = []
        ret_struct_ids = []
        for struct_id, structure in zip(struct_ids, structures):
            if not (set([el.name for el in structure.composition.elements]) >= least_chem_sys):
                continue
            ret_structures.append(structure)
            ret_struct_ids.append(struct_id)
    else:
        ret_structures = structures
        ret_struct_ids = struct_ids

    return ret_struct_ids, ret_structures


def species_string(site: Site) -> str:
    return ", ".join(sp.symbol for sp in sorted(site.species))


def write_cif(structure: Structure, filename: str) -> None:
    CifWriter(structure, symprec=0.1).write_file(filename)