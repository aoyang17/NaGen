# NOTE: 指定使用显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用显卡

# NOTE: 导入基础运行环境
import sys
from contextlib import contextmanager

# Pymatgen 相关环境
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from mp_api.client import MPRester
from monty.serialization import loadfn, dumpfn

# Atomic Simulation Environment (ASE)
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE, BFGS
from ase.filters import ExpCellFilter
from ase.io import Trajectory

# NOTE: 全局配置变量

# UMA / MP 对齐用的 U 值 (eV)
UMA_HUBBARDS = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}


# OMat24 的 POTCAR 映射
POTCAR_MAP = {
    "Ac": "Ac", "Ag": "Ag", "Al": "Al", "Ar": "Ar", "As": "As", "Au": "Au",
    "B": "B", "Ba": "Ba_sv", "Be": "Be_sv", "Bi": "Bi", "Br": "Br", "C": "C",
    "Ca": "Ca_sv", "Cd": "Cd", "Ce": "Ce", "Cl": "Cl", "Co": "Co", "Cr": "Cr_pv",
    "Cs": "Cs_sv", "Cu": "Cu_pv", "Dy": "Dy_3", "Er": "Er_3", "Eu": "Eu", "F": "F",
    "Fe": "Fe_pv", "Ga": "Ga_d", "Gd": "Gd", "Ge": "Ge_d", "H": "H", "He": "He",
    "Hf": "Hf_pv", "Hg": "Hg", "Ho": "Ho_3", "I": "I", "In": "In_d", "Ir": "Ir",
    "K": "K_sv", "Kr": "Kr", "La": "La", "Li": "Li_sv", "Lu": "Lu_3", "Mg": "Mg_pv",
    "Mn": "Mn_pv", "Mo": "Mo_pv", "N": "N", "Na": "Na_pv", "Nb": "Nb_pv",
    "Nd": "Nd_3", "Ne": "Ne", "Ni": "Ni_pv", "Np": "Np", "O": "O", "Os": "Os_pv",
    "P": "P", "Pa": "Pa", "Pb": "Pb_d", "Pd": "Pd", "Pm": "Pm_3", "Pr": "Pr_3",
    "Pt": "Pt", "Pu": "Pu", "Rb": "Rb_sv", "Re": "Re_pv", "Rh": "Rh_pv",
    "Ru": "Ru_pv", "S": "S", "Sb": "Sb", "Sc": "Sc_sv", "Se": "Se", "Si": "Si",
    "Sm": "Sm_3", "Sn": "Sn_d", "Sr": "Sr_sv", "Ta": "Ta_pv", "Tb": "Tb_3",
    "Tc": "Tc_pv", "Te": "Te", "Th": "Th", "Ti": "Ti_pv", "Tl": "Tl_d",
    "Tm": "Tm_3", "U": "U", "V": "V_pv", "W": "W_sv", "Xe": "Xe", "Y": "Y_sv",
    "Yb": "Yb_3", "Zn": "Zn", "Zr": "Zr_sv",
    }

_MP_ENTRIES_CACHE: dict = {}
_COMPAT = MaterialsProject2020Compatibility()

@contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def _get_mp_entries(chemsys: str, elements: list[str], mp_api_key: str, cache_dir: str):
    """ 按化学体系取 MP entries: 内存 -> 硬盘 -> 联网 """
    if chemsys in _MP_ENTRIES_CACHE:
        return _MP_ENTRIES_CACHE[chemsys]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{chemsys}.json")
    if os.path.exists(cache_path):
        entries = loadfn(cache_path)
    else:
        with MPRester(mp_api_key, use_progress_bar=False) as mpr:
            with _suppress_output():
                entries = mpr.get_entries_in_chemsys(elements)
        dumpfn(entries, cache_path)
    _MP_ENTRIES_CACHE[chemsys] = entries
    return entries

def calculate_energy_above_hull(
    structure: Structure,
    energy_per_atom: float,
    mp_api_key: str | None = None,
    cache_dir: str = "./phase_diagram_cache",
    allow_negative: bool = False,
    ) -> float:
    """
    相对 Materials Project 凸包计算 E_hull (eV/atom)。
    流程与 notebook 中 calculate_corrected_energy_above_hull 单条记录一致：
    构造 GGA/GGA+U ComputedEntry -> MP2020 能量修正 -> MP 相图 -> get_e_above_hull。
    
    Args:
        structure: 优化后的 pymatgen Structure（通常来自 ASE 回转）。
        energy_per_atom: 对应总能量 / 原子数，单位 eV/atom。
        mp_api_key: MP API Key；默认读环境变量 MP_API_KEY。
        cache_dir: 按化学体系缓存 MP entries 的目录。
        allow_negative: 若 entry 低于 MP 凸包，是否返回负值；默认截断为 0。
    
    Returns:
        energy above hull，单位 eV/atom。
    """

    mp_api_key = mp_api_key or os.environ.get("MP_API_KEY")
    if not mp_api_key:
        raise ValueError("请传入 mp_api_key，或设置环境变量 MP_API_KEY")
    composition = structure.composition
    total_energy = float(energy_per_atom) * composition.num_atoms
    current_hubbards = {
        el.symbol: UMA_HUBBARDS[el.symbol]
        for el in composition.elements
        if el.symbol in UMA_HUBBARDS
        }
    potcar_symbols = [
        f"PAW_PBE {POTCAR_MAP.get(el.symbol, el.symbol)}"
        for el in composition.elements
        ]
    parameters = {
        "run_type": "GGA+U" if current_hubbards else "GGA",
        "hubbards": current_hubbards,
        "potcar_symbols": potcar_symbols,
        }
    raw_entry = ComputedEntry(composition, total_energy, parameters=parameters)
    corrected_entry = _COMPAT.process_entry(raw_entry) or raw_entry
    elements = [el.symbol for el in composition.elements]
    mp_entries = _get_mp_entries(composition.chemical_system, elements, mp_api_key, cache_dir)
    pd = PhaseDiagram(mp_entries)

    return pd.get_e_above_hull(corrected_entry, allow_negative=allow_negative)

def universal_struct_optimizer(
    atoms: Atoms, calculator: Calculator, relax_cell: bool = True, 
    fmax_coarse: float=0.05, max_steps_coarse: int=1000, fmax_fine: float=0.01, max_steps_fine: int=1000,
    output_dir: str="./", system_id: str="Untitled"
    ) -> Atoms:
    """ 通用的结构优化函数，自适应 0D/1D/2D/3D 材料 """

    traj_path = os.path.join(output_dir, f"traj_{system_id}.traj")
    log_fire = os.path.join(output_dir, f"opt_stage1_FIRE_{system_id}.log")
    log_bfgs = os.path.join(output_dir, f"opt_stage2_BFGS_{system_id}.log")

    atoms.calc = calculator
    traj = Trajectory(traj_path, 'w', atoms)

    # 阶段一：晶胞与原子位置联合优化
    if relax_cell and any(atoms.pbc):
        p = atoms.pbc
        dynamic_mask = [
            p[0], p[1], p[2], 
            p[1] and p[2], p[0] and p[2], p[0] and p[1]
        ]
        cell_filter = ExpCellFilter(atoms, mask=dynamic_mask)
        opt_cell = FIRE(cell_filter, logfile=log_fire)
        opt_cell.attach(traj.write, interval=1)
        opt_cell.run(fmax=fmax_coarse, steps=max_steps_coarse)

    # 阶段二：高精度原子位置优化
    opt_pos = BFGS(atoms, logfile=log_bfgs)
    opt_pos.attach(traj.write, interval=1)
    opt_pos.run(fmax=fmax_fine, steps=max_steps_fine)

    return atoms

if __name__ == "__main__":
    import torch
    from ase.build import bulk
    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor
    from fairchem.core import FAIRChemCalculator

    # ---------- API_KEY & 各种目录及模型路径设定 ----------
    MP_API_KEY = "shSAwXiyVy1OGTyCbjzV69mNURCAk4gG"

    UMA_MODEL = "/home/LiuSW/work_space/models/MLFF_n_MLIP/UMA/uma-m-1p1.pt"
    WORK_SPACE = "/home/LiuSW/work_space/shared_space/NaCathode/CodingSpace/work_space"
    PD_CACHE_DIR = f"{WORK_SPACE}/phase_diagram_cache"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 1. 构建金刚石 Si 用于测试 (晶格略大于实验值，让弛豫有位移) ----------
    SYSTEM_ID = "Si"
    
    atoms = bulk("Si", "diamond", a=5.50)
    write(os.path.join(WORK_SPACE, "Si_init.cif"), atoms)

    print(f"初始结构已保存: {os.path.join(WORK_SPACE, 'Si_init.cif')}")
    print(f"formula={atoms.get_chemical_formula()}, natoms={len(atoms)}, cell={atoms.cell.lengths()}")

    # ---------- 2. UMA 结构优化 ----------
    print(f"初始化 UMA 结构优化器, 模型路径: {UMA_MODEL}, 设备: {DEVICE}")

    calculator = FAIRChemCalculator.from_model_checkpoint(UMA_MODEL, task_name="omat", device=DEVICE)  # 初始化 UMA 结构优化器

    opt_atoms = universal_struct_optimizer(
        atoms=atoms,
        calculator=calculator,
        relax_cell=True,
        fmax_coarse=0.05,
        max_steps_coarse=1500,
        fmax_fine=0.01,
        max_steps_fine=1500,
        output_dir=WORK_SPACE,
        system_id=SYSTEM_ID
        )

    write(os.path.join(WORK_SPACE, "Si_opt.cif"), opt_atoms)

    energy_per_atom = opt_atoms.get_potential_energy() / len(opt_atoms)
    opt_structure = AseAtomsAdaptor.get_structure(opt_atoms)

    print(f"优化结构已保存: {os.path.join(WORK_SPACE, 'Si_opt.cif')}")
    print(f"E/atom = {energy_per_atom:.6f} eV")
    print(f"轨迹/日志: traj_{SYSTEM_ID}.traj, opt_stage1_FIRE_{SYSTEM_ID}.log, opt_stage2_BFGS_{SYSTEM_ID}.log")

    # ---------- 3. 相对 MP 凸包计算 E_hull ----------
    ehull = calculate_energy_above_hull(
        structure=opt_structure,
        energy_per_atom=energy_per_atom,
        mp_api_key=MP_API_KEY,
        cache_dir=PD_CACHE_DIR,
        allow_negative=False
        )
    
    print(f"E_hull = {ehull:.6f} eV/atom  ({ehull * 1000:.2f} meV/atom)")
    print(f"MP 相图缓存: {os.path.join(PD_CACHE_DIR, 'Si.json')}")