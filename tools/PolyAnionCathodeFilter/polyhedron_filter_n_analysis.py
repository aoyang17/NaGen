# NOTE: 项目路径设定
import os, sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_dir = Path(str(current_dir)[:str(current_dir).find("ATLAS") + len("ATLAS")])
sys.path.append(str(project_dir))

# NOTE: 导入基础运行环境
from collections import Counter
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

# NOTE: 导入自定义运行环境
from src.analysis_poly import species_string, PolyhedraRecipe
from src.analysis_poly.configuration import Configuration
from src.analysis_poly.find_poly import in_hull

# NOTE: 全局配置变量

# 过渡金属元素
TM = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "La", "Lu", "Ce"}
P_LIKE = {"P"}  # 磷族元素
ALLOWED = {"Na"} | TM | P_LIKE | {"O"}  # 允许的元素
TMO_CNS = (4, 5, 6)  # 过渡金属氧化物的配位数

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

_MP_ENTRIES_CACHE: dict = {}  # MP 凸包缓存
_COMPAT = MaterialsProject2020Compatibility()  # MP 能量修正兼容器

# -------------------------------- 多面体规则过滤 & 统计模块 --------------------------------

def _build_recipes(composition) -> list[PolyhedraRecipe]:
    recipes = []
    for center, cutoff in [(c, 2.0) for c in P_LIKE] + [(c, 2.5) for c in TM]:
        if center in composition:
            recipes.append(PolyhedraRecipe(
                method="distance cutoff",
                coordination_cutoff=cutoff,
                central_atoms=center,
                vertex_atoms=["O", "F"],
            ))
    return recipes

def _detect_polyhedra(structure, recipes, vol_th=0.5, center_in_hull=False):
    """ 与 poly_embed.passes_polyhedron_rules 默认行为对齐 (center_in_hull=False) """
    kept = []
    for poly in Configuration(structure=structure, recipes=recipes).polyhedra:
        if poly.coordination_number < 4 or poly.volume < vol_th:
            continue
        if center_in_hull and not in_hull(
            poly.central_atom.site.coords, [v.site.coords for v in poly.vertices]
        ):
            continue
        kept.append(poly)
    return kept

def _poly_str(poly) -> str:
    c = species_string(poly.central_atom.site)
    v_cnt = Counter(species_string(v.site) for v in poly.vertices)
    return c + "".join(f"{k}{v_cnt[k]}" for k in sorted(v_cnt))

def analyze_polyhedra(structure: Structure, vol_th: float = 0.5) -> dict:
    """
    过滤多面体规则，并统计纯氧配位的 TMO4/5/6
    
    Returns:
        {"passed": bool, "reason": str, "num_TMO4": int, "num_TMO5": int, "num_TMO6": int}
        未通过时 counts 仍尽量给出 (便于排查), reason 为失败原因
    """
    counts = {f"num_TMO{n}": 0 for n in TMO_CNS}
    elems = {species_string(s) for s in structure.sites}
    extra = elems - ALLOWED

    if extra:
        return {"passed": False, "reason": f"Invalid elements: {extra}", **counts}
    
    
    try:
        polyhedra = _detect_polyhedra(structure, _build_recipes(structure.composition), vol_th)
    except Exception as e:
        return {"passed": False, "reason": f"Error in polyhedron detection: {e}", **counts}
    
    if not polyhedra:
        return {"passed": False, "reason": "No polyhedra detected", **counts}
    
    for poly in polyhedra:
        c = species_string(poly.central_atom.site)
        cn = poly.coordination_number
        if c in TM and not (4 <= cn <= 8):
            return {"passed": False, "reason": f"Invalid TM coordination: {_poly_str(poly)}", **counts}
        if c in P_LIKE:
            if cn != 4:
                return {"passed": False, "reason": f"Invalid {c} coordination: {_poly_str(poly)}", **counts}
            if any(species_string(v.site) == "F" for v in poly.vertices):
                return {"passed": False, "reason": f"Invalid polyhedron ({c}-F): {_poly_str(poly)}", **counts}
        if c not in (TM | P_LIKE):
            return {"passed": False, "reason": f"Invalid central atom: {_poly_str(poly)}", **counts}
    
    target = {i for i, s in enumerate(structure.sites) if species_string(s) in ALLOWED - {"Na"}}
    covered = set()
    for poly in polyhedra:
        covered.add(poly.central_atom.index)
        covered.update(v.index for v in poly.vertices)
    
    if target != covered:
        free = [f"{species_string(structure.sites[i])}@{i}" for i in sorted(target - covered)]
        return {"passed": False, "reason": f"Free atoms not in any polyhedra: {', '.join(free)}", **counts}
    
    for poly in polyhedra:
        if species_string(poly.central_atom.site) not in TM:
            continue
        if not all(species_string(v.site) == "O" for v in poly.vertices):
            continue
        key = f"num_TMO{poly.coordination_number}"
        if key in counts:
            counts[key] += 1
    return {"passed": True, "reason": "", **counts}

# -------------------------------- 相对 MP 凸包计算 E_hull 模块 --------------------------------

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

# -------------------------------- 通用结构优化模块 --------------------------------

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

# -------------------------------- 命令行解析器模块 --------------------------------

if __name__ == "__main__":
    """ 主函数 """