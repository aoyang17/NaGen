import json
import time
from collections import defaultdict
from pymatgen.core import Structure
from typing import Optional
from tqdm import tqdm  # 🌟 新增：导入进度条库

# 我的源码文件名为 core.py
from core import PolyhedronGraph_Extractor, PolyhedronGraph_TopoVisualizer

GENERIC_TMS = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", 
    "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W", 
    "La", "Lu", "Ce"
}

def analyze_tmox_dataset(jsonl_path, target_x=6, output_path: Optional[str]=None) -> None:
    print(f"🚀 开始解析数据集: {jsonl_path}")
    print(f"🔍 目标: 提取全局唯一的 TMO{target_x} 拓扑构型 (忽略畸变，泛化TM)")
    
    # 🌟 新增：预扫描文件获取总行数，让进度条能显示 100%
    print(f"📊 正在预扫描文件获取总数据量...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"✅ 共发现 {total_lines} 条数据，开始执行聚类分析...\n")

    extractor = PolyhedronGraph_Extractor(target_x=target_x)
    global_unique_pgs = []
    cluster_distribution = defaultdict(list)
    
    total_structures = 0
    total_polyhedra = 0
    start_time = time.time()

    # 🌟 新增：使用 tqdm 包装文件迭代器
    with open(jsonl_path, "r", encoding="utf-8") as f:
        pbar = tqdm(f, total=total_lines, desc="解析与聚类", unit="struct", colour="green")
        
        for line_idx, line in enumerate(pbar):
            if not line.strip():
                continue
                
            data = json.loads(line)
            mat_id = data.get("material_id", f"unknown_{line_idx}")
            
            try:
                struct = Structure.from_dict(data["structure"])
            except Exception as e:
                # 🌟 优化：使用 tqdm.write 替代 print，防止破坏进度条排版
                tqdm.write(f"⚠️ 跳过 {mat_id}: 结构解析失败 ({e})")
                continue
                
            total_structures += 1
            
            # 提取多面体
            polyhedra = extractor.extract(struct)
            total_polyhedra += len(polyhedra)
            
            # 元素泛化拦截
            for pg in polyhedra:
                if pg.center_element in GENERIC_TMS:
                    pg.label = f"TM{extractor.ligand}{pg.coord_number}"
                    pg.center_element = "TM"
                    for n, d in pg.graph.nodes(data=True):
                        if d.get("role") == "center":
                            pg.graph.nodes[n]["element"] = "TM"
                            break
            
            # 全局聚类逻辑
            for pg in polyhedra:
                is_new_topology = True
                for i, unique_pg in enumerate(global_unique_pgs):
                    if extractor._is_equivalent(pg, unique_pg, length_tol=float('inf'), angle_tol=float('inf')):
                        is_new_topology = False
                        if mat_id not in cluster_distribution[i]:
                            cluster_distribution[i].append(mat_id)
                        break
                
                if is_new_topology:
                    global_unique_pgs.append(pg)
                    cluster_distribution[len(global_unique_pgs) - 1].append(mat_id)
            
            # 🌟 新增：在进度条右侧实时显示当前发现的独特拓扑数和多面体总数
            pbar.set_postfix({
                "独特拓扑": len(global_unique_pgs), 
                "提取多面体": total_polyhedra
            })

    # 结果汇总与输出
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("🎯 分析完成!")
    print(f"总计扫描结构数: {total_structures}")
    print(f"总计提取多面体: {total_polyhedra}")
    print(f"发现独特拓扑数: {len(global_unique_pgs)}")
    print(f"耗时: {elapsed:.2f} 秒")
    print("="*50)

    for i, pg in enumerate(global_unique_pgs):
        mats = cluster_distribution[i]
        print(f"🔹 类别 {i+1}: {pg.label} | 包含材料数: {len(mats)} | 示例材料: {mats[:3]}")

    # 可视化
    if global_unique_pgs and output_path:
        print("\n🎨 正在生成独特拓扑构型的对比图...")
        from matplotlib import pyplot as plt
        
        fig = PolyhedronGraph_TopoVisualizer.plot_compare(
            global_unique_pgs, 
            ncols=min(4, len(global_unique_pgs)), 
            suptitle=f"Global Unique TM-O{target_x} Topologies (Generic TM)"
        )

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"✅ 拓扑对比图已保存至: {output_path}")

if __name__ == "__main__":
    # jsonl_file = "/home/LiuSW/work_space/struct_gallery/NaCathode/PolyAnalysis/MP_dataset_demo.jsonl"
    jsonl_file = "/home/LiuSW/work_space/struct_gallery/NaCathode/PolyAnalysis/MP_dataset_ver0_statistic.jsonl"

    target_x = 5

    analyze_tmox_dataset(
        jsonl_file, target_x=target_x, 
        output_path=f"/home/LiuSW/CodingProjects/ATLAS/results/NaCathode/work_space/testing_{target_x}.png"
        )
