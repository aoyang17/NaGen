from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
from pymatgen.core import Lattice, Structure

COLORS = {"Na":"#6baed6", "Fe":"#d62728", "P":"#ffbf00", "O":"#7f7f7f"}

def draw(structure, path, title):
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    xyz = structure.cart_coords
    for e in COLORS:
        sel = [i for i, s in enumerate(structure) if str(s.specie) == e]
        if sel:
            ax.scatter(xyz[sel,0], xyz[sel,1], xyz[sel,2], s=35 if e!="O" else 18,
                       c=COLORS[e], label=e, edgecolors="black", linewidths=.25)
    origin = structure.lattice.matrix[0]*0
    vecs = structure.lattice.matrix
    for i in range(3):
        ax.plot([0,vecs[i,0]],[0,vecs[i,1]],[0,vecs[i,2]], c="black", lw=1)
    ax.set_title(title, fontsize=9); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.legend(loc="upper left", fontsize=8); ax.view_init(22, 35); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--out", required=True); a=p.parse_args()
    data=json.load(open(a.input)); out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for r in data["records"]:
        s=Structure(Lattice(r["lattice"]), r["elements"], r["frac_coords"])
        stem=f"shooting_{int(r['sample_id']):02d}"
        s.to(filename=str(out/(stem+".cif")))
        draw(s, out/(stem+".png"), f"{stem}  {r['formula']}  surrogate Ehull={r['optimized_e_hull_surrogate']:.4f}")
        rows.append((stem, r["formula"], r["optimized_e_hull_surrogate"], r["analytic_constraint_checks"]["feasible"]))
    fig, axes=plt.subplots(3,4, figsize=(16,12), subplot_kw={"projection":"3d"})
    for ax,r in zip(axes.flat,data["records"]):
        xyz=Structure(Lattice(r["lattice"]),r["elements"],r["frac_coords"]).cart_coords
        for e,c in COLORS.items():
            sel=[i for i,x in enumerate(r["elements"]) if x==e]
            if sel: ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],s=20 if e!="O" else 10,c=c,label=e)
        ax.set_title(f"#{r['sample_id']} {r['formula']}\nEhull={r['optimized_e_hull_surrogate']:.4f}",fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(22,35)
    handles=[plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,label=e,markersize=7) for e,c in COLORS.items()]
    fig.legend(handles=handles, labels=list(COLORS), loc="lower center", ncol=4); fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(out/"shooting_12_overview.png",dpi=180); plt.close(fig)
    with open(out/"index.tsv","w") as f:
        f.write("id\tformula\tsurrogate_Ehull\tall_constraints\n")
        for row in rows: f.write("\t".join(map(str,row))+"\n")
if __name__=="__main__": main()
