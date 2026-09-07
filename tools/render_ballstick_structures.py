from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Lattice, Structure

COLORS={"Na":"#8e63c7","Fe":"#2f6fb0","P":"#f0a33b","O":"#e46b6b"}
SIZES={"Na":260,"Fe":180,"P":170,"O":95}
RADII={"Na":1.02,"Fe":0.76,"P":0.70,"O":0.66}

def bond_cut(a,b):
    s={a,b}
    if s=={"P","O"}: return 2.05
    if s=={"Fe","O"}: return 2.55
    if s=={"O"}: return 2.15
    return 2.35

def render(s, out, title):
    fig=plt.figure(figsize=(7,7), facecolor="white"); ax=fig.add_subplot(111,projection="3d")
    lat=np.asarray(s.lattice.matrix); frac=np.asarray(s.frac_coords); cart=np.asarray(s.cart_coords); elems=[str(x.specie) for x in s]
    # periodic bonds, including bonds crossing the displayed unit-cell boundary
    for i in range(len(s)):
      for j in range(i+1,len(s)):
        for shift in np.ndindex(3,3,3):
          sh=np.asarray(shift)-1
          if np.all(sh==0) or True:
            d=np.linalg.norm((frac[j]+sh-frac[i])@lat)
            if d<=bond_cut(elems[i],elems[j]) and d>1e-5:
              q=cart[j]+sh@lat
              # only draw each central-cell bond once, and boundary bonds once
              if np.all(sh==0) or (i==0 and j==1):
                ax.plot([cart[i,0],q[0]],[cart[i,1],q[1]],[cart[i,2],q[2]],c="#777777",lw=1.0,alpha=.65)
    for e,c in COLORS.items():
      ids=[i for i,x in enumerate(elems) if x==e]
      if ids: ax.scatter(cart[ids,0],cart[ids,1],cart[ids,2],s=SIZES[e],c=c,edgecolors="black",linewidths=.5,depthshade=True,label=e)
    # unit-cell edges
    corners=[np.array([i,j,k])@lat for i in (0,1) for j in (0,1) for k in (0,1)]
    for i in (0,1):
      for j in (0,1):
       for k in (0,1):
        p=np.array([i,j,k])@lat
        for axis in range(3):
          if [i,j,k][axis]==0:
            q=np.array([i,j,k],float); q[axis]=1
            q=q@lat; ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],c="#222222",lw=.9,alpha=.8)
    ax.set_title(title,fontsize=10); ax.set_axis_off(); ax.view_init(20,35); ax.legend(loc="upper left",fontsize=8)
    fig.tight_layout(); fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)

def main():
  p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--out",required=True); a=p.parse_args(); d=json.load(open(a.input)); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
  for r in d["records"]:
    s=Structure(Lattice(r["lattice"]),r["elements"],r["frac_coords"])
    render(s,out/f"shooting_{int(r['sample_id']):02d}_ballstick.png",f"#{r['sample_id']} {r['formula']}  E_hull={r['optimized_e_hull_surrogate']:.4f} eV/atom")
if __name__=="__main__": main()
