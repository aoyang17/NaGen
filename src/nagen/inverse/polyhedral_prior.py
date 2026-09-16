"""Random, parent-free orthophosphate source law for crystal Flow Matching."""
from __future__ import annotations

import torch


TETRAHEDRON = torch.tensor(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    dtype=torch.float32,
) / (3.0 ** 0.5)


def _minimum_image_distances(points, references, lattice):
    delta=points[:,None,:]-references[None,:,:]
    delta=delta-torch.floor(delta+.5)
    return torch.linalg.vector_norm(delta@lattice,dim=-1)


def _spread_points(count,lattice,device,dtype,candidates=256,generator=None):
    chosen=[]
    for _ in range(count):
        proposals=torch.rand(candidates,3,device=device,dtype=dtype,generator=generator)
        if chosen:
            refs=torch.stack(chosen)
            score=_minimum_image_distances(proposals,refs,lattice).min(1).values
            point=proposals[score.argmax()]
        else:
            point=proposals[0]
        chosen.append(point)
    return torch.stack(chosen) if chosen else torch.empty(0,3,device=device,dtype=dtype)


def polyhedral_source_frac(target_types,mask,lattice,element_to_index=None,generator=None):
    """Sample PO4 units plus Fe/Na void sites; requires exactly O=4P.

    Output ordering follows ``target_types`` solely so fixed species channels
    remain aligned. No target coordinate, target graph, or parent is consulted.
    """
    mapping=element_to_index or {'Na':1,'Fe':2,'P':3,'O':4}
    result=torch.zeros((*target_types.shape,3),device=target_types.device,dtype=lattice.dtype)
    for b in range(len(target_types)):
        indices={e:torch.where(mask[b] & (target_types[b]==i))[0] for e,i in mapping.items()}
        if len(indices['O']) != 4*len(indices['P']) or not len(indices['P']):
            raise ValueError('polyhedral prior requires nonempty orthophosphate O=4P')
        cell=lattice[b].to(result); inverse=torch.linalg.inv(cell)
        p=_spread_points(len(indices['P']),cell,result.device,result.dtype,generator=generator)
        oxy=[]
        for center in p:
            matrix=torch.randn(3,3,device=result.device,dtype=result.dtype,generator=generator)
            q,r=torch.linalg.qr(matrix)
            q=q*torch.sign(torch.diag(r)).clamp(min=-1,max=1)
            if torch.det(q)<0: q[:,0]*=-1
            directions=TETRAHEDRON.to(result)@q.T
            oxy.extend(torch.remainder(center+(1.55*directions)@inverse,1.0))
        o=torch.stack(oxy)
        # Fe: random candidates scored for an octahedral oxygen shell. Besides
        # CN and bond length, opposing/balanced directions encourage the center
        # to lie inside the oxygen polyhedron rather than beside a cluster.
        fe=[]
        for _ in range(len(indices['Fe'])):
            proposals=torch.rand(1024,3,device=result.device,dtype=result.dtype,generator=generator)
            d=_minimum_image_distances(proposals,o,cell)
            cn=((d>1.65)&(d<2.55)).sum(1)
            shell,which=torch.topk(d,6,largest=False,dim=1)
            delta=proposals[:,None,:]-o[which]
            delta=delta-torch.floor(delta+.5)
            vectors=delta@cell
            unit=vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            balance=unit.mean(1).norm(dim=-1)
            gram=unit@unit.transpose(1,2)
            tri=torch.triu_indices(6,6,1,device=result.device)
            cosine=gram[:,tri[0],tri[1]].sort(dim=1).values
            ideal=torch.tensor([-1.,-1.,-1.]+[0.]*12,device=result.device,dtype=result.dtype)
            angular=(cosine-ideal).abs().mean(1)
            bond_violation=(torch.relu(1.82-shell).square()+torch.relu(shell-2.52).square()).sum(1)
            score=((cn-6).abs().to(result)+.45*(shell-2.1).abs().mean(1)
                   +1.5*balance+.5*angular+20*bond_violation)
            if fe:
                crowd=_minimum_image_distances(proposals,torch.stack(fe),cell).min(1).values
                score=score+torch.relu(2.3-crowd)*4
            fe.append(proposals[score.argmin()])
        fe=torch.stack(fe) if fe else torch.empty(0,3,device=result.device,dtype=result.dtype)
        occupied=torch.cat((p,o,fe),dim=0)
        na=[]
        for _ in range(len(indices['Na'])):
            proposals=torch.rand(512,3,device=result.device,dtype=result.dtype,generator=generator)
            refs=torch.cat((occupied,torch.stack(na)),dim=0) if na else occupied
            clearance=_minimum_image_distances(proposals,refs,cell).min(1).values
            na.append(proposals[clearance.argmax()])
        na=torch.stack(na) if na else torch.empty(0,3,device=result.device,dtype=result.dtype)
        for element,values in [('P',p),('O',o),('Fe',fe),('Na',na)]:
            result[b,indices[element]]=values
    return result*mask.unsqueeze(-1)
