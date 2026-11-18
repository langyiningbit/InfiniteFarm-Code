#!/usr/bin/env python3
import argparse,json
import numpy as np
import trimesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from trimesh.visual.material import PBRMaterial

def bridge(a,b,r):
    v=b-a; d=np.linalg.norm(v)
    if d<1e-8:return None
    m=trimesh.creation.cylinder(radius=r,height=d+2*r,sections=12)
    m.apply_transform(trimesh.geometry.align_vectors([0,0,1],v/d))
    m.apply_translation((a+b)*.5)
    return m

def direction_score(chunk,a,b,diag):
    local=chunk[np.linalg.norm(chunk-a,axis=1)<diag*.025]
    if len(local)<8:return 0.
    cov=np.cov((local-local.mean(0)).T)
    eig,vec=np.linalg.eigh(cov)
    axis=vec[:,-1]
    anis=float(eig[-1]/max(eig[-2],1e-12))
    if anis<2.0:return 0.
    toward=(b-a)/max(np.linalg.norm(b-a),1e-9)
    return float(abs(np.dot(axis,toward)))

def main(args):
    scene=trimesh.load(args.input,force='scene',process=False)
    src=scene.dump(concatenate=True)
    pts=np.asarray(src.vertices,dtype=np.float32)
    low=pts.min(0); span=np.ptp(pts,axis=0);diag=float(np.linalg.norm(span))
    connections=[];stages=[];unresolved=[]
    for scale in args.voxels:
        voxel=diag*scale
        ijk=np.floor((pts-low+3*voxel)/voxel).astype(np.int32)
        shape=tuple((np.ceil(span/voxel).astype(int)+8).tolist())
        if np.prod(shape)>args.max_cells:
            stages.append({'voxel':scale,'skipped':'grid too large'});continue
        occupied=np.zeros(shape,bool);occupied[ijk[:,0],ijk[:,1],ijk[:,2]]=True
        dilated=ndi.binary_dilation(occupied,iterations=args.dilate)
        labels,n=ndi.label(dilated,structure=ndi.generate_binary_structure(3,2))
        ids=labels[ijk[:,0],ijk[:,1],ijk[:,2]]
        del labels,dilated,occupied
        counts=np.bincount(ids,minlength=n+1)
        root=int(np.argmax(counts[1:])+1)
        anchored={root}
        if connections:
            treeall=cKDTree(pts)
            for c in connections:
                for end in (c['start'],c['end']):
                    _,idx=treeall.query(end)
                    if ids[idx]:anchored.add(int(ids[idx]))
            del treeall
        eligible=[]
        for cid in np.argsort(counts[1:])[::-1]+1:
            if counts[cid]<args.min_vertices:break
            if cid in anchored:continue
            p=pts[ids==cid]
            extent=float(np.linalg.norm(np.ptp(p,axis=0)))
            if extent>=diag*args.min_extent:eligible.append((int(cid),extent))
        count_before=len(connections)
        while True:
            remaining=[(cid,ext) for cid,ext in eligible if cid not in anchored]
            if not remaining:break
            anchor=pts[np.isin(ids,list(anchored))]
            if len(anchor)>args.max_anchor_points:
                anchor=anchor[np.linspace(0,len(anchor)-1,args.max_anchor_points,dtype=int)]
            tree=cKDTree(anchor)
            best=None
            for cid,extent in remaining:
                original=pts[ids==cid]
                chunk=original
                if len(chunk)>args.max_query_points:
                    chunk=chunk[np.linspace(0,len(chunk)-1,args.max_query_points,dtype=int)]
                d,ii=tree.query(chunk,k=min(5,len(anchor)))
                d=np.atleast_2d(d) if np.ndim(d)==1 else d
                ii=np.atleast_2d(ii) if np.ndim(ii)==1 else ii
                if d.shape[0]!=len(chunk):d=d.T;ii=ii.T
                order=np.argsort(d[:,0])[:min(args.candidate_points,len(chunk))]
                chosen=None
                for j in order:
                    for k in range(d.shape[1]):
                        gap=float(d[j,k]);a=chunk[j];b=anchor[ii[j,k]]
                        if gap<diag*args.min_gap or gap>diag*args.max_gap:continue
                        direction=direction_score(chunk,a,b,diag)
                        if any(np.linalg.norm(a-np.array(c['start']))<diag*args.spacing for c in connections):continue
                        score=gap/diag*(1.0-args.direction_weight*direction)
                        if chosen is None or score<chosen[0]:chosen=(score,gap,direction,a.copy(),b.copy())
                if chosen is None:continue
                score,gap,direction,a,b=chosen
                priority=extent/diag + .05*np.log1p(counts[cid]) - .15*score
                if best is None or priority>best[0]:best=(priority,cid,extent,gap,direction,a,b)
            if best is None:break
            _,cid,extent,gap,direction,a,b=best
            anchored.add(cid)
            connections.append({'stage_voxel':scale,'cluster':cid,'vertices':int(counts[cid]),
                'extent':extent,'gap':gap,'direction_alignment':direction,
                'start':a.astype(float).tolist(),'end':b.astype(float).tolist()})
            if len(connections)%10==0:print('bridges',len(connections),flush=True)
        remaining=[{'cluster':cid,'vertices':int(counts[cid]),'extent':ext} for cid,ext in eligible if cid not in anchored]
        unresolved.append({'voxel':scale,'components':remaining})
        stages.append({'voxel':scale,'clusters':int(n),'eligible':len(eligible),
                       'new_connections':len(connections)-count_before,'unresolved':len(remaining)})
        print('stage',scale,stages[-1],flush=True)
    color=[int(v) for v in args.color]+[255]
    for i,c in enumerate(connections):
        m=bridge(np.array(c['start']),np.array(c['end']),diag*args.radius)
        if m is None:continue
        m.visual=trimesh.visual.TextureVisuals(material=PBRMaterial(name='Stem_Green_Repair',
            baseColorFactor=color,metallicFactor=0,roughnessFactor=.9,doubleSided=True))
        scene.add_geometry(m,geom_name=f'v5_bridge_{i:04d}')
    scene.export(args.output)
    with open(args.report,'w') as f:json.dump({'input':args.input,'stages':stages,
        'repairs':len(connections),'connections':connections,'unresolved':unresolved,
        'warning':'Connectivity is approximate; anatomical parent identity cannot be guaranteed.'},f,indent=2)
    print('Saved',args.output,'repairs',len(connections),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',default='stem_repaired_v5.glb')
    p.add_argument('--report',default='repair_report_v5.json')
    p.add_argument('--voxels',nargs='+',type=float,default=[.002,.0015])
    p.add_argument('--dilate',type=int,default=1);p.add_argument('--min-vertices',type=int,default=120)
    p.add_argument('--min-extent',type=float,default=.012);p.add_argument('--min-gap',type=float,default=.003)
    p.add_argument('--max-gap',type=float,default=.22);p.add_argument('--spacing',type=float,default=.008)
    p.add_argument('--radius',type=float,default=.0018);p.add_argument('--max-cells',type=int,default=65000000)
    p.add_argument('--max-anchor-points',type=int,default=80000)
    p.add_argument('--max-query-points',type=int,default=12000)
    p.add_argument('--candidate-points',type=int,default=15)
    p.add_argument('--direction-weight',type=float,default=.6)
    p.add_argument('--color',nargs=3,type=int,default=[83,125,32])
    main(p.parse_args())
