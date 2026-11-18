#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_mesh(path):
    import trimesh
    scene = trimesh.load(path, force='scene', process=False)
    parts=[]
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh=scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        parts.append(mesh)
    if not parts:
        raise RuntimeError('GLB contains no mesh')
    return trimesh.util.concatenate(parts)


def step1(args):
    import trimesh
    mesh=load_mesh(args.input)
    print('Vertices:',len(mesh.vertices),'Triangles:',len(mesh.faces),'Bounds:',mesh.bounds.tolist())
    points, _=trimesh.sample.sample_surface(mesh, args.samples)
    np.savez_compressed(args.out/'surface_points.npz', points=points.astype(np.float32))
    cloud=trimesh.points.PointCloud(points)
    cloud.export(args.out/'surface_preview.ply')
    print('Saved surface_points.npz and surface_preview.ply')


def step2(args):
    from skimage.morphology import skeletonize, ball
    points=np.load(args.out/'surface_points.npz')['points'].astype(np.float64)
    low=points.min(axis=0)-args.voxel*3
    high=points.max(axis=0)+args.voxel*3
    shape=np.ceil((high-low)/args.voxel).astype(int)+1
    print('Voxel grid shape:',shape,'total:',int(np.prod(shape)))
    if np.prod(shape)>args.max_voxels:
        raise RuntimeError('Grid too large; increase --voxel (e.g. 0.008)')
    idx=np.floor((points-low)/args.voxel).astype(int)
    occ=np.zeros(tuple(shape),dtype=bool)
    occ[idx[:,0],idx[:,1],idx[:,2]]=True
    occ=ndi.binary_dilation(occ,structure=ball(args.dilate))
    occ=ndi.binary_closing(occ,structure=ball(args.close))
    occ=ndi.binary_fill_holes(occ)
    print('Occupied voxels:',int(occ.sum()))
    skel=skeletonize(occ)
    coords=np.argwhere(skel).astype(np.int32)
    print('Skeleton voxels:',len(coords))
    np.savez_compressed(args.out/'skeleton.npz',indices=coords, origin=low, voxel=args.voxel)
    np.savez_compressed(args.out/'occupancy.npz',indices=np.argwhere(occ).astype(np.int32), origin=low, voxel=args.voxel)
    print('Saved skeleton.npz, occupancy.npz')


def make_graph(coords):
    from scipy.sparse import coo_matrix
    lookup={tuple(p):i for i,p in enumerate(coords)}
    rows=[];cols=[];vals=[]
    offsets=[(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)
             if (x,y,z)!=(0,0,0)]
    for i,p in enumerate(coords):
        for d in offsets:
            j=lookup.get((int(p[0]+d[0]),int(p[1]+d[1]),int(p[2]+d[2])))
            if j is not None and j>i:
                rows.extend((i,j));cols.extend((j,i));vals.extend((np.linalg.norm(d),)*2)
    return coo_matrix((vals,(rows,cols)),shape=(len(coords),len(coords))).tocsr()


def shortest_path(graph,start,end):
    from scipy.sparse.csgraph import dijkstra
    dist,pred=dijkstra(graph,directed=False,indices=int(start),return_predecessors=True)
    if not np.isfinite(dist[end]):return []
    path=[int(end)];cur=int(end)
    while cur!=start:
        cur=int(pred[cur]);
        if cur<0:return []
        path.append(cur)
    return path[::-1]


def step3(args):
    from scipy.sparse.csgraph import connected_components, dijkstra
    s=np.load(args.out/'skeleton.npz');coords=s['indices'];origin=s['origin'];voxel=float(s['voxel'])
    if len(coords)==0:raise RuntimeError('No skeleton voxels')
    graph=make_graph(coords)
    n,labels=connected_components(graph,directed=False)
    counts=np.bincount(labels)
    comp=np.argmax(counts)
    keep=np.flatnonzero(labels==comp)
    coords=coords[keep];graph=graph[keep][:,keep]
    xyz=origin+coords*voxel
    y=xyz[:,1]; ylo=np.quantile(y,.08);yhi=np.quantile(y,.92)
    center=np.median(xyz[y<=ylo, :][:,[0,2]],axis=0)
    bottom=np.where(y<=ylo)[0]
    start=int(bottom[np.argmin(np.linalg.norm(xyz[bottom][:,[0,2]]-center,axis=1)+2*(y[bottom]-y.min()))])
    dist=dijkstra(graph,directed=False,indices=start)
    tops=np.where(y>=yhi)[0]
    end=int(tops[np.argmax(y[tops]-0.05*dist[tops]*voxel)])
    path=shortest_path(graph,start,end)
    if not path:raise RuntimeError('Could not find main stem path')
    np.savez_compressed(args.out/'graph.npz',indices=coords,origin=origin,voxel=voxel)
    np.save(args.out/'main_path_indices.npy',np.array(path,dtype=np.int32))
    save_json(args.out/'main_stem.json',{'points':xyz[path].round(6).tolist(),'note':'Candidate main stem; verify visually'})
    print('Connected components:',n,'largest skeleton nodes:',len(coords),'main path nodes:',len(path))
    print('Saved main_stem.json, graph.npz')


def step4(args):
    from scipy.sparse.csgraph import connected_components, dijkstra
    g=np.load(args.out/'graph.npz');coords=g['indices'];origin=g['origin'];voxel=float(g['voxel'])
    xyz=origin+coords*voxel;graph=make_graph(coords)
    main=np.load(args.out/'main_path_indices.npy')
    tree=cKDTree(xyz[main]);dist_main,_=tree.query(xyz)
    lateral=np.flatnonzero(dist_main>args.stem_radius)
    sub=graph[lateral][:,lateral]
    n,labels=connected_components(sub,directed=False)
    branches=[]
    for c in range(n):
        local=np.where(labels==c)[0]
        if len(local)<args.min_branch_nodes:continue
        ids=lateral[local];pts=xyz[ids]
        lateral_extent=float(np.max(dist_main[ids]))
        if lateral_extent<args.min_branch_length:continue
        root_local=int(np.argmin(dist_main[ids]))
        sg=sub[local][:,local]
        dd=dijkstra(sg,directed=False,indices=root_local)
        tip_local=int(np.argmax(np.where(np.isfinite(dd),dd,-1)))
        path=shortest_path(sg,root_local,tip_local)
        if len(path)<3:continue
        path_xyz=pts[path]
        direction=path_xyz[-1]-path_xyz[0]
        horizontal=float(np.linalg.norm(direction[[0,2]]))
        vertical=float(abs(direction[1]))
        if horizontal/(horizontal+vertical+1e-8)<args.min_horizontal_ratio:continue
        occ=np.load(args.out/'occupancy.npz')
        occ_xyz=occ['origin']+occ['indices']*float(occ['voxel'])
        otree=cKDTree(occ_xyz)
        samples=path_xyz[::max(1,len(path_xyz)//12)]
        neighborhood=set()
        for p in samples:
            neighborhood.update(otree.query_ball_point(p,args.pca_radius))
        if len(neighborhood)>=10:
            cloud=occ_xyz[list(neighborhood)]
            eig=np.sort(np.linalg.eigvalsh(np.cov(cloud.T)))[::-1]
            planarity=float((eig[1]-eig[2])/(eig[0]+1e-10))
        else:planarity=0.
        if planarity>args.max_planarity:continue
        branches.append({'id':len(branches),'points':path_xyz.round(6).tolist(),
                         'lateral_extent':round(lateral_extent,5),
                         'horizontal_ratio':round(horizontal/(horizontal+vertical+1e-8),3),
                         'planarity':round(planarity,3)})
    save_json(args.out/'branches.json',{'branches':branches,'note':'Heuristic candidates; inspect before use'})
    print('Candidate branches:',len(branches),'saved branches.json')


def step5(args):
    data=json.loads((args.out/'branches.json').read_text())
    rng=np.random.default_rng(args.seed)
    out=[]
    for b in data['branches']:
        pts=np.asarray(b['points'],dtype=float)
        if len(pts)<2:continue
        lens=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        cumulative=np.r_[0,np.cumsum(lens)]
        if cumulative[-1]<args.min_branch_length:continue
        for frac in np.linspace(.25,.80,args.points_per_branch):
            distance=frac*cumulative[-1]
            j=int(np.clip(np.searchsorted(cumulative,distance)-1,0,len(lens)-1))
            t=(distance-cumulative[j])/max(lens[j],1e-8)
            pos=pts[j]*(1-t)+pts[j+1]*t
            tangent=pts[j+1]-pts[j];tangent/=max(np.linalg.norm(tangent),1e-8)
            out.append({'branch_id':b['id'],'position':pos.round(6).tolist(),
                        'tangent':tangent.round(6).tolist()})
    save_json(args.out/'fruit_points.json',{'coordinate_system':'GLB mesh local coordinates; verify Unity import axis and scale',
                                            'points':out})
    print('Fruit candidate points:',len(out),'saved fruit_points.json')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--step',type=int,choices=range(1,6),required=True)
    p.add_argument('--input',default='stem.glb')
    p.add_argument('--out',type=Path,default=Path('stem_analysis'))
    p.add_argument('--samples',type=int,default=250000)
    p.add_argument('--voxel',type=float,default=.006)
    p.add_argument('--max-voxels',type=int,default=5000000)
    p.add_argument('--dilate',type=int,default=1)
    p.add_argument('--close',type=int,default=1)
    p.add_argument('--stem-radius',type=float,default=.018)
    p.add_argument('--min-branch-nodes',type=int,default=6)
    p.add_argument('--min-branch-length',type=float,default=.055)
    p.add_argument('--min-horizontal-ratio',type=float,default=.5)
    p.add_argument('--pca-radius',type=float,default=.025)
    p.add_argument('--max-planarity',type=float,default=.7)
    p.add_argument('--points-per-branch',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    {1:step1,2:step2,3:step3,4:step4,5:step5}[a.step](a)
if __name__=='__main__':main()
