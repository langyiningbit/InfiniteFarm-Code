#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, types
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
import geometry_analysis as ga
import repair_engine as re
import distribution_model as dm
import spatial_estimator as se
from scipy import ndimage as ndi

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def meshes(scene):
    return [m.copy() for m in scene.dump(concatenate=False) if isinstance(m,trimesh.Trimesh)]

def bounds(parts):
    return np.array([np.min([m.bounds[0] for m in parts],axis=0),np.max([m.bounds[1] for m in parts],axis=0)])

def load_measurements(path, crop, stage):
    if not Path(path).exists(): return [], None
    with open(path,newline='',encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r.get('crop')==crop and r.get('stage')==stage]
    if not rows: return [],None
    plants={}
    for r in rows:
        pid=r.get('plant_id','')
        if not pid:raise ValueError('plant_id required')
        if pid in plants:
            for k in ('plant_height_m','stem_diameter_m','fruit_diameter_m','fruit_count'):
                if r.get(k)!=plants[pid].get(k):raise ValueError(f'Inconsistent {k} for {pid}')
        else:plants[pid]=r
    for pid,r in plants.items():
        observed=sum(1 for x in rows if x['plant_id']==pid)
        if observed!=int(r['fruit_count']):
            raise ValueError(f'{pid}: fruit_count={r["fruit_count"]} but {observed} fruit rows')
    def values(key, source):
        out=[]
        for r in source:
            try:
                v=float(r.get(key,''))
                if np.isfinite(v) and v>0:out.append(v)
            except (TypeError,ValueError):pass
        return out
    stats={}
    for k in ('plant_height_m','stem_diameter_m','fruit_diameter_m','fruit_count','tip_distance_fraction'):
        v=values(k,rows if k=='tip_distance_fraction' else plants.values())
        stats[k]={'n':len(v),'mean':float(np.mean(v)) if v else None,
                  'std':float(np.std(v,ddof=1)) if len(v)>1 else None,'values':v}
    return rows,stats

def repair_and_prune(plant, out, args):
    original=out/'_original_stem.glb'
    plant.export(original)
    report=out/'repair_report.json'
    repaired=out/'_repaired_stem.glb'
    cfg=types.SimpleNamespace(input=str(original),output=str(repaired),report=str(report),
        voxels=[.0035,.003],dilate=1,min_vertices=120,min_extent=.012,
        min_gap=.003,max_gap=.22,spacing=.008,radius=.0018,max_cells=65000000,
        max_anchor_points=80000,max_query_points=12000,candidate_points=15,
        direction_weight=.6,color=[83,125,32])
    re.main(cfg)
    data=json.loads(report.read_text())
    src=plant.dump(concatenate=True)
    pts=np.asarray(src.vertices,dtype=np.float32)
    low=pts.min(0);diag=float(np.linalg.norm(np.ptp(pts,axis=0)))
    scale=.0035;v=diag*scale
    ijk=np.floor((pts-low+3*v)/v).astype(np.int32)
    shape=tuple((np.ceil(np.ptp(pts,axis=0)/v).astype(int)+8).tolist())
    occ=np.zeros(shape,bool)
    occ[ijk[:,0],ijk[:,1],ijk[:,2]]=True
    labels,n=ndi.label(ndi.binary_dilation(occ,iterations=1),structure=ndi.generate_binary_structure(3,2))
    ids=labels[ijk[:,0],ijk[:,1],ijk[:,2]]
    counts=np.bincount(ids,minlength=n+1)
    root=int(np.argmax(counts[1:])+1)
    deleted=set()
    for c in data['connections']:
        if abs(c['stage_voxel']-scale)>1e-8:continue
        cid=int(c['cluster'])
        if cid==root or cid>=len(counts):continue
        if c['gap']/diag>=args.prune_gap_ratio and c['direction_alignment']<.8 and counts[cid]>=2000:
            deleted.add(cid)
    if deleted and len(plant.geometry)==1:
        m=next(iter(plant.geometry.values())).copy()
        if len(m.vertices)==len(ids):
            keep=~np.any(np.isin(ids[np.asarray(m.faces)],list(deleted)),axis=1)
            m.update_faces(keep);m.remove_unreferenced_vertices()
            clean=trimesh.Scene();clean.add_geometry(m)
        else:
            clean=plant.copy();deleted=set()
    else:clean=plant.copy()
    removed_points=pts[np.isin(ids,list(deleted))]
    tree=cKDTree(removed_points) if len(removed_points) else None
    kept=0
    for c in data['connections']:
        ends=np.asarray([c['start'],c['end']],float)
        if int(c['cluster']) in deleted:continue
        if tree is not None and np.min(tree.query(ends)[0])<diag*.001:continue
        m=re.bridge(ends[0],ends[1],diag*.0018)
        if m is None:continue
        from trimesh.visual.material import PBRMaterial
        m.visual=trimesh.visual.TextureVisuals(material=PBRMaterial(name='Stem_Green_Repair',
            baseColorFactor=[83,125,32,255],metallicFactor=0,roughnessFactor=.9,doubleSided=True))
        clean.add_geometry(m,geom_name=f'repair_bridge_{kept:04d}');kept+=1
    summary={'bridges_found':len(data['connections']),'bridges_retained':kept,
             'deleted_isolated_clusters':sorted(deleted)}
    return clean,summary

def analyze(scene,out,args):
    temp=out/'analysis';temp.mkdir(exist_ok=True)
    source=out/'analysis_source.glb';scene.export(source)
    import types
    cfg=types.SimpleNamespace(input=str(source),out=temp,samples=args.samples,voxel=args.voxel,
        max_voxels=5000000,dilate=1,close=1,stem_radius=args.stem_radius,
        min_branch_nodes=6,min_branch_length=args.min_branch_length,
        min_horizontal_ratio=.5,pca_radius=.025,max_planarity=.7,
        points_per_branch=args.points_per_branch,seed=args.seed)
    for step in (ga.step1,ga.step2,ga.step3,ga.step4,ga.step5):step(cfg)
    return json.loads((temp/'branches.json').read_text())['branches']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stem',required=True);p.add_argument('--fruit',required=True)
    p.add_argument('--output-dir',default='module3_output');p.add_argument('--measurements',default=str(Path(__file__).with_name('measurements.csv')))
    p.add_argument('--crop',default='tomato');p.add_argument('--stage',default='mature')
    p.add_argument('--root-exclusion-fraction',type=float,default=0.20,help='Exclude fruit attachment anchors in bottom fraction of plant height (Y-up; default 0.20)')
    p.add_argument('--count',type=int,default=None,help='Override measured count distribution')
    p.add_argument('--fruit-diameter-m',type=float,default=None,help='Explicit target fruit diameter in meters')
    p.add_argument('--plant-height-m',type=float,default=None,help='Explicit target plant height in meters')
    p.add_argument('--min-distance-ratio',type=float,default=.07)
    p.add_argument('--max-attachment-attempts',type=int,default=50)
    p.add_argument('--skip-repair',action='store_true',help='Skip integrated repair and isolated leaf pruning')
    p.add_argument('--prune-gap-ratio',type=float,default=.035)
    p.add_argument('--samples',type=int,default=180000);p.add_argument('--voxel',type=float,default=.008)
    p.add_argument('--stem-radius',type=float,default=.018);p.add_argument('--min-branch-length',type=float,default=.055)
    p.add_argument('--points-per-branch',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resolution',type=int,default=768);p.add_argument('--frames',type=int,default=120)
    p.add_argument('--fps',type=int,default=15);p.add_argument('--no-video',action='store_true')
    a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed)
    rows,stats=load_measurements(a.measurements,a.crop,a.stage)
    if stats is None:raise ValueError("Measurement CSV missing or crop/stage has no records")
    fitted=dm.fit(stats)
    observations=se.read_attachments(a.measurements,a.crop,a.stage)
    if observations:
        fitted['tip_distance_fraction']=se.fit_beta([r['tip_distance_fraction'] for r in observations])
        height_prior=se.fit_beta([r['height_fraction'] for r in observations])
    else:
        height_prior=None
    fitted['attachment_height_fraction']=height_prior

    plant=trimesh.load(a.stem,force='scene',process=False)
    fruit=trimesh.load(a.fruit,force='scene',process=False)
    fp=meshes(fruit)
    if not fp:raise ValueError('Fruit GLB contains no meshes')
    pruning={'status':'skipped'}
    if not a.skip_repair:plant,pruning=repair_and_prune(plant,out,a)
    pb=plant.bounds;plant_height=float(pb[1,1]-pb[0,1]);fb=bounds(fp)
    fruit_size=float(np.max(fb[1]-fb[0]));fruit_center=np.mean(fb,axis=0)
    if plant_height<=0 or fruit_size<=0:raise ValueError('Degenerate input bounds')
    height_target=a.plant_height_m if a.plant_height_m is not None else dm.sample_normal(fitted['plant_height_m'],rng)
    stem_diameter_sample=dm.sample_normal(fitted['stem_diameter_m'],rng)
    scale_plant=height_target/plant_height if height_target else 1.
    if scale_plant!=1:
        T=trimesh.transformations.scale_matrix(scale_plant)
        plant.apply_transform(T)
    actual_height=plant_height*scale_plant
    unit='m' if height_target else 'input_glb_units'
    if a.count is not None:count=a.count;count_source='user_override'
    else:
        count=dm.sample_count(fitted['fruit_count'],rng);count_source='fitted_truncated_poisson'
    if count<0:raise ValueError('Fruit count must be >=0')
    diam_source='user_override' if a.fruit_diameter_m is not None else 'fitted_truncated_normal'
    if a.fruit_diameter_m is not None and a.fruit_diameter_m<=0:
        raise ValueError('Fruit diameter must be positive')
    if unit!='m':raise ValueError('A calibrated plant height is required for metric sampling')
    if not 0 <= a.root_exclusion_fraction < 1:
        raise ValueError('--root-exclusion-fraction must be in [0,1)')
    root_cutoff_y=float(plant.bounds[0,1]+a.root_exclusion_fraction*actual_height)
    try:branches=analyze(plant,out,a)
    except Exception as e:
        branches=[];print('WARNING: branch extraction failed:',e,file=sys.stderr)
    if count and not branches:raise RuntimeError('No candidate branches identified; adjust voxel/branch parameters; cannot attach fruit reliably')
    eligible_branches=[];excluded_branches=[]
    for branch in branches:
        points=np.asarray(branch['points'],dtype=float)
        if len(points)<2:
            excluded_branches.append(int(branch['id']));continue
        clipped=[]
        for u,v in zip(points[:-1],points[1:]):
            uok=u[1]>=root_cutoff_y;vok=v[1]>=root_cutoff_y
            if uok and not clipped:clipped.append(u.tolist())
            if uok and vok:clipped.append(v.tolist())
            elif uok != vok:
                t=(root_cutoff_y-u[1])/(v[1]-u[1])
                cross=(u+t*(v-u)).tolist()
                clipped.append(cross)
                if vok:clipped.append(v.tolist())
        if len(clipped)>=2 and np.linalg.norm(np.diff(np.asarray(clipped),axis=0),axis=1).sum()>=a.min_branch_length:
            eligible_branches.append({**branch,'points':clipped})
        else:excluded_branches.append(int(branch['id']))
    if count and not eligible_branches:raise RuntimeError('No eligible branches above root exclusion zone; reduce --root-exclusion-fraction or --min-branch-length')
    selected,failures,attempts_log=se.place_fruits(
        eligible_branches,count,fitted['fruit_diameter_m'],
        fitted['tip_distance_fraction'],height_prior,
        float(plant.bounds[0,1]),actual_height,root_cutoff_y,rng,
        attempts=a.max_attachment_attempts,
        min_anchor_distance=a.min_distance_ratio*actual_height,
        fixed_diameter=a.fruit_diameter_m)
    result=plant.copy()
    for j,item in enumerate(selected):
        angle=rng.uniform(0,2*np.pi)
        R=trimesh.transformations.rotation_matrix(angle,[0,1,0]);S=trimesh.transformations.scale_matrix(item['diameter']/fruit_size)
        T=trimesh.transformations.translation_matrix(item['center'])@R@S@trimesh.transformations.translation_matrix(-fruit_center)
        for k,m in enumerate(fp):
            result.add_geometry(m.copy(),geom_name=f'fruit_{j:03d}_part_{k}',transform=T)
        # bridge to the upper fruit surface, green material
        top=item['center']+np.array([0,item['diameter']*.5,0]);anchor=item['anchor']
        if np.linalg.norm(top-anchor)>1e-5:
            stalk=trimesh.creation.cylinder(radius=max(item['diameter']*.025,1e-5),segment=np.array([anchor,top]),sections=10)
            stalk.visual.vertex_colors=[83,125,32,255]
            result.add_geometry(stalk,geom_name=f'peduncle_{j:03d}')
    glb=out/'complete_tomato.glb';result.export(glb)

    print('GLB:',glb,'placed:',len(selected),'/',count)
    if not a.no_video:
        cmd=[sys.executable,str(Path(__file__).with_name('studio_renderer.py')),'--input',str(glb),
            '--output',str(out/'complete_tomato.mp4'),'--resolution',str(a.resolution),
            '--frames',str(a.frames),'--fps',str(a.fps)]
        subprocess.run(cmd,check=True,env={**os.environ,'PYOPENGL_PLATFORM':'egl'})
if __name__=='__main__':main()
