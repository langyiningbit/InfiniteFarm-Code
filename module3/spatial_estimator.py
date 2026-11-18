import csv
from pathlib import Path
import numpy as np
from scipy.stats import truncnorm, beta


def read_attachments(path, crop, stage):
    if not path or not Path(path).is_file():
        return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r.get('crop') == crop and r.get('stage') == stage]
    clean=[]
    for r in rows:
        try:
            h=float(r['height_fraction']); t=float(r['tip_distance_fraction'])
            if np.isfinite(h) and np.isfinite(t) and 0<h<1 and 0<t<1:
                clean.append({'height_fraction':h,'tip_distance_fraction':t,
                              'plant_id':r.get('plant_id',''), 'data_origin':r.get('data_origin','unknown')})
        except (ValueError,KeyError,TypeError):
            pass
    return clean


def fit_beta(values):
    a=np.asarray(values,float)
    if len(a)<2 or np.any((a<=0)|(a>=1)):
        raise ValueError('Need >=2 valid observations strictly between 0 and 1')
    mean=float(a.mean()); variance=float(a.var(ddof=1))
    concentration=max(2.,mean*(1-mean)/max(variance,1e-8)-1.)
    return {'family':'beta_moments','alpha':mean*concentration,
            'beta':(1-mean)*concentration,'n':len(a), 'mean':mean}


def draw_beta(model,rng):
    return float(beta.rvs(model['alpha'],model['beta'],random_state=rng))


def sample_normal(model,rng):
    mu=model['mean']; sd=model['std']
    return float(truncnorm.rvs((model['lower']-mu)/sd,(model['upper']-mu)/sd,
                              loc=mu,scale=sd,random_state=rng))


def sample_count_empirical(values,rng):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)&(values>=0)]
    if not len(values): raise ValueError('No fruit counts')
    mu=float(values.mean()); lo=int(np.ceil(.8*mu)); hi=int(np.floor(1.2*mu))
    eligible=np.rint(values).astype(int)
    eligible=eligible[(eligible>=lo)&(eligible<=hi)]
    if not len(eligible): raise ValueError('No empirical fruit counts within ±20% of mean')
    return int(rng.choice(eligible))


def prepare_branches(branches):
    result=[]
    for b in branches:
        p=np.asarray(b['points'],float)
        if len(p)<2:continue
        segment=np.linalg.norm(np.diff(p,axis=0),axis=1)
        cum=np.r_[0.,np.cumsum(segment)]
        if cum[-1]<=1e-8:continue
        result.append({'id':int(b['id']),'points':p,'cum':cum,'length':float(cum[-1])})
    return result


def interpolate(branch, tip_fraction):
    d=(1-tip_fraction)*branch['length']; cum=branch['cum']; p=branch['points']
    j=int(np.clip(np.searchsorted(cum,d,side='right')-1,0,len(cum)-2))
    t=float(np.clip((d-cum[j])/max(cum[j+1]-cum[j],1e-12),0,1))
    return p[j]*(1-t)+p[j+1]*t


def candidate_order(branches, target_tip, target_height, bottom, height, rng,
                    height_weight=2.0, tip_weight=1.0):
    proposals=[]
    for b in branches:
        fractions=np.linspace(.01,.99,81)
        pts=np.asarray([interpolate(b,f) for f in fractions])
        if target_height is None:
            f=target_tip; pos=interpolate(b,f); cost=0.
        else:
            normalized_y=(pts[:,1]-bottom)/height
            costs=height_weight*(normalized_y-target_height)**2+tip_weight*(fractions-target_tip)**2
            idx=int(np.argmin(costs)); f=float(fractions[idx]);pos=pts[idx];cost=float(costs[idx])
        proposals.append({'branch_id':b['id'],'anchor':pos,'tip_fraction':f,
                          'target_tip_fraction':target_tip,'target_height_fraction':target_height,
                          'target_cost':cost,'tie_break':float(rng.random())})
    proposals.sort(key=lambda x:(x['target_cost'],x['tie_break']))
    return proposals


def place_fruits(branches, n, fruit_model, tip_model, height_model,
                 bottom, plant_height, root_cutoff, rng, attempts=50,
                 min_anchor_distance=0.0, fixed_diameter=None):
    branches=prepare_branches(branches)
    if n and not branches:raise ValueError('No eligible branches')
    selected=[]; failures=[]; attempts_log=[]
    for fruit_id in range(n):
        diameter=float(fixed_diameter) if fixed_diameter is not None else sample_normal(fruit_model,rng)
        placed=False
        for attempt in range(attempts):
            target_tip=draw_beta(tip_model,rng)
            target_height=draw_beta(height_model,rng) if height_model is not None else None
            choices=candidate_order(branches,target_tip,target_height,bottom,plant_height,rng)
            for c in choices:
                anchor=c['anchor']; centre=anchor+np.array([0.,-diameter*.6,0.]);radius=diameter*.5
                if centre[1]-radius < root_cutoff:continue
                if any(np.linalg.norm(centre-s['center']) < radius+s['radius'] for s in selected):continue
                if any(np.linalg.norm(anchor-s['anchor']) < min_anchor_distance for s in selected):continue
                item={**c,'fruit_id':fruit_id,'anchor':anchor.copy(),'center':centre,
                      'radius':radius,'diameter':diameter,'attempt':attempt}
                selected.append(item);placed=True;break
            attempts_log.append({'fruit_id':fruit_id,'attempt':attempt,
                                 'sampled_tip_fraction':target_tip,
                                 'sampled_height_fraction':target_height,
                                 'accepted':placed})
            if placed:break
        if not placed:failures.append({'fruit_id':fruit_id,'attempts':attempts})
    return selected,failures,attempts_log
