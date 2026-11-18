import numpy as np
from scipy.stats import truncnorm, beta, poisson

def fit(stats):
    result={}
    for key in ('plant_height_m','stem_diameter_m','fruit_diameter_m'):
        vals=np.asarray(stats[key]['values'],float)
        if len(vals)<2:raise ValueError(f'At least two valid {key} observations required')
        mu=float(vals.mean()); sd=float(vals.std(ddof=1))
        result[key]={'family':'truncated_normal','mean':mu,'std':max(sd,1e-8),
                     'lower':mu*.8,'upper':mu*1.2,'n':len(vals)}
    counts=np.asarray(stats['fruit_count']['values'],float)
    if len(counts)<2:raise ValueError('Need fruit count observations')
    result['fruit_count']={'family':'poisson','lambda':float(counts.mean()),
                           'lower':max(0,int(np.floor(counts.mean()*.8))),
                           'upper':int(np.ceil(counts.mean()*1.2)),'n':len(counts)}
    fr=np.asarray(stats['tip_distance_fraction']['values'],float)
    if len(fr)<2 or np.any((fr<=0)|(fr>=1)):
        raise ValueError('tip_distance_fraction must contain >=2 values strictly between 0 and 1')
    mean=float(fr.mean());var=float(fr.var(ddof=1));concentration=max(2.,mean*(1-mean)/max(var,1e-6)-1)
    result['tip_distance_fraction']={'family':'beta_moments','alpha':float(mean*concentration),
                                     'beta':float((1-mean)*concentration),'n':len(fr)}
    return result

def sample_normal(model,rng):
    mu=model['mean'];sd=model['std']
    return float(truncnorm.rvs((model['lower']-mu)/sd,(model['upper']-mu)/sd,
                              loc=mu,scale=sd,random_state=rng))

def sample_count(model,rng):
    for _ in range(1000):
        x=int(poisson.rvs(model['lambda'],random_state=rng))
        if model['lower']<=x<=model['upper']:return x
    return int(np.clip(round(model['lambda']),model['lower'],model['upper']))

def sample_fraction(model,rng):
    return float(beta.rvs(model['alpha'],model['beta'],random_state=rng))

