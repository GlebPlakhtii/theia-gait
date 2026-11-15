"""Cohort-blind regional motion density model. No clinical/classification labels."""
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import logsumexp, ndtr
REGIONS=['trunk','hips','shoulders','left_knee','right_knee','left_arm','right_arm']
TITLES=['Trunk','Pelvis / hips','Shoulders','Left knee / leg','Right knee / leg','Left arm','Right arm']
FEATURES=['Median projected angle','Projected angular excursion (P95–P05)','Angular standard deviation','Median angular speed']
UNITS=['deg','deg','deg','deg/s']
JOINTS=[[0,8],[1,4],[11,14],[4,5,6],[1,2,3],[4,11,12],[1,14,15]]

def angle(a,b):
    norm=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/np.maximum(norm,1e-8),-1,1)))

def features(pose,view):
    p=pose[...,:2].astype(float);valid=pose[...,2]>=.4
    torso=np.linalg.norm(p[:,8]-p[:,0],axis=-1);tv=valid[:,8]&valid[:,0]
    scale=float(np.median(torso[tv])) if tv.any() else 0.
    values=np.full((7,4),np.nan);coverage=np.zeros(7);reasons=[]
    for j,joints in enumerate(JOINTS):
        why=[];ok=valid[:,joints].all(1);coverage[j]=ok.mean()
        if scale<.08:why.append('torso too small or not visible')
        if ok.mean()<.8:why.append('fewer than 80% reliable regional frames')
        if j in [1,2] and view!='front_rear':why.append('line tilt requires frontal/rear view')
        if j==0:
            d=p[:,8]-p[:,0];curve=np.degrees(np.arctan2(d[:,0],-d[:,1]));edges=[(0,8)]
        elif j in [1,2]:
            a,b=joints;d=p[:,b]-p[:,a];curve=np.degrees(np.arctan2(d[:,1],np.abs(d[:,0])));edges=[(a,b)]
            if ok.any() and np.median(np.abs(d[ok,0]))<scale*.15:why.append('line too foreshortened')
        elif j in [3,4]:
            a,b,c=joints;curve=180-angle(p[:,a]-p[:,b],p[:,c]-p[:,b]);edges=[(a,b),(b,c)]
        else:
            h,s,e=joints;curve=angle(p[:,h]-p[:,s],p[:,e]-p[:,s]);edges=[(h,s),(s,e)]
        for a,b in edges:
            lengths=np.linalg.norm(p[:,a]-p[:,b],axis=-1)[ok]
            if len(lengths) and (np.percentile(lengths,5)<scale*.05 or np.percentile(lengths,95)>max(np.percentile(lengths,5),1e-6)*2.5):why.append('unstable or collapsed projected segment')
        if why:reasons.append(sorted(set(why)));continue
        # Smooth only contiguous valid runs; never bridge missing joints.
        smooth=curve.copy();ids=np.flatnonzero(ok);runs=np.split(ids,np.where(np.diff(ids)>1)[0]+1)
        for run in runs:
            if len(run)>=5:smooth[run]=savgol_filter(curve[run],5,2)
        pair=ok[1:]&ok[:-1]
        if pair.sum()<20:reasons.append(['too few adjacent reliable frames']);continue
        q=smooth[ok];median=abs(float(np.median(q))) if j<3 else float(np.median(q))
        values[j]=[median,float(np.percentile(q,95)-np.percentile(q,5)),float(np.std(q)),float(np.median(np.abs(np.diff(smooth)[pair])*30))]
        reasons.append([])
    return values,coverage,reasons

def balanced_weights(rows):
    """Equal person, then trial, then clip, then window weight. No source/type strata."""
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for i,m in enumerate(rows):groups[m['subject']][m['trial_block']][m['clip_id']].append(i)
    w=np.zeros(len(rows),float)
    for trials in groups.values():
        for clips in trials.values():
            for ids in clips.values():w[ids]=1/(len(groups)*len(trials)*len(clips)*len(ids))
    assert np.isclose(w.sum(),1);return w

def log_components(z,model):
    out=[];d=z.shape[1]
    for mu,cov,pi in zip(model['means'],model['covariances'],model['mixture_weights']):
        diff=z-mu;inv=np.linalg.inv(cov);_,ld=np.linalg.slogdet(cov)
        out.append(np.log(max(pi,1e-12))-.5*(d*np.log(2*np.pi)+ld+np.einsum('ni,ij,nj->n',diff,inv,diff)))
    return np.stack(out,1)

def nll(x,model):
    z=(np.log1p(np.maximum(x,0))-model['center'])/model['scale']
    return -logsumexp(log_components(z,model),axis=1)

def fit(x,weights,seed=20261115):
    raw=np.log1p(np.maximum(x,0));center=np.average(raw,axis=0,weights=weights);scale=np.sqrt(np.average((raw-center)**2,axis=0,weights=weights));scale=np.maximum(scale,.08)
    z=(raw-center)/scale;best=None
    for initialization in range(2):
        rng=np.random.default_rng(seed+initialization);means=z[rng.choice(len(z),2,replace=False,p=weights)].copy();cov=np.repeat(np.eye(4)[None],2,axis=0);pi=np.array([.5,.5]);hist=[]
        for it in range(200):
            model=dict(center=center,scale=scale,means=means,covariances=cov,mixture_weights=pi)
            lc=log_components(z,model);ll=logsumexp(lc,1);objective=float(-np.sum(weights*ll));hist.append(objective)
            resp=np.exp(lc-ll[:,None])*weights[:,None];nk=resp.sum(0);pi=(nk+.01)/(1+.02)
            means=resp.T@z/np.maximum(nk[:,None],1e-12)
            cov=[]
            for k in range(2):
                diff=z-means[k];c=(diff.T*resp[:,k])@diff/max(nk[k],1e-12);c=.85*c+.15*np.diag(np.diag(c))+.05*np.eye(4);cov.append(c)
            cov=np.array(cov)
            if len(hist)>10 and abs(hist[-1]-hist[-2])<1e-6:break
        model=dict(center=center,scale=scale,means=means,covariances=cov,mixture_weights=pi)
        score=float(np.average(nll(x,model),weights=weights))
        if best is None or score<best['training_nll']:best=dict(**model,training_nll=score,history=hist,initialization=initialization)
    return best

def marginal_cdf(value,j,model):
    z=(np.log1p(max(float(value),0))-model['center'][j])/model['scale'][j]
    return float(np.sum(model['mixture_weights']*ndtr((z-model['means'][:,j])/np.sqrt(model['covariances'][:,j,j]))))

def marginal_quantile(q,j,model):
    lo,hi=-12.,12.
    for _ in range(70):
        mid=(lo+hi)/2;cdf=np.sum(model['mixture_weights']*ndtr((mid-model['means'][:,j])/np.sqrt(model['covariances'][:,j,j])))
        if cdf<q:lo=mid
        else:hi=mid
    return float(max(0,np.expm1(np.clip(((lo+hi)/2)*model['scale'][j]+model['center'][j],-20,20))))

def weighted_quantile(x,w,q):
    order=np.argsort(x);return float(x[order][min(len(x)-1,np.searchsorted(np.cumsum(w[order]),q))])
