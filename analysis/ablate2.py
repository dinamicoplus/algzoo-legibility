import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
x=sample(200000,seed=int(sys.argv[1]) if len(sys.argv)>1 else 7)
L,H=run(x,Wi_r,Wh_r,Wo_r)
Hpos=H[:,1:]
def thr_for(frac): return th.stack([th.quantile(Hpos[:,:,j][Hpos[:,:,j]>0][:500000],frac) for j in range(d)])
for frac in [0.1,0.25,0.5,0.75]:
    thr=thr_for(frac)
    for mode in ['final step only','steps 1-9 only']:
        def hk(t,h,x,thr=thr,mode=mode):
            if (mode=='final step only') != (t==n-1): return h
            return th.where(h<thr,th.zeros_like(h),h)
        L2,_=run(x,hook=hk,Wi=Wi_r,Wh=Wh_r,Wo=Wo_r)
        print(f'weakest {int(frac*100):2d}%  {mode:16s}: acc {acc(L2,x):.3f}')
# strongest: remove top activations with same mass as weakest 75% (final step only)
hf=H[:,-1]
for frac in [0.75]:
    thr=thr_for(frac)
    mass_weak=(hf*(hf<thr)).sum(0)
    # per neuron find top threshold with equal mass
    out=hf.clone()
    for j in range(d):
        v=hf[:,j]; sv,_=v.sort(descending=True); cs=sv.cumsum(0)
        k=int((cs<mass_weak[j]).sum()); tj=sv[k] if k<len(sv) else -1
        out[:,j]=th.where(v>tj, th.zeros_like(v), v)
    print('remove strongest, equal mass, final step only: acc', acc(out@Wo_r.T,x))
# How large are logit margins? fraction of samples whose top-2 logit margin is < 1% of logit scale
m=L.sort(1,descending=True).values; marg=(m[:,0]-m[:,1]); print('median margin',marg.median().item(),'logit std',L.std().item())
