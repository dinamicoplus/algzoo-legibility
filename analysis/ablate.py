import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
x=sample(200000,seed=7)
L,H=run(x,Wi_r,Wh_r,Wo_r); base=acc(L,x); print('base',base)
mean=H[:,1:].mean((0,1))
print('per-neuron ablation (zero / mean, all timesteps):')
rows=[]
for j in range(d):
    def hz(t,h,x,j=j): h=h.clone(); h[:,j]=0; return h
    def hm(t,h,x,j=j): h=h.clone(); h[:,j]=mean[j]; return h
    az=acc(run(x,hook=hz,Wi=Wi_r,Wh=Wh_r,Wo=Wo_r)[0],x); am=acc(run(x,hook=hm,Wi=Wi_r,Wh=Wh_r,Wo=Wo_r)[0],x)
    rows.append((j,az,am)); print(f' n{j:2d} zero {az:.3f}  mean {am:.3f}')
# weak-activation pruning: zero activations below q-quantile of that neuron's positive activations (essay style), at every step
print('weak-activation pruning:')
Hpos=H[:,1:]
for frac in [0.25,0.5,0.75,0.9]:
    thr=th.stack([th.quantile(Hpos[:,:,j][Hpos[:,:,j]>0][:500000],frac) for j in range(d)])
    def hk(t,h,x,thr=thr): return th.where(h<thr,th.zeros_like(h),h)
    L2,_=run(x,hook=hk,Wi=Wi_r,Wh=Wh_r,Wo=Wo_r)
    ch=(L2.argmax(1)!=L.argmax(1)).double().mean().item()
    # what fraction of total activation mass removed
    mass=(Hpos*(Hpos<thr)).sum()/Hpos.sum()
    print(f' remove weakest {int(frac*100)}% of active values (mass {mass:.2f}): acc {acc(L2,x):.3f}, answers changed {ch:.3f}')
# compare: remove the strongest activations carrying the same mass as weakest 75%
