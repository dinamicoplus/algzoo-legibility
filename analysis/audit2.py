import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *; from feats2 import features2
x=sample(20000,seed=1)
_,H=run(x,Wi_r,Wh_r,Wo_r)
Zp = H[:,:-1] @ Wh_r.T + x[:,:,None]*Wi_r
F=features2(x); names=list(F); print(len(names),'features')
X=th.stack([F[k] for k in names],-1).reshape(-1,len(names))
XtX=X.T@X
def greedy(y,kmax=6):
    sel=[];out=[];tot=(y-y.mean()).pow(2).sum(); Xy=X.T@y; yy=y@y
    for _ in range(kmax):
        best=None
        for j in range(len(names)):
            if j in sel: continue
            S=sel+[j]; A=XtX[S][:,S]; b=Xy[S]
            try: c=th.linalg.solve(A+1e-9*th.eye(len(S),dtype=A.dtype),b)
            except: continue
            res=yy-2*c@b+c@A@c
            if best is None or res<best[0]: best=(res,j,c)
        sel.append(best[1]); out.append((1-best[0]/tot).item()); c=best[2]
    return sel,out,c
res={}
for j in range(d):
    y=Zp[:,:,j].reshape(-1)
    sel,r2,c=greedy(y)
    res[j]=(sel,r2,c)
    terms=' '.join(f'{c[i]:+.2f}*{names[s]}' for i,s in enumerate(sel[:4]))
    print(f'n{j:2d} R2 {[round(v,3) for v in r2]}  ≈ {terms}')
