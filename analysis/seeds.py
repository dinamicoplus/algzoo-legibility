import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *; from feats2 import features2
from alg_zoo import zoo_2nd_argmax
x=sample(20000,seed=1); F=features2(x); names=list(F)
X=th.stack([F[k] for k in names],-1).reshape(-1,len(names)); XtX=X.T@X
def best_r2(y,kmax=3):
    sel=[];tot=(y-y.mean()).pow(2).sum();Xy=X.T@y;yy=y@y
    for _ in range(kmax):
        best=None
        for j in range(len(names)):
            if j in sel: continue
            S=sel+[j];A=XtX[S][:,S];b=Xy[S]
            c=th.linalg.solve(A+1e-9*th.eye(len(S),dtype=A.dtype),b);res=yy-2*c@b+c@A@c
            if best is None or res<best[0]: best=(res,j)
        sel.append(best[1])
    return (1-best[0]/tot).item(), [names[s] for s in sel]
for seed in range(5):
    m=zoo_2nd_argmax(16,10,seed=seed); sd=m.state_dict()
    wi=sd['rnn.weight_ih_l0'][:,0].double(); wh=sd['rnn.weight_hh_l0'].double(); wo=sd['linear.weight'].double()
    L,H=run(x,wi,wh,wo); Zp=H[:,:-1]@wh.T+x[:,:,None]*wi
    r2=[best_r2(Zp[:,:,j].reshape(-1)) for j in range(16)]
    vals=sorted([v for v,_ in r2],reverse=True)
    crisp=[(j,nm) for j,(v,nm) in enumerate(r2) if v>0.95]
    print(f'seed {seed}: acc {acc(L,x):.3f}  #neurons R2>0.95 (3 feats): {len(crisp)}/16  R2s {[round(v,2) for v in vals]}')
    for j,nm in crisp: print('    ',j,nm)
