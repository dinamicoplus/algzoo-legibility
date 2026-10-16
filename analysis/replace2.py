import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *; from feats2 import features2
x=sample(40000,seed=1); _,H=run(x,Wi_r,Wh_r,Wo_r)
F=features2(x); names=list(F); X=th.stack([F[k][:,-1] for k in names],-1)  # final step only
xe=sample(50000,seed=4242); Fe=features2(xe); Xe=th.stack([Fe[k][:,-1] for k in names],-1)
Le,He=run(xe,Wi_r,Wh_r,Wo_r); hT=He[:,-1]
XtX=X.T@X
def greedy(y,kmax):
    sel=[];tot=(y-y.mean()).pow(2).sum();Xy=X.T@y;yy=y@y;r2=[]
    for _ in range(kmax):
        best=None
        for j in range(len(names)):
            if j in sel: continue
            S=sel+[j];A=XtX[S][:,S];b=Xy[S]
            c=th.linalg.solve(A+1e-9*th.eye(len(S),dtype=A.dtype),b);res=yy-2*c@b+c@A@c
            if best is None or res<best[0]: best=(res,j,c)
        sel.append(best[1]);r2.append((1-best[0]/tot).item())
    return sel,best[2],r2
# fit post-activation directly at final step (best case for labels)
for k in [3,6,10]:
    labs={};r2s={}
    for j in range(d):
        sel,c,r2=greedy(H[:,-1,j],k); labs[j]=Xe[:,sel]@c; r2s[j]=r2[-1]
    h=hT.clone()
    for j in range(d): h[:,j]=labs[j]
    # R2 on held out
    ho=[1-((labs[j]-hT[:,j]).pow(2).mean()/hT[:,j].var()).item() for j in range(d)]
    print(f'{k} feats/neuron, final-step fit: median heldout R2 {np.median(ho):.3f}, min {min(ho):.3f} -> acc {acc(h@Wo_r.T,xe):.3f}  (true {acc(Le,xe):.3f})')
# Exact formulas from ARC blog for neurons 2,4,6,7,1 (scaled by best scalar)
t=n; p=xe
def mx(excl): 
    m=th.ones(n,dtype=bool)
    for k in excl: m[n-k]=False
    return th.relu(p[:,m].max(1).values)
exact={2:mx([1]), 4:mx([])-mx([1]), 7:mx([])-p[:,-1], 6:mx([2])-p[:,-1], 1:mx([3])-p[:,-1]}
h=hT.clone()
for j,f in exact.items():
    a=(f@hT[:,j])/(f@f); h[:,j]=a*f
    print(f' n{j}: exact-formula R2 {1-((a*f-hT[:,j]).pow(2).mean()/hT[:,j].var()).item():.4f}')
print('replace 5 ARC-identified neurons with exact formulas at readout: acc',round(acc(h@Wo_r.T,xe),3))
