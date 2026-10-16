import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *; from feats2 import features2
x=sample(40000,seed=1); _,H=run(x,Wi_r,Wh_r,Wo_r)
F=features2(x); names=list(F)
xe=sample(50000,seed=4242); Fe=features2(xe)
Le,He=run(xe,Wi_r,Wh_r,Wo_r); hT=He[:,-1]; base=acc(Le,xe); print('true acc',round(base,3))
def fit(j,k,step):
    X=th.stack([F[nm][:,step] for nm in names],-1); y=H[:,step+1,j]
    XtX=X.T@X;Xy=X.T@y;yy=y@y;sel=[]
    for _ in range(k):
        best=None
        for i in range(len(names)):
            if i in sel: continue
            S=sel+[i];A=XtX[S][:,S];b=Xy[S]
            c=th.linalg.solve(A+1e-9*th.eye(len(S),dtype=A.dtype),b);res=yy-2*c@b+c@A@c
            if best is None or res<best[0]: best=(res,i,c)
        sel.append(best[1])
    return sel,best[2]
crisp=[1,2,4,6,7,9]
labs={}
for j in crisp:
    sel,c=fit(j,3,n-1)
    Xe=th.stack([Fe[names[s]][:,-1] for s in sel],-1); lab=th.relu(Xe@c)
    r2=1-((lab-hT[:,j]).pow(2).mean()/hT[:,j].var()).item()
    labs[j]=lab
    h=hT.clone(); h[:,j]=lab
    print(f' n{j}: 3-feat label {[names[s] for s in sel]}  heldout R2 {r2:.3f}  swap alone -> acc {acc(h@Wo_r.T,xe):.3f}')
h=hT.clone()
for j in crisp: h[:,j]=labs[j]
print('swap all 6 crisp neurons at readout -> acc', round(acc(h@Wo_r.T,xe),3))
# exact formulas individually
p=xe
def mx(excl):
    m=th.ones(n,dtype=bool)
    for k in excl: m[n-k]=False
    return th.relu(p[:,m].max(1).values)
exact={2:mx([1]),4:mx([])-mx([1]),7:mx([])-p[:,-1],6:mx([2])-p[:,-1],1:mx([3])-p[:,-1]}
for j,f in exact.items():
    a=(f@hT[:,j])/(f@f); h=hT.clone(); h[:,j]=a*f
    print(f' exact formula n{j} alone -> acc {acc(h@Wo_r.T,xe):.3f}')
