import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
def prep(x):
    B=x.shape[0]; p0=th.cat([th.zeros(B,1,dtype=x.dtype),x],1)
    srt,idx=p0.sort(1,descending=True)
    lagM=th.where(idx[:,0]==0,th.zeros_like(idx[:,0]),n-idx[:,0]+1)
    lag2=th.where(idx[:,1]==0,th.zeros_like(idx[:,1]),n-idx[:,1]+1)
    base=th.cat([srt[:,:3],x,th.relu(x)],1)  # M,M2,M3, raw inputs, relu inputs : 23 feats
    return base,lagM,lag2
xf=sample(300000,seed=21); _,Hf=run(xf,Wi_r,Wh_r,Wo_r); Bf,lMf,l2f=prep(xf)
xe=sample(100000,seed=22); Le,He=run(xe,Wi_r,Wh_r,Wo_r); Be,lMe,l2e=prep(xe); hT=He[:,-1]
print('true acc',round(acc(Le,xe),3))
def fit_regime(j, keyf, keye, ncell):
    pred=th.zeros(xe.shape[0],dtype=xe.dtype); params=0
    yf=Hf[:,-1,j]
    for c in range(ncell):
        mf=keyf==c; me=keye==c
        if mf.sum()<50: 
            continue
        A=Bf[mf]; co=th.linalg.lstsq(A,yf[mf][:,None]).solution[:,0]
        pred[me]=Be[me]@co; params+=A.shape[1]
    return th.relu(pred),params
schemes={'global linear':(th.zeros_like(lMf),th.zeros_like(lMe),1),
         'by lag of max (11 cells)':(lMf,lMe,11),
         'by lag of max x lag of 2nd (121 cells)':(lMf*11+l2f,lMe*11+l2e,121)}
for name,(kf,ke,nc) in schemes.items():
    labs={};r2s={};P=0
    for j in range(d):
        lab,pp=fit_regime(j,kf,ke,nc); labs[j]=lab; P+=pp
        r2s[j]=1-((lab-hT[:,j]).pow(2).mean()/hT[:,j].var()).item()
    print(f'== {name}: params total {P} (model has 432)')
    print('   R2 per neuron:',' '.join(f'n{j}:{r2s[j]:.3f}' for j in range(d)))
    for group,js in [('n1,n6,n7',[1,6,7]),('6 crisp',[1,2,4,6,7,9]),('all 16',list(range(d)))]:
        h=hT.clone()
        for j in js: h[:,j]=labs[j]
        print(f'   swap {group:9s} -> acc {acc(h@Wo_r.T,xe):.3f}')
    for j in [1,6,7]:
        h=hT.clone(); h[:,j]=labs[j]; print(f'   swap n{j} alone -> acc {acc(h@Wo_r.T,xe):.3f}')
