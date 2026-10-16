import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'regime.py')).read().split("schemes=")[0])
def cf(x):
    B=x.shape[0]; p0=th.cat([th.zeros(B,1,dtype=x.dtype),x],1); srt=p0.sort(1,descending=True).values
    return th.stack([srt[:,0],srt[:,1],x[:,-1],x[:,-2],x[:,-3]],1)  # M, M2, x[t-1], x[t-2], x[t-3]
Cf,Ce=cf(xf),cf(xe)
labs={}
for j in [1,6,7]:
    yf=Hf[:,-1,j]; pred=th.zeros(xe.shape[0],dtype=xe.dtype); P=0
    for c in range(11):
        mf=lMf==c; me=lMe==c
        co=th.linalg.lstsq(Cf[mf],yf[mf][:,None]).solution[:,0]; pred[me]=Ce[me]@co; P+=5
    lab=th.relu(pred); labs[j]=lab
    r2=1-((lab-hT[:,j]).pow(2).mean()/hT[:,j].var()).item()
    h=hT.clone(); h[:,j]=lab
    print(f'n{j}: compact age-of-max form, {P} params, R2 {r2:.4f}, swap alone -> acc {acc(h@Wo_r.T,xe):.3f}')
h=hT.clone()
for j in labs: h[:,j]=labs[j]
print('swap n1,n6,n7 together -> acc',round(acc(h@Wo_r.T,xe),3))
