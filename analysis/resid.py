import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
x=sample(100000,seed=11); L,H=run(x,Wi_r,Wh_r,Wo_r); hT=H[:,-1]; p=x
def mx(excl):
    m=th.ones(n,dtype=bool)
    for k in excl: m[n-k]=False
    return th.relu(p[:,m].max(1).values)
p0=th.cat([th.zeros(x.shape[0],1,dtype=x.dtype),p],1)
srt,idx=p0.sort(1,descending=True)
lagM=th.where(idx[:,0]==0,th.zeros_like(idx[:,0]),n-idx[:,0]+1)   # 0 means max is the zero pad (all negative)
lag2=th.where(idx[:,1]==0,th.zeros_like(idx[:,1]),n-idx[:,1]+1)
exact={7:mx([])-p[:,-1],6:mx([2])-p[:,-1],1:mx([3])-p[:,-1]}
for j,f in exact.items():
    a=(f@hT[:,j])/(f@f); res=hT[:,j]-a*f
    print(f'== n{j}: scale {a:.3f}, R2 {1-res.var()/hT[:,j].var():.4f}')
    # where does the residual live? by lag of max (1=latest input)
    print('   residual var share by lag of max:', {int(k): round((res[lagM==k].pow(2).sum()/res.pow(2).sum()).item(),2) for k in range(0,11) if (lagM==k).any() and (res[lagM==k].pow(2).sum()/res.pow(2).sum())>0.03})
    print('   residual var share by lag of 2nd max:', {int(k): round((res[lag2==k].pow(2).sum()/res.pow(2).sum()).item(),2) for k in range(0,11) if (lag2==k).any() and (res[lag2==k].pow(2).sum()/res.pow(2).sum())>0.03})
    # when the formula predicts 0 (relu off) vs on
    on=f>0; print(f'   share of residual where formula==0: {(res[~on].pow(2).sum()/res.pow(2).sum()).item():.2f} (formula==0 on {(~on).double().mean():.2f} of data)')
    # correlate residual with each raw input relu, and with x[t-1]
    cs=[(f'x{t}',th.corrcoef(th.stack([res,p[:,t]]))[0,1].item()) for t in range(n)]
    cs+= [(f'relu x{t}',th.corrcoef(th.stack([res,th.relu(p[:,t])]))[0,1].item()) for t in range(n)]
    cs+= [('M2',th.corrcoef(th.stack([res,srt[:,1]]))[0,1].item()),('M',th.corrcoef(th.stack([res,srt[:,0]]))[0,1].item())]
    print('   top correlates:', sorted(cs,key=lambda c:-abs(c[1]))[:5])
