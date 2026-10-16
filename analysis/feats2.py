import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import itertools
def features2(x):
    B,n_=x.shape; cols={}
    def add(name,t,val): cols.setdefault(name,[None]*n_)[t-1]=val
    z=th.zeros(B,dtype=x.dtype)
    for t in range(1,n_+1):
        p=x[:,:t]; p0=th.cat([th.zeros(B,1,dtype=x.dtype),p],1)
        srt,idx=p0.sort(1,descending=True)
        M=srt[:,0]; M2=srt[:,1]
        add('M',t,M); add('M2',t,M2)
        for k in range(1,4): add(f'x[t-{k}]',t,p[:,-k] if k<=t else z)
        for k in range(1,n_+1): add(f'relu x[t-{k}]',t,th.relu(p[:,-k]) if k<=t else z)
        # max excluding a set of recent lags
        for r in (1,2):
            for S in itertools.combinations(range(1,n_+1),r):
                mask=th.ones(t,dtype=bool)
                for k in S:
                    if k<=t: mask[t-k]=False
                q=th.cat([th.zeros(B,1,dtype=x.dtype),p[:,mask]],1)
                add('LOO'+''.join(f'{{{k}}}' for k in S),t,q.max(1).values)
        # age of max (lag): idx in p0 (0 = the zero pad)
        lagM = th.where(idx[:,0]==0, th.full_like(idx[:,0],-1), t-idx[:,0]+1)  # lag k such that argmax at t-k
        lagM2= th.where(idx[:,1]==0, th.full_like(idx[:,1],-1), t-idx[:,1]+1)
        for k in range(1,n_+1):
            add(f'M*[maxlag={k}]',t,M*(lagM==k)); add(f'M2*[2ndlag={k}]',t,M2*(lagM2==k))
    return {k:th.stack(v,1) for k,v in cols.items()}
