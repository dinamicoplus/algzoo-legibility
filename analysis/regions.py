import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from collections import Counter
x=sample(1000000,seed=5)
_,H=run(x,Wi_r,Wh_r,Wo_r)
on=(H[:,1:]>0).reshape(x.shape[0],-1).numpy().astype(np.uint8)   # 10 steps x 16 neurons = 160 bits
keys=[r.tobytes() for r in np.packbits(on,axis=1)]
c=Counter(keys); counts=np.array(sorted(c.values(),reverse=True)); cum=np.cumsum(counts)/len(keys)
print('samples',len(keys),'distinct linear regions seen',len(c))
for q in [0.5,0.9,0.99]: print(f' regions needed to cover {int(q*100)}% of inputs: {int((cum<q).sum())+1}')
print(' regions seen only once:', int((counts==1).sum()))
# homogeneity check: scaling inputs by c>0 scales every neuron by c
x2=x[:1000]; _,Ha=run(x2,Wi_r,Wh_r,Wo_r); _,Hb=run(3.7*x2,Wi_r,Wh_r,Wo_r)
print('max |h(3.7x) - 3.7 h(x)|:', (Hb-3.7*Ha).abs().max().item())
