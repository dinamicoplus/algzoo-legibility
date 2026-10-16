import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'regime.py')).read().split("schemes=")[0].split("print('true acc'")[0])
names=['M','M2','M3']+[f'x{t}' for t in range(n)]+[f'r{t}' for t in range(n)]
np.set_printoptions(precision=2,suppress=True,linewidth=200)
for j in [7]:
    print(f'n{j}: per-regime coefficients (rows: lag of max; 0 = all inputs negative)')
    print('lag  '+' '.join(f'{nm:>6s}' for nm in names))
    yf=Hf[:,-1,j]
    for c in range(11):
        mf=lMf==c
        co=th.linalg.lstsq(Bf[mf],yf[mf][:,None]).solution[:,0]
        print(f'{c:3d}  '+' '.join(f'{v:6.2f}' if abs(v)>0.05 else '     .' for v in co.numpy()))
