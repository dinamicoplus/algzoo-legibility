import os
import sys,torch as th
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from models import *
th.set_flush_denormal(True)
path=sys.argv[1]; H=int(path.split('_')[-2])
m=MaxRNN(H); m.load_state_dict(th.load(path)); m.eval()
xe=th.randn(200000,N,generator=th.Generator().manual_seed(4242))
acc=lambda L:(L.argmax(1)==target(xe)).float().mean().item()
with th.no_grad():
    w=m.gates()
    print(f'== {path}: trained (hard max) acc {acc(m(xe)):.4f}')
    S=m.S; names=[f'h{k}' for k in range(H)]+['x_t','0']
    # distribution of gate values
    wv=w.flatten(); print('gate values: ', ' '.join(f'{q}:{(wv>q).float().mean():.2f}' for q in [-0.01,-0.05,-0.2,-1,-3]),' (fraction of gates above each level)')
    def snap(delta):
        on=(w>-delta); on[th.arange(H),w.argmax(1)]=True   # keep at least the strongest source
        return th.where(on,th.zeros_like(w),th.full_like(w,-1e4)),on
    for delta in [0.01,0.05,0.1,0.2,0.5,1.0]:
        ws,on=snap(delta); print(f'  snap gates (on if w>-{delta}): acc {acc(m(xe,w=ws)):.4f}, ON gates {int(on.sum())}/{H*S}')
    # ---- symbolic reading of the snapped model (best delta by accuracy) ----
    best=max([0.01,0.05,0.1,0.2,0.5,1.0],key=lambda dl: acc(m(xe,w=snap(dl)[0])))
    ws,on=snap(best); print(f'  using delta={best}')
    # S_i(t): set of input positions (and 'Z' for the zero floor) whose max equals unit i after t inputs
    sets=[frozenset({'Z'}) for _ in range(H)]   # h(0)=0
    for t in range(N):
        new=[]
        for i in range(H):
            s=set()
            for k in range(S):
                if on[i,k]:
                    if k<H: s|=sets[k]
                    elif k==H: s.add(t)
                    else: s.add('Z')
            new.append(frozenset(s))
        sets=new
    def fmt(s):
        pos=sorted(p for p in s if p!='Z'); z='0, ' if 'Z' in s else ''
        if not pos: return 'max(0)' if z else '(empty)'
        return f"max({z}x{{{','.join(map(str,pos))}}})"
    # verify the symbolic sets reproduce the snapped hidden state exactly
    Hs=m.hidden(xe[:2000],w=ws)[:,-1]
    ok=0
    for i in range(H):
        s=sets[i]; cols=[xe[:2000,p] for p in s if p!='Z']+([th.zeros(2000)] if 'Z' in s else [])
        v=th.stack(cols,1).max(1).values; ok+=int(th.allclose(v,Hs[:,i],atol=1e-3))
    print(f'  symbolic sets reproduce snapped units exactly: {ok}/{H}')
    Wo=m.Wo.detach()
    for i in range(H):
        missing=sorted(set(range(N))-set(p for p in sets[i] if p!='Z'))
        desc= 'max of ALL inputs' if not missing else f'max excluding positions {missing}'
        print(f'  unit {i:2d} = {fmt(sets[i]):34s}  [{desc}]   readout weights: '+' '.join(f'{v:+.1f}' for v in Wo[:,i].tolist()))
