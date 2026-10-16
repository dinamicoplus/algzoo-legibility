import os
import sys,torch as th,torch.nn.functional as Fn
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from models import *
th.set_flush_denormal(True); th.manual_seed(0); th.set_num_threads(1)
def unit_sets(on,H,S):
    sets=[frozenset({'Z'}) for _ in range(H)]
    for t in range(N):
        sets=[frozenset(set().union(*[sets[k] for k in range(H) if on[i,k]]) | ({t} if on[i,H] else set()) | ({'Z'} if S>H+1 and on[i,H+1] else set())) for i in range(H)]
    return sets
def feats(x,sets):
    out=[]
    for s in sets:
        cols=[x[:,p] for p in s if p!='Z']+([th.zeros(x.shape[0])] if 'Z' in s else [])
        out.append(th.stack(cols,1).max(1).values)
    return th.stack(out,1)
if __name__=='__main__':
    path=sys.argv[1]; H=int(path.split('_')[-2]); delta=float(sys.argv[2])
    m=MaxRNN(H); m.load_state_dict(th.load(path))
    xe=th.randn(200000,N,generator=th.Generator().manual_seed(4242)); acc=lambda L,x:(L.argmax(1)==target(x)).float().mean().item()
    with th.no_grad(): w=m.gates(); on=(w>-delta); on[th.arange(H),w.argmax(1)]=True
    sets=unit_sets(on,H,m.S)
    xt=th.randn(400000,N); Ft=feats(xt,sets); Fe=feats(xe,sets); yt=target(xt)
    W=th.nn.Parameter(m.Wo.detach().clone()); opt=th.optim.Adam([W],lr=5e-2)
    for s in range(600):
        idx=th.randint(0,len(xt),(8192,)); loss=Fn.cross_entropy(Ft[idx]@W.T,yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        print(f'{path}: snapped δ={delta}: trained readout {acc(Fe@m.Wo.T,xe):.4f}  -> refitted readout {acc(Fe@W.T,xe):.4f}')
        pred=(Fe@W.T).argmax(1); tg=target(xe)
        print('   acc by position of true 2nd max:',' '.join(f'{p}:{(pred[tg==p]==p).float().mean():.2f}' for p in range(N)))
        covered=sorted({p for s in sets for p in s if p!='Z' and len([q for q in s if q!='Z'])==1})
        loo=sorted({tuple(sorted(set(range(N))-set(q for q in s if q!='Z')))[0] for s in sets if len(set(range(N))-set(q for q in s if q!='Z'))==1})
        print(f'   single-input copies of positions {covered}; leave-one-out maxes excluding {loo}')
    th.save({'sets':sets,'W':W.detach()},path.replace('.pt',f'_snapped.pt'))
