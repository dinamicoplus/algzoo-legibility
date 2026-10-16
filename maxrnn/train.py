import os
import sys,time,math,torch as th,torch.nn.functional as Fn
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from models import *
kind,H,steps,seed=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
th.manual_seed(seed); th.set_num_threads(1); th.set_flush_denormal(True)
m=ReLURNN(H) if kind=='relu' else MaxRNN(H)
opt=th.optim.Adam(m.parameters(),lr=1e-2 if kind=='max' else 4e-3)
sched=th.optim.lr_scheduler.LambdaLR(opt,lambda s: min(1,(s+1)/500)*0.5*(1+math.cos(math.pi*s/steps))+0.01)
xv=th.randn(20000,N,generator=th.Generator().manual_seed(999)); t0=time.time()
for s in range(steps):
    x=th.randn(1024,N)
    tau = None if kind=='relu' else max(0.3*(0.02/0.3)**(s/(0.8*steps)),0.02)
    L=m(x,tau); loss=Fn.cross_entropy(L,target(x))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (s+1)%(steps//10)==0:
        with th.no_grad(): a=(m(xv).argmax(1)==target(xv)).float().mean().item()
        print(f'{kind} H={H} seed={seed} step {s+1} loss {loss.item():.3f} val acc (hard) {a:.4f}  tau {tau}  {time.time()-t0:.0f}s',flush=True)
th.save(m.state_dict(),os.path.join(os.path.dirname(os.path.abspath(__file__)),f'{kind}_{H}_{seed}.pt'))
