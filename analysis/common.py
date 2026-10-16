import torch as th, numpy as np
th.set_grad_enabled(False)
from alg_zoo import example_2nd_argmax
sd = example_2nd_argmax().state_dict()
Wi = sd['rnn.weight_ih_l0'][:,0].double().clone()   # (16,)
Wh = sd['rnn.weight_hh_l0'].double().clone()        # (16,16)
Wo = sd['linear.weight'].double().clone()           # (10,16)
d, n = 16, 10
# standardize: h_j -> h_j / |Wi_j| (ReLU positive-homogeneous)
s = Wi.abs().clone(); s[s<1e-6]=1.0
S = th.diag(s); Si = th.diag(1/s)
Wi_s = Wi / s
Wh_s = Si @ Wh @ S
Wo_s = Wo @ S
def run(x, Wi=Wi_s, Wh=Wh_s, Wo=Wo_s, hook=None):
    # x: (B,n) double. returns logits (B,n), hidden states (B,n+1,d)
    B = x.shape[0]; h = th.zeros(B, Wi.shape[0], dtype=x.dtype)
    hs=[h]
    for t in range(x.shape[1]):
        h = th.relu(h @ Wh.T + x[:,t:t+1]*Wi)
        if hook is not None: h = hook(t,h,x)
        hs.append(h)
    return h @ Wo.T, th.stack(hs,1)
def target(x):
    return x.argsort(1)[:,-2]
def acc(logits,x): return (logits.argmax(1)==target(x)).double().mean().item()
def sample(B, seed=0):
    g=th.Generator().manual_seed(seed); return th.randn(B,n,generator=g,dtype=th.float64)
# Re-standardize by activation RMS (more robust than |Wi| when Wi ~ 0)
_x = sample(100000, seed=123)
_, _H = run(_x, Wi, Wh, Wo)
r = _H[:,1:].pow(2).mean((0,1)).sqrt()
R = th.diag(r); Ri = th.diag(1/r)
Wi_r = Wi / r
Wh_r = Ri @ Wh @ R
Wo_r = Wo @ R
