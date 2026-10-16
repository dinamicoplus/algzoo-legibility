import torch as th, torch.nn as nn, torch.nn.functional as Fn
N=10
class ReLURNN(nn.Module):
    def __init__(s,H):
        super().__init__(); s.H=H
        s.Wi=nn.Parameter(th.randn(H)*1.0); s.Wh=nn.Parameter(th.randn(H,H)/H**0.5); s.Wo=nn.Parameter(th.randn(N,H)/H**0.5)
    def forward(s,x,tau=None):
        h=x.new_zeros(x.shape[0],s.H)
        for t in range(x.shape[1]): h=th.relu(h@s.Wh.T+x[:,t:t+1]*s.Wi)
        return h@s.Wo.T
class MaxRNN(nn.Module):
    """h_i(t) = max_k ( src_k + w_ik ),  src = [h(t-1), x_t, 0],  w_ik = -softplus(theta_ik) <= 0.
    w ~ 0 : source k is ON;  w very negative : source k is OFF.  Readout is linear, as in ARC's model."""
    def __init__(s,H,zero_src=True):
        super().__init__(); s.H=H; s.S=H+1+(1 if zero_src else 0); s.zero_src=zero_src
        s.theta=nn.Parameter(th.randn(H,s.S)*1.0+1.5)   # starts mostly ~off-ish, w in roughly (-4,-0.3)
        s.Wo=nn.Parameter(th.randn(N,H)/H**0.5)
    def gates(s): return -Fn.softplus(s.theta)
    def step(s,h,xt,tau,w):
        parts=[h,xt[:,None]]+([th.zeros_like(xt)[:,None]] if s.zero_src else [])
        src=th.cat(parts,1)                     # (B,S)
        z=src[:,None,:]+w[None]                 # (B,H,S)
        if not tau: return z.max(-1).values
        mx=z.max(-1,keepdim=True).values.detach()
        return mx[...,0]+tau*th.log(th.exp(((z-mx)/tau).clamp(min=-30)).sum(-1))
    def forward(s,x,tau=None,w=None):
        w=s.gates() if w is None else w
        h=x.new_zeros(x.shape[0],s.H)
        if s.zero_src: h=h  # start at 0
        for t in range(x.shape[1]): h=s.step(h,x[:,t],tau,w)
        return h@s.Wo.T
    def hidden(s,x,w=None):
        w=s.gates() if w is None else w
        h=x.new_zeros(x.shape[0],s.H); hs=[]
        for t in range(x.shape[1]): h=s.step(h,x[:,t],None,w); hs.append(h)
        return th.stack(hs,1)
def target(x): return x.argsort(1)[:,-2]
