import torch
import torch.nn as nn
from .layers import GradientReversal

class TransformerFusion(nn.Module):
    def __init__(self, d=2048, heads=8, layers=2, drop=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 3, d))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4*d,
                                         dropout=drop, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.Dropout(drop))

    def forward(self, sem, lr, hr):
        x = torch.stack([sem, lr, hr], 1) + self.pos  
        x = self.norm(self.enc(x)).mean(1)     
        return self.out(x)

class GatedAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gs = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.Sigmoid()) for _ in range(3)])
    def forward(self, sem, lr, hr):
        return [feat*g(feat) for feat, g in zip([sem, lr, hr], self.gs)]

class DomainDiscriminator(nn.Module):
    def __init__(self, d=2048, n_dom=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_dom))
    def forward(self, x): return self.net(x)

class ExpertFusionTransformerDA(nn.Module):
    def __init__(self, sem_d=640, lr_d=2048, hr_d=2048, hidden=2048, n_dom=10):
        super().__init__()
        self.sem_proj = nn.Linear(sem_d, hidden)
        self.lr_proj  = nn.Linear(lr_d,  hidden)
        self.hr_proj  = nn.Linear(hr_d,  hidden)
        self.sem_cls  = nn.Sequential(nn.Linear(sem_d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self.lr_cls   = nn.Sequential(nn.Linear(lr_d,  256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self.hr_cls   = nn.Sequential(nn.Linear(hr_d,  256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

        self.gate = GatedAttention(hidden)
        self.fuse = TransformerFusion(hidden)
        self.clf  = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1))
        self.dom_clf = DomainDiscriminator(hidden, n_dom)

    def forward(self, sem, lr, hr, domain_adapt=False, alpha=1.0):
        sem_p, lr_p, hr_p = self.sem_proj(sem), self.lr_proj(lr), self.hr_proj(hr)
        sem_g, lr_g, hr_g = self.gate(sem_p, lr_p, hr_p)
        fused = self.fuse(sem_g, lr_g, hr_g)
        out   = self.clf(fused)
        
        sem_o = self.sem_cls(sem)
        lr_o  = self.lr_cls(lr)
        hr_o  = self.hr_cls(hr)

        dom_o = None
        if domain_adapt:
            rev = GradientReversal.apply(fused, alpha)
            dom_o = self.dom_clf(rev)
        return out, dom_o, sem_o, lr_o, hr_o