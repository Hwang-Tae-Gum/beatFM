import torch
from torch import nn
import math



class MSConv(nn.Module):
    """
    Temporal usage:  in (B, N, T) -> out (B, N, T)   # after T-AvgPool: h.mean(dim=2)
    Frequency usage: in (B, N, F) -> out (B, N, F)   # after F-AvgPool: h.mean(dim=3)
    Returns MLP_out (pre-sigmoid); apply sigmoid outside.
    """

    def __init__(self, n_layers, dilations=(1, 2, 4, 8), kernel_size=3, hidden=None):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(n_layers, n_layers, kernel_size,
                      dilation=d, padding=d * (kernel_size - 1) // 2)   # keep T (or F) length
            for d in dilations
        ])
        hidden = hidden or n_layers
        M = len(dilations)
        # per-position MLP (1x1 conv): M*N -> N at each time step t
        self.mlp = nn.Sequential(
            nn.Conv1d(M * n_layers, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, n_layers, 1),
        )

    def forward(self, x): # (B, N, T)
        outs = [conv(x) for conv in self.convs] # M x (B, N, T) eq.(4)
        x = torch.cat(outs, dim=1) # (B, M*N, T) eq.(5) Concat
        return self.mlp(x) # (B, N, T) eq.(5) MLP

class TemporalAggregation(nn.Module):

    def __init__(self, n_layers, dilations=(1, 2, 4, 8), kernel_size= 3):
        super().__init__()
        self.msconv = MSConv(n_layers, dilations, kernel_size)

    def forward(self, h):
        h_t = h.mean(dim=2)
        a = self.msconv(h_t)
        att_t = torch.sigmoid(a)
        return att_t.unsqueeze(2)

class FrequencyAggregation(nn.Module):
    def __init__(self, n_layers, dilations=(1, 2, 4, 8), kernel_size=3):
        super().__init__()
        self.msconv = MSConv(n_layers, dilations, kernel_size)

    def forward(self, h):
        h_f = h.mean(dim = 3)
        a = self.msconv(h_f)
        att_f = torch.sigmoid(a)
        return att_f.unsqueeze(3)

class channelAggregation(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.C = embed_dim
        self.q = nn.Conv1d(1, embed_dim, kernel_size=1)
        self.k = nn.Conv1d(1, embed_dim, kernel_size=1)
        self.v = nn.Conv1d(1, embed_dim, kernel_size=1)
        self.out = nn.Conv1d(embed_dim, 1, kernel_size = 1)

    def forward(self, h):
        h_c = h.mean(dim=(2, 3))
        x = h_c.unsqueeze(1)

        Q = self.q(x).transpose(1, 2)
        K = self.k(x).transpose(1, 2)
        V = self.v(x).transpose(1, 2)

        scores = Q @ K.transpose(1, 2) / math.sqrt(self.C)
        probs = torch.softmax(scores, dim=-1)
        z = probs @ V

        a = self.out(z.transpose(1, 2))
        attn_c = torch.sigmoid(a).squeeze(1)
        return attn_c[:, :, None, None]

class MSAM(nn.Module):
    def __init__(self, n_layers=13, dilations=(1, 2, 4, 8), kernel_size=3, embed_dim=16):
        super().__init__()
        self.temporal = TemporalAggregation(n_layers, dilations, kernel_size)
        self.frequency = FrequencyAggregation(n_layers, dilations, kernel_size)
        self.channel = channelAggregation(embed_dim)
    def forward(self, h):
        attn = self.temporal(h) * self.frequency(h) * self.channel(h)
        h_tilde = h + attn * h
        return h_tilde

if __name__ == "__main__":
    h = torch.randn(2, 13, 1024, 375)                     # (B, N, F, T)
    msam = MSAM(n_layers=13)
    h_tilde = msam(h)
    print(h_tilde.shape)                                  # torch.Size([2, 13, 1024, 375])
