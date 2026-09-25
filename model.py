import torch
from torch import nn
from einops import rearrange

from MusicFMExtractor import MusicFMExtractor
from MSAM import MSAM

from spect_convert import bt_to_musicfm, build_freq_map


class BeatFM(nn.Module):
    """BeatFM = frozen MusicFM -> MSAM -> FC classifier
    in : wav (B, samples), 24 kHz mono
    out: beat logits (B, T), downbeat logits (B, T), T at 25 fps
    """

    def __init__(self, layers=None, feat_dim=1024, hidden_dim=512, classifier="mlp",
                 dilations=(1, 2, 4, 8), kernel_size=3, embed_dim=16, input_type = "wav"):
        super().__init__()
        self.extractor = MusicFMExtractor(layers=layers)
        self.input_type = input_type                                                        # NEW
        if input_type == "spect":                                                           # NEW
            width_bt, freq_map = build_freq_map()                 # fixed; rebuilt on load, not saved
            self.register_buffer("width_bt", width_bt, persistent=False)
            self.register_buffer("freq_map", freq_map, persistent=False)
        elif input_type != "wav":                                                           # NEW
            raise ValueError(f"unknown input_type: {input_type}")
        n_layers = self.extractor.n_layers
        self.msam = MSAM(n_layers, dilations, kernel_size, embed_dim)

        # paper: "fully connected layers" (depth / activation not specified)
        in_dim = n_layers * feat_dim
        if classifier == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        elif classifier == "linear":
            self.classifier = nn.Linear(in_dim, 2)
        else:
            raise ValueError(f"unknown classifier: {classifier}")

    def features(self, x):
        if self.input_type == "wav":
            return self.extractor(x)
        mel = bt_to_musicfm(x, self.extractor.musicfm.stat, self.width_bt, self.freq_map)
        return self.extractor.forward_mel(mel)
    
    def forward(self, x):                                   # (B, samples)
        h = self.features(x)                             # (B, N, F, T)   Eq.(1)(2)
        h = self.msam(h)                                      # (B, N, F, T)   Eq.(3)-(10)
        h = rearrange(h, "b n f t -> b t (n f)")              # (B, T, N*F)
        logits = self.classifier(h)                           # (B, T, 2)
        return logits[..., 0], logits[..., 1]                 # beat, downbeat
