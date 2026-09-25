import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
from musicfm.model.musicfm_25hz import MusicFM25Hz

MUSICFM_DIR = Path(__file__).resolve().parent / "third_party" / "musicfm"
DEFAULT_STAT_PATH = MUSICFM_DIR / "data" / "msd_stats.json"
DEFAULT_MODEL_PATH = MUSICFM_DIR / "data" / "pretrained_msd.pt"

class MusicFMExtractor(nn.Module):
    def __init__(self, stat_path=DEFAULT_STAT_PATH, model_path=DEFAULT_MODEL_PATH, layers=None, freeze=True):
        super().__init__()
        self.musicfm = MusicFM25Hz(is_flash=False, stat_path=stat_path, model_path=model_path)
        self.layers = list(layers) if layers is not None else list(range(13))
        self.freeze = freeze
        if freeze:
            for p in self.musicfm.parameters():
                p.requires_grad = False
            self.musicfm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.musicfm.eval()
        return self

    @property
    def n_layers(self):
        return len(self.layers)

    def forward(self, wav):
        with torch.set_grad_enabled(not self.freeze):
            _, hidden_emb = self.musicfm.get_predictions(wav)
        h = torch.stack([hidden_emb[i] for i in self.layers], dim = 1)
        return h.transpose(2, 3)

    def forward_mel(self, mel):
        """mel: (B, 128, T_mel) normalized MusicFM input at 100 fps -> (B, N, F, T_mel / 4)"""
        with torch.set_grad_enabled(not self.freeze):
            _, hidden_emb = self.musicfm.encoder(mel)
        h = torch.stack([hidden_emb[i] for i in self.layers], dim=1)
        return h.transpose(2, 3)

if __name__ == "__main__":
    import argparse
    import librosa

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to an audio file")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio, sr=24000, mono=True)   # whole track, 24 kHz mono
    wav = torch.from_numpy(y).unsqueeze(0)                  # (1, samples)
    sec = wav.shape[1] / sr

    ext = MusicFMExtractor()                                # default weights in third_party/musicfm/data
    h = ext(wav)
    print(f"{sec:.2f} s -> expected T ~ {sec * 25:.0f}")
    print(h.shape)
