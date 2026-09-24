import sys
from pathlib import Path

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # musicfm_bridge/
from MusicFMExtractor import MusicFMExtractor
from MSAM import MSAM

AUDIO_PATH = "/disk1/taegum/mnt/Video2Music/demo_ref/v2m_demo.mp3"
STAT_PATH = "/disk1/taegum/mnt/musicfm/data/msd_stats.json"
MODEL_PATH = "/disk1/taegum/mnt/musicfm/data/pretrained_msd.pt"
SR = 24000                                                # MusicFM expects 24 kHz mono


if __name__ == "__main__":
    y, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)     # whole track
    wav = torch.from_numpy(y).unsqueeze(0)                # (1, samples)
    sec = wav.shape[1] / SR

    ext = MusicFMExtractor(stat_path=STAT_PATH, model_path=MODEL_PATH)
    msam = MSAM(n_layers=ext.n_layers)

    h = ext(wav)                                          # (1, N, F, T)
    h_tilde = msam(h)                                     # (1, N, F, T)

    print(f"audio   : {sec:.2f} s -> expected T ~ {sec * 25:.0f}")
    print(f"h       : {tuple(h.shape)}")
    print(f"h_tilde : {tuple(h_tilde.shape)}")
