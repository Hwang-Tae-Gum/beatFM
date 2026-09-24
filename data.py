import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SR = 24000                  # MusicFM input sample rate
FPS = 25                    # MusicFM output frame rate
HOP = SR // FPS             # 960 samples per output frame
WIDEN = ((0, 1.0), (1, 0.5), (2, 0.25))   # paper: +-2 frames with weights 0.5 / 0.25

CV_DATASETS = ("ballroom", "hainsworth", "smc")   # 8-fold train + test
TEST_ONLY = ("gtzan",)                            # test only


def read_manifest(path):
    """CSV columns: dataset, name, audio, annotation, fold, has_downbeats"""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fold"] = int(r["fold"])
        r["has_downbeats"] = r["has_downbeats"] == "1"
    return rows


def split_rows(rows, fold, val_ratio=0.1, seed=0):
    """test = CV datasets' fold k + test-only datasets; the rest is split per piece into train / val"""
    test, pool = [], []
    for r in rows:
        if r["dataset"] in TEST_ONLY or (r["dataset"] in CV_DATASETS and r["fold"] == fold):
            test.append(r)
        else:
            pool.append(r)
    perm = np.random.default_rng(seed).permutation(len(pool))
    n_val = int(round(len(pool) * val_ratio))
    val = [pool[i] for i in perm[:n_val]]
    train = [pool[i] for i in perm[n_val:]]
    return train, val, test


def load_annotation(path):
    """lines: time [position_in_bar]; position 1 = downbeat"""
    arr = np.loadtxt(path, ndmin=2)
    beats = arr[:, 0]
    downbeats = beats[arr[:, 1] == 1] if arr.shape[1] > 1 else np.zeros(0)
    return beats, downbeats


def cache_path(row, cache_dir):
    return Path(cache_dir) / row["dataset"] / f"{row['name']}.npy"


def build_cache(rows, cache_dir):
    """resample every track once to 24 kHz mono and store as float16 .npy (memory-mapped later)"""
    todo = [r for r in rows if not cache_path(r, cache_dir).exists()]
    for r in tqdm(todo, desc="caching 24 kHz audio"):
        y, _ = librosa.load(r["audio"], sr=SR, mono=True)
        path = cache_path(r, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, y.astype(np.float16))


def load_audio(row, cache_dir):
    return np.load(cache_path(row, cache_dir), mmap_mode="r")


def frame_targets(times, n_frames, offset_sec=0.0):
    """soft targets at FPS: 1 on the annotated frame, 0.5 at +-1, 0.25 at +-2"""
    target = np.zeros(n_frames, np.float32)
    frames = np.round((np.asarray(times) - offset_sec) * FPS).astype(int)
    for shift, weight in WIDEN:
        for sign in ((1,) if shift == 0 else (1, -1)):
            idx = frames + sign * shift
            idx = idx[(idx >= 0) & (idx < n_frames)]
            np.maximum.at(target, idx, weight)
    return target


class ClipDataset(Dataset):
    """15 s clips with 5 s overlap (hop 10 s), paper Sec. IV-B"""

    def __init__(self, rows, cache_dir, clip_sec=15, hop_sec=10):
        self.rows, self.cache_dir = rows, cache_dir
        self.clip = clip_sec * SR                 # multiple of HOP -> exactly clip_sec * FPS frames
        self.n_frames = self.clip // HOP
        self.ann = [load_annotation(r["annotation"]) for r in rows]
        self.index = []
        for i, r in enumerate(rows):
            n = len(load_audio(r, cache_dir))
            starts = list(range(0, max(n - self.clip, 0) + 1, hop_sec * SR))
            if starts[-1] + self.clip < n:        # cover the tail
                starts.append(n - self.clip)
            self.index += [(i, s) for s in starts]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, start = self.index[k]
        row = self.rows[i]
        seg = np.asarray(load_audio(row, self.cache_dir)[start:start + self.clip], dtype=np.float32)
        wav = np.zeros(self.clip, np.float32)
        wav[:len(seg)] = seg
        mask = np.zeros(self.n_frames, bool)
        mask[:int(np.ceil(len(seg) / HOP))] = True
        beats, downbeats = self.ann[i]
        offset = start / SR
        return {
            "wav": torch.from_numpy(wav),                                                # (samples,)
            "beat": torch.from_numpy(frame_targets(beats, self.n_frames, offset)),       # (T,)
            "downbeat": torch.from_numpy(frame_targets(downbeats, self.n_frames, offset)),
            "mask": torch.from_numpy(mask),                                              # (T,)
            "has_downbeats": torch.tensor(row["has_downbeats"]),
        }


class PieceDataset(Dataset):
    """whole pieces for testing (paper: complete, unsegmented pieces)"""

    def __init__(self, rows, cache_dir):
        self.rows, self.cache_dir = rows, cache_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        wav = np.asarray(load_audio(row, self.cache_dir), dtype=np.float32)
        beats, downbeats = load_annotation(row["annotation"])
        return {
            "wav": torch.from_numpy(wav),
            "beats": beats,
            "downbeats": downbeats,
            "has_downbeats": row["has_downbeats"],
            "dataset": row["dataset"],
            "name": row["name"],
        }
