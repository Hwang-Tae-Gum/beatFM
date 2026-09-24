import argparse
from collections import defaultdict
from pathlib import Path

import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data import FPS, HOP, ClipDataset, PieceDataset, build_cache, read_manifest, split_rows
from model import BeatFM


def masked_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def beat_scores(ref, est, trim_sec):
    ref = mir_eval.beat.trim_beats(np.asarray(ref, float), min_beat_time=trim_sec)
    est = mir_eval.beat.trim_beats(np.asarray(est, float), min_beat_time=trim_sec)
    _, cmlt, _, amlt = mir_eval.beat.continuity(ref, est)
    return {"F": mir_eval.beat.f_measure(ref, est), "CMLt": cmlt, "AMLt": amlt}   # 70 ms window


def first(batch):
    return batch[0]


class PLBeatFM(LightningModule):
    def __init__(self, lr=3e-4, layers=None, hidden_dim=512, classifier="mlp", embed_dim=16,
                 dbn=True, chunk_sec=0.0, trim_sec=5.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = BeatFM(layers=layers, hidden_dim=hidden_dim, classifier=classifier, embed_dim=embed_dim)
        self.dbn = None
        if dbn:
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor
            self.dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0,
                                                    fps=FPS, transition_lambda=100)
        self.test_results = []

    # ---- training ---------------------------------------------------------------
    def _losses(self, batch):
        beat, down = self.model(batch["wav"])                                # (B, T) each
        mask = batch["mask"].float()
        db_mask = mask * batch["has_downbeats"].float()[:, None]             # e.g. SMC has no downbeats
        return masked_bce(beat, batch["beat"], mask), masked_bce(down, batch["downbeat"], db_mask)

    def training_step(self, batch, _):
        lb, ld = self._losses(batch)
        self.log_dict({"train_loss": lb + ld, "train_beat": lb, "train_downbeat": ld},
                      on_step=False, on_epoch=True, batch_size=len(batch["wav"]))
        return lb + ld

    def validation_step(self, batch, _):
        lb, ld = self._losses(batch)
        self.log_dict({"val_loss": lb + ld, "val_beat": lb, "val_downbeat": ld},
                      on_epoch=True, prog_bar=True, batch_size=len(batch["wav"]))

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)

    # frozen MusicFM is reloaded from its own checkpoint -> don't store 1.3 GB per save
    def on_save_checkpoint(self, ckpt):
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("model.extractor.")}

    def on_load_checkpoint(self, ckpt):
        ckpt["state_dict"].update({k: v for k, v in self.state_dict().items() if k.startswith("model.extractor.")})

    # ---- testing on whole pieces --------------------------------------------------
    @torch.no_grad()
    def predict_piece(self, wav):
        """whole piece -> framewise probabilities (T,); chunk_sec > 0 splits long pieces"""
        n = -(-len(wav) // HOP) * HOP
        wav = F.pad(wav, (0, n - len(wav)))                                  # multiple of HOP -> T = n / HOP
        T, win = n // HOP, int(self.hparams.chunk_sec * FPS)
        if win <= 0 or T <= win:
            beat, down = self.model(wav[None])
            return beat[0].sigmoid(), down[0].sigmoid()
        border = win // 8
        beat, down = torch.zeros(T, device=wav.device), torch.zeros(T, device=wav.device)
        for s in list(range(0, T - win, win - 2 * border)) + [T - win]:
            b, d = self.model(wav[s * HOP:(s + win) * HOP][None])
            lo = 0 if s == 0 else border                                     # later windows overwrite borders
            beat[s + lo:s + win], down[s + lo:s + win] = b[0, lo:].sigmoid(), d[0, lo:].sigmoid()
        return beat, down

    def decode(self, beat, down):
        beat, down = beat.float().cpu().numpy(), down.float().cpu().numpy()
        if self.dbn is not None:
            act = np.stack([np.clip(beat - down, 0, 1), down], axis=1)       # madmom: (beat-only, downbeat)
            out = self.dbn(act)
            if len(out) == 0:
                return np.zeros(0), np.zeros(0)
            return out[:, 0], out[out[:, 1] == 1, 0]

        def peaks(p):                                                        # fallback: local maxima > 0.5
            return np.flatnonzero((p > 0.5) & (p >= np.roll(p, 1)) & (p >= np.roll(p, -1))) / FPS
        return peaks(beat), peaks(down)

    def test_step(self, item, _):
        est_beats, est_downbeats = self.decode(*self.predict_piece(item["wav"]))
        res = {"dataset": item["dataset"]}
        res.update({f"beat_{k}": v for k, v in beat_scores(item["beats"], est_beats, self.hparams.trim_sec).items()})
        if item["has_downbeats"]:
            res.update({f"downbeat_{k}": v for k, v in
                        beat_scores(item["downbeats"], est_downbeats, self.hparams.trim_sec).items()})
        self.test_results.append(res)

    def on_test_epoch_end(self):
        by_ds = defaultdict(list)
        for r in self.test_results:
            by_ds[r["dataset"]].append(r)
        keys = ["beat_F", "beat_CMLt", "beat_AMLt", "downbeat_F", "downbeat_CMLt", "downbeat_AMLt"]
        print(f"\n{'dataset':12s} {'n':>4s} " + " ".join(f"{k:>13s}" for k in keys))
        for ds, rs in sorted(by_ds.items()):
            vals = []
            for k in keys:
                xs = [r[k] for r in rs if k in r]
                vals.append(f"{100 * np.mean(xs):13.1f}" if xs else f"{'-':>13s}")
                if xs:
                    self.log(f"test/{ds}/{k}", float(np.mean(xs)))
            print(f"{ds:12s} {len(rs):4d} " + " ".join(vals))
        self.test_results.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache-dir", required=True, help="where 24 kHz mono .npy copies are stored")
    p.add_argument("--fold", type=int, default=0, help="8-fold CV fold held out for testing")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--val-ratio", type=float, default=0.1, help="fraction of training pieces for validation")
    p.add_argument("--classifier", choices=["mlp", "linear"], default="mlp")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="MusicFM hidden states (default: all 13)")
    p.add_argument("--no-dbn", action="store_true", help="peak picking instead of the DBN")
    p.add_argument("--chunk-sec", type=float, default=0.0, help="test-time window; 0 = whole piece")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default="32-true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--limit-tracks", type=int, default=0, help="debug: keep only N tracks per split")
    args = p.parse_args()

    seed_everything(args.seed, workers=True)
    rows = read_manifest(args.manifest)
    train_rows, val_rows, test_rows = split_rows(rows, args.fold, args.val_ratio, args.seed)
    if args.limit_tracks:
        train_rows, val_rows, test_rows = (r[:args.limit_tracks] for r in (train_rows, val_rows, test_rows))
    build_cache(train_rows + val_rows + test_rows, args.cache_dir)
    print(f"tracks  train {len(train_rows)}  val {len(val_rows)}  test {len(test_rows)}")

    train_dl = DataLoader(ClipDataset(train_rows, args.cache_dir), batch_size=args.batch_size,
                          shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_dl = DataLoader(ClipDataset(val_rows, args.cache_dir), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    test_dl = DataLoader(PieceDataset(test_rows, args.cache_dir), batch_size=1, collate_fn=first,
                         num_workers=args.num_workers)

    model = PLBeatFM(lr=args.lr, layers=args.layers, classifier=args.classifier,
                     dbn=not args.no_dbn, chunk_sec=args.chunk_sec)
    run_dir = Path(args.out_dir) / f"fold{args.fold}"
    ckpt = ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_loss", mode="min", save_top_k=1)
    cuda = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if cuda else "cpu",
        devices=[args.gpu] if cuda else 1,
        precision=args.precision,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), ckpt],
        logger=CSVLogger(run_dir, name="logs"),
    )
    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
