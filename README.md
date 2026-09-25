# BeatFM 재현

**BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model** (ICME 2025) 재현 코드입니다.
목표는 논문 Table I의 GTZAN 점수를 다시 얻는 것입니다.

| GTZAN (BeatFM, MusicFM) | F-measure | CMLt | AMLt |
|---|---|---|---|
| Beat | 89.1 | 80.6 | 93.5 |
| Downbeat | 79.6 | 74.4 | 88.7 |

**현재 상태:** 입력으로 **Beat This가 제공하는 spectrogram**(`<dataset>.npz`)을 받을 수 있게 했습니다(`--input spect`). 오디오 입력(`--input wav`)도 그대로 쓸 수 있습니다. GPU에서 학습 → validation → 체크포인트 → test(DBN) 점수표까지 확인했고, 본 학습(fold 0, 30 epoch)을 돌리고 있습니다.

---

## 구조

```
wav (B, samples) @ 24 kHz                  --input wav   : MusicFM이 직접 mel 계산
Beat This spect (B, T_bt, 128) @ 50 fps    --input spect : spect_convert.py로 MusicFM mel(100 fps)로 변환
  → MusicFMExtractor   frozen MusicFM, 층별 hidden state를 쌓음      → h  (B, N=13, F=1024, T)   Eq.(1)(2)
  → MSAM               temporal / frequency / channel attention      → h̃ (B, N, F, T)           Eq.(3)-(10)
  → classifier         (B, T, N·F) → Linear → ReLU → Linear(→2)      → beat, downbeat logits (B, T)
  → DBN (madmom)       beat / downbeat 시각
```
T는 25 fps (MusicFM 출력 해상도 그대로, 1 frame = 40 ms).

| 파일 | 내용 |
|---|---|
| `MusicFMExtractor.py` | frozen MusicFM. `forward(wav)` / `forward_mel(mel)`. `.train()`을 불러도 eval 모드 유지 |
| `spect_convert.py` | Beat This spectrogram → MusicFM 입력 mel 변환 (아래 "Beat This spectrogram 입력" 참고) |
| `mmnpz.py` | `.npz` memory-map 로더 (Beat This 코드, MIT) |
| `MSAM.py` | `MSConv`, `TemporalAggregation`, `FrequencyAggregation`, `channelAggregation`, `MSAM` |
| `model.py` | `BeatFM` = extractor → MSAM → classifier (`input_type="wav"` / `"spect"`) |
| `data.py` | 오디오 캐시 또는 Beat This `.npz` 읽기, 15 s clip / 5 s overlap, ±2 frame soft label, 분할 |
| `make_manifest.py` | 데이터셋별 라벨·fold와 오디오(`--input wav`) 또는 spectrogram(`--input spect`) 경로를 모아 manifest 생성 |
| `train.py` | Lightning 학습 + 곡 전체 test (DBN, mir_eval) |
| `scripts/` | 확인용 스크립트 (서버 절대경로 포함). `check_convert.py` / `check_align.py` 등 변환 검증 |
| `third_party/musicfm` | MusicFM 원본 (git submodule) |

---

## 설치

```bash
git clone --recursive https://github.com/Hwang-Tae-Gum/beatFM.git
cd beatFM

conda create -n musicfm python=3.10 -y
conda activate musicfm
# S3 서버 드라이버는 CUDA 12.4까지 지원 → cu124 빌드 필요
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers<5" einops librosa pytorch_lightning mir_eval tqdm setuptools cython
pip install git+https://github.com/CPJKU/madmom      # PyPI 0.16은 최신 python에서 import 실패

# MusicFM 가중치 (MSD)
wget -P third_party/musicfm/data https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
wget -P third_party/musicfm/data https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt
```

> **transformers 5.x에서는 MusicFM이 깨집니다** (`KeyError: 'hidden_states'` — 5.x의 `Wav2Vec2ConformerEncoder`가 hidden states를 반환하지 않음). 반드시 4.x.

---

## 실행

### Beat This spectrogram 입력 (현재 사용)

```bash
# 1. manifest: Beat This .npz 기준 (Harmonix 포함 3,313곡)
python make_manifest.py --input spect --out manifest_spect.csv

# 2. 학습 + test (fold 0, 30 epoch)
nohup python -u train.py --manifest manifest_spect.csv --input spect --fold 0 --gpu 0 \
    --max-epochs 30 > logs/spect_fold0_ep30.log 2>&1 &
```
- spectrogram 위치: `make_manifest.py`의 `SPECT_ROOT` (`<dataset>.npz`, key `<name>/track`)
- 오디오 캐시가 필요 없습니다 (`--cache-dir` 불필요).

### 오디오 입력

```bash
# 1. manifest (dataset, name, audio, spect, annotation, fold, has_downbeats)
python make_manifest.py --out manifest.csv

# 2. smoke test
python train.py --manifest manifest.csv --cache-dir /path/to/cache \
    --limit-tracks 2 --batch-size 2 --max-epochs 1 --num-workers 0 --gpu 0

# 3. 본 학습 (fold k = 8-fold CV에서 test로 뺄 fold)
python train.py --manifest manifest.csv --cache-dir /path/to/cache --fold 0 --gpu 0
```
- 첫 실행 시 모든 오디오를 24 kHz mono float16 `.npy`로 캐시합니다 (약 15 GB).
- 끝나면 best 체크포인트(val_loss 기준)로 test를 돌리고 데이터셋별 F / CMLt / AMLt 표를 출력합니다.
- 체크포인트에는 frozen MusicFM(1.3 GB)을 빼고 저장합니다 (~80 MB). 로드 시 MusicFM 가중치 파일에서 다시 채웁니다.
- 로그: `runs/fold{k}/logs/` (CSV)

---

## 데이터

라벨과 8-fold 분할은 Beat This annotation set(`/disk1/jaehoon/dataset_store/beat_this_annotations`)을 사용합니다. 데이터셋 구성과 분할은 BeatFM 논문(Sec. IV-A) 방식입니다: Ballroom / Hainsworth / SMC는 8-fold 중 fold k를 test, GTZAN은 test only, 나머지 곡의 10%를 val.

| 데이터셋 | 📄 용도 | 곡 수: spect | 곡 수: 오디오 | 비고 |
|---|---|---|---|---|
| Beatles | train | 179 | 179 | 빈 annotation 1곡(Revolution 9) 제외 |
| RWC Popular | train | 100 | 100 | |
| **Harmonix** | train | **911** | 0 | 정렬된 오디오가 없어 오디오 입력으로는 사용 불가 |
| Ballroom | 8-fold | 685 | 672 | |
| Hainsworth | 8-fold | 222 | 222 | |
| SMC | 8-fold | 217 | 217 | downbeat 라벨 없음 → downbeat loss에서 제외 |
| GTZAN | test only | 999 | 993 | |
| **합계** | | **3,313** | 2,383 | |

fold 0 기준 (spect): train 1,956 / val 217 / test 1,140곡 (GTZAN 999 + Ballroom 86 + Hainsworth 28 + SMC 27).

---

## Beat This spectrogram 입력

두 모델의 입력 spectrogram은 설정이 다릅니다.

| | Beat This (저장된 값) | MusicFM (원래 입력) |
|---|---|---|
| 샘플레이트 / n_fft / hop | 22.05 kHz / 1024 / 441 (**50 fps**) | 24 kHz / 2048 / 240 (**100 fps**) |
| mel | slaney, 30–11000 Hz, **magnitude**를 합산 | htk, 0–12000 Hz, **power**를 합산 |
| 크기 | `log1p(1000 · x)` | `10·log10(x)` → MSD 통계로 정규화 |

`spect_convert.py`는 Beat This 단계를 거꾸로 풀고 MusicFM 단계를 다시 적용합니다.
```
bt_to_mag        log1p 역변환 (정확)
mel_bt_to_fm     band 폭으로 나눔 → 제곱 → MusicFM band 중심으로 보간 → × MusicFM band 폭 (band 안 평탄 가정)
power_to_musicfm 10·log10 + DB_OFFSET(34.49) → (x − 6.77) / 18.42
upsample_time    50 → 100 fps 선형 보간
```

**검증** (GTZAN, 같은 곡을 오디오로 만든 MusicFM mel과 비교, `scripts/check_convert.py`, `scripts/check_align.py`)
- mel 오차 2.77 dB. band별로 중간 대역 1.2–1.4 dB, 최저역 4.8 dB, 최고역 8.3 dB. Beat This에 30 Hz 미만, 11 kHz 이상 정보가 없어서입니다.
- MusicFM hidden state cosine: layer 0 **0.734**, layer 6 **0.787**, layer 12 **0.873**. 참고로 같은 오디오를 10 ms 밀었을 때 약 0.95입니다.
- 시간 정렬: 오디오 입력과 spect 입력의 feature를 ±3 frame 밀어 비교했을 때 lag 0에서 최대(대칭). frame 수도 일치(30 s → 750).
- 앞뒤 ±2 frame을 보는 선형 매핑을 데이터로 학습하면 cosine 0.93–0.96까지 올라갑니다(GTZAN 10곡 학습, 10곡 평가). 아직 적용하지 않았습니다.

---

## 논문 명시 사항 vs 직접 정한 값

**📄 논문에 명시 (Sec. III, IV) → 그대로 구현**
- frozen FM, 층별 출력 concat → `h ∈ R^{b×n×f×t}`
- MS-Conv: M = 4, dilation [1, 2, 4, 8], 병렬 → concat → MLP → sigmoid; frequency는 같은 구조, 파라미터 비공유
- channel: C-AvgPool → Conv Q/K/V → softmax(QKᵀ/√C)V → Conv → sigmoid
- `Attn = Attn_t · Attn_f · Attn_c`, `h̃ = h + Attn * h`
- BCE, beat·downbeat 동시 학습, 라벨 ±2 frame 확장 (weight 0.5, 0.25)
- 15 s clip, 5 s overlap, 같은 곡은 train/val 중 한쪽에만
- Adam, lr 3e-4, batch 16, val loss 20 epoch patience early stopping
- DBN 후처리, 곡 전체 test, mir_eval, 70 ms

**논문에 없어서 직접 정한 값 (전부 인자로 변경 가능)** — 검토 부탁드립니다

| 항목 | 현재 값 | 인자 / 위치 |
|---|---|---|
| MusicFM 체크포인트 | MSD | `MusicFMExtractor.py` |
| 사용 층 N | 13 (conv 출력 + conformer 12) — 논문은 "Encoder 1…n" | `--layers` |
| 출력 fps | 25 (업샘플 없음) → ±2 frame = ±80 ms | `data.py` |
| MS-Conv kernel / padding | 3 / 길이 유지 | `MSAM.py` |
| MS-Conv 채널 해석 | N(층)을 채널로, T 또는 F 방향 conv | `MSAM.py` |
| MS-Conv 뒤 MLP | 1×1 conv 2층, hidden = N, ReLU | `MSAM.py` |
| channel Q/K/V Conv | 층당 스칼라 → `Conv1d(1, C=16, 1)`, head 1개, 출력 `Conv1d(C, 1, 1)` | `MSAM.py` |
| classifier | N·F flatten → Linear(512) → ReLU → Linear(2) ("fully connected layers") | `--classifier mlp/linear` |
| beat/downbeat 출력 | 독립 | `model.py` |
| val 비율 | 학습 곡의 10% (곡 단위) | `--val-ratio` |
| weight decay / scheduler / augmentation | 없음 | — |
| DBN | beats_per_bar [3,4], 55–215 bpm, transition_lambda 100 | `train.py` |
| test 입력 | 곡 전체 한 번에 | `--chunk-sec` |
| 평가 시 앞부분 제외 | 5 s (mir_eval 관례) | `train.py` |
| precision | fp32 | `--precision` |
| spect 변환 dB offset | 34.49 (GTZAN 10곡으로 맞춤, 곡별 편차 0.14 dB) | `spect_convert.py` |
| DBN 입력 하한 | 1e-5 (madmom log(0) 방지) | `train.py` |

---

## 확인된 것 / 남은 것

- ✅ 실제 오디오(GTZAN 30 s) → `(1, 13, 1024, 751)`, MusicFM 가중치 보존, 학습 파라미터 6.8 M / frozen 329 M
- ✅ Beat This spectrogram 입력: 변환 검증, 시간 정렬, 15 s clip → 375 frame
- ✅ GPU(RTX A6000) 학습 → val → 체크포인트 → test(DBN) 점수표까지 동작
- ✅ 긴 곡 chunk 추론 로직 (`--chunk-sec`)
- ⏳ 본 학습: spect 입력, fold 0, 30 epoch (epoch당 약 13분)
