"""
Speech Direction Classifier using DWN (Differentiable Weightless Networks)
===========================================================================
Trains a lookup-table-based neural network to classify directional speech
commands (up/down/left/right/go/stop ...) from the Google Speech Commands dataset.

Input compression options
--------------------------
Raw audio is 32 000 samples/sec × 1 sec = 32 000 values.
At 5 bits per sample that would be 160 000 input bits – too many for a
reasonable LUT network.  Common strategies:

  1. Decimation  – take every N-th sample (fast, may alias).
  2. Block-avg   – average non-overlapping windows of N samples (low-pass).
  3. MFCC        – compute mel-frequency cepstral coefficients (~40 coeff ×
                   ~100 frames = 4 000 features).  Best for speech but
                   requires librosa/scipy.

We default to block-avg with TARGET_SAMPLES=200, giving
200 × 5 = 1 000 input bits – a practical size for the LUT layers.

Noise augmentation
------------------
The `_background_noise_` folder in the dataset contains ~7 long WAV files.
When USE_NOISE=True, each training sample is randomly mixed with a short
segment of a random noise file at a configurable SNR.
"""

import os
import sys
import random
import wave

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Make DWN importable from the local repo ──────────────────────────────────
# DWN_SRC = os.path.join(os.path.dirname(__file__), "DWN", "src")
# sys.path.insert(0, DWN_SRC)
import torch_dwn as dwn  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================

# Words that count as "direction" commands (only those actually present in the
# dataset will be used as classes).
DIRECTION_WORDS = ["up", "down", "left", "right", "go", "stop",
                   "forward", "backward", "follow"]

# Compression
TARGET_SAMPLES = 300          # samples per clip after compression
COMPRESSION    = "average"     # "decimate" | "average"

# Thermometer encoding
NUM_BITS = 15                  # bits per sample → input_size = TARGET_SAMPLES * NUM_BITS

# LUT architecture
N_LUT      = 2                 # number of inputs per LUT node
N_LAYERS   = 5                 # hidden LUT layers (all same width = input_size)

# Noise augmentation
USE_NOISE  = True              # mix background noise into training samples
NOISE_PROB = 0.5               # probability of adding noise per sample
NOISE_SNR_DB = 6.0            # signal-to-noise ratio in dB

# Data augmentation options
AUG_TIME_SHIFT = True         # randomly shift audio in time
TIME_SHIFT_MAX = 0.1          # max shift as fraction of length (e.g. 0.1 = 10%)
AUG_AMP_SCALE  = True         # randomly scale amplitude
AMP_SCALE_RANGE = (0.7, 1.3)  # min/max amplitude scaling

# Training
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15             # remaining 0.15 becomes test set
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def load_wav(path: str) -> np.ndarray:
    """Load a WAV file and return float32 mono samples normalised to [-1, 1]."""
    with wave.open(path, "rb") as wf:
        sw  = wf.getsampwidth()
        ch  = wf.getnchannels()
        n   = wf.getnframes()
        raw = wf.readframes(n)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype     = dtype_map.get(sw, np.int16)
    samples   = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if ch > 1:                          # mix down to mono
        samples = samples.reshape(-1, ch).mean(axis=1)

    if sw == 1:                         # 8-bit PCM: unsigned, 128 = silence
        samples = (samples - 128.0) / 128.0
    else:                               # 16/32-bit PCM: two's-complement signed
        max_val = float(2 ** (sw * 8 - 1))
        samples /= max_val

    return samples


def compress(samples: np.ndarray, target: int, method: str) -> np.ndarray:
    """Resize audio to `target` samples using the chosen method."""
    n = len(samples)
    if n == 0:
        return np.zeros(target, dtype=np.float32)

    if n < target:                      # zero-pad short clips
        samples = np.pad(samples, (0, target - n))
        n = target

    if method == "decimate":
        # Pick `target` evenly-spaced indices – fast but may alias at high decimation.
        indices = np.round(np.linspace(0, n - 1, target)).astype(int)
        return samples[indices].astype(np.float32)

    elif method == "average":
        # Average non-overlapping blocks → acts as low-pass filter before downsampling.
        block = n // target
        if block == 0:
            return np.resize(samples, target).astype(np.float32)
        return samples[: block * target].reshape(target, block).mean(axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unknown compression method: {method!r}. "
                         "Choose 'decimate' or 'average'.")


def mix_noise(signal: np.ndarray,
              noise_clips: list,
              snr_db: float) -> np.ndarray:
    """
    Mix a random segment of a random noise clip into `signal` at `snr_db` SNR.
    Returns a new float32 array (not clipped – the thermometer handles scaling).
    """
    noise = random.choice(noise_clips)
    length = len(signal)

    if len(noise) >= length:
        start     = random.randint(0, len(noise) - length)
        noise_seg = noise[start : start + length]
    else:
        noise_seg = np.resize(noise, length)

    sig_power   = np.mean(signal    ** 2) + 1e-9
    noise_power = np.mean(noise_seg ** 2) + 1e-9
    #scale       = np.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
    scale = 1/snr_db
    return (signal + scale * noise_seg).astype(np.float32)


# ===================== DATA AUGMENTATION =====================
def random_time_shift(samples: np.ndarray, max_frac: float) -> np.ndarray:
    """Randomly shift audio left/right by up to max_frac of its length."""
    n = len(samples)
    max_shift = int(n * max_frac)
    if max_shift < 1:
        return samples
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return samples
    elif shift > 0:
        return np.pad(samples, (shift, 0), mode='constant')[:n]
    else:
        return np.pad(samples, (0, -shift), mode='constant')[-shift:n-shift]

def random_amplitude_scale(samples: np.ndarray, scale_range: tuple) -> np.ndarray:
    """Randomly scale amplitude by a factor in scale_range (min, max), then clamp to [-1, 1]."""
    scale = random.uniform(*scale_range)
    scaled = samples * scale
    scaled = np.clip(scaled, -1.0, 1.0)
    return scaled.astype(np.float32)


# =============================================================================
# DATASET LOADING
# =============================================================================

def find_direction_folders(dataset_path: str) -> dict:
    """Return {folder_name: class_index} for direction words found in the dataset."""
    lower_words = {w.lower() for w in DIRECTION_WORDS}
    present = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and d.lower() in lower_words
    ])
    return {name: idx for idx, name in enumerate(present)}


def load_noise_clips(dataset_path: str) -> list:
    """Load all WAV files from _background_noise_ as full-length float arrays."""
    noise_dir = os.path.join(dataset_path, "_background_noise_")
    clips = []
    if not os.path.isdir(noise_dir):
        print("  No _background_noise_ folder found – noise augmentation disabled.")
        return clips
    for fname in os.listdir(noise_dir):
        if fname.endswith(".wav"):
            try:
                clips.append(load_wav(os.path.join(noise_dir, fname)))
            except Exception as exc:
                print(f"  Warning: could not load noise file {fname}: {exc}")
    print(f"  Loaded {len(clips)} noise file(s) from _background_noise_.")
    return clips


def load_dataset(dataset_path: str,
                 class_map: dict,
                 noise_clips: list,
                 use_noise: bool,
                 is_train: bool) -> tuple:
    """
    Load and preprocess all WAV files for the given class map.

    Noise augmentation is only applied during training (`is_train=True`).
    Returns (numpy array of shape [N, TARGET_SAMPLES], list of int labels).
    """
    samples_list, labels_list = [], []


    for folder, label in class_map.items():
        folder_path = os.path.join(dataset_path, folder)
        wav_files   = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        print(f"  [{folder:>10}]  {len(wav_files)} files")

        for fname in wav_files:
            try:
                sig = load_wav(os.path.join(folder_path, fname))
                sig = compress(sig, TARGET_SAMPLES, COMPRESSION)

                # Data augmentation (train only)
                if is_train:
                    if AUG_TIME_SHIFT and random.random() < 0.5:
                        sig = random_time_shift(sig, TIME_SHIFT_MAX)
                    if AUG_AMP_SCALE and random.random() < 0.5:
                        sig = random_amplitude_scale(sig, AMP_SCALE_RANGE)

                # Noise augmentation (train only)
                if use_noise and is_train and noise_clips and random.random() < NOISE_PROB:
                    sig = mix_noise(sig, noise_clips, NOISE_SNR_DB)

                samples_list.append(sig)
                labels_list.append(label)

            except Exception:
                pass  # skip corrupted files

    return np.array(samples_list, dtype=np.float32), labels_list


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device : {DEVICE}")
    if DEVICE.type != "cuda":
        print("WARNING: DWN LUTLayer requires CUDA. "
              "The EFDFunction forward/backward are GPU-only.")

    # ── Download dataset ──────────────────────────────────────────────────────
    import kagglehub
    dataset_path = kagglehub.dataset_download("yashdogra/speech-commands")
    print(f"Dataset: {dataset_path}\n")

    # ── Discover direction classes ────────────────────────────────────────────
    class_map = find_direction_folders(dataset_path)
    if not class_map:
        raise RuntimeError(
            f"No direction folders found in dataset. "
            f"Available folders: {os.listdir(dataset_path)}"
        )
    num_classes = len(class_map)
    idx_to_name = {v: k for k, v in class_map.items()}
    print(f"Direction classes ({num_classes}): {list(class_map.keys())}\n")

    # ── Load noise clips ──────────────────────────────────────────────────────
    noise_clips = load_noise_clips(dataset_path) if USE_NOISE else []
    print()

    # ── Load all audio samples ────────────────────────────────────────────────
    print("Loading audio files...")
    raw_samples, raw_labels = load_dataset(
        dataset_path, class_map, noise_clips,
        use_noise=USE_NOISE, is_train=True   # noise applied here; we re-split below
    )
    print(f"Total samples: {len(raw_samples)}\n")

    # ── Shuffle and split ─────────────────────────────────────────────────────
    indices = np.random.permutation(len(raw_samples))
    raw_samples = raw_samples[indices]
    raw_labels  = [raw_labels[i] for i in indices]

    n_total = len(raw_samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    x_train = to_tensor(raw_samples[:n_train])
    x_val   = to_tensor(raw_samples[n_train : n_train + n_val])
    x_test  = to_tensor(raw_samples[n_train + n_val :])

    y_train = torch.tensor(raw_labels[:n_train],              dtype=torch.long)
    y_val   = torch.tensor(raw_labels[n_train:n_train+n_val], dtype=torch.long)
    y_test  = torch.tensor(raw_labels[n_train+n_val:],        dtype=torch.long)

    print(f"Split  →  train: {len(x_train)}  |  val: {len(x_val)}  |  test: {len(x_test)}\n")

    # ── Thermometer encoding (fit on training data only) ──────────────────────
    print(f"Encoding with DistributiveThermometer ({NUM_BITS} bits, "
          f"compression={COMPRESSION!r}, target={TARGET_SAMPLES} samples)...")

    thermometer = dwn.DistributiveThermometer(NUM_BITS).fit(x_train)

    def encode(x: torch.Tensor) -> torch.Tensor:
        return thermometer.binarize(x).flatten(start_dim=1)

    x_train = encode(x_train)
    x_val   = encode(x_val)
    x_test  = encode(x_test)

    input_size = x_train.size(1)   # TARGET_SAMPLES * NUM_BITS  (e.g. 5 000)
    print(f"Input size after encoding: {input_size} bits  "
          f"({TARGET_SAMPLES} samples × {NUM_BITS} bits)\n")

    # ── Build model ───────────────────────────────────────────────────────────
    # Architecture:
    #   5 × LUTLayer(input_size → input_size, n=N_LUT)
    #   GroupSum(k=num_classes)  – groups input_size outputs into num_classes scores
    layers = []
    layer_width = input_size * 4
    layers.append(dwn.LUTLayer(input_size, layer_width, n=N_LUT, mapping="learnable"))
    for i in range(1, N_LAYERS):
        mapping = "random" #"learnable" if i == 0 else "random"
        layers.append(dwn.LUTLayer(layer_width, layer_width, n=N_LUT, mapping=mapping))

    layers.append(dwn.GroupSum(k=num_classes, tau=1 / 0.3))

    model = nn.Sequential(*layers).to(DEVICE)
    print(f"Model: {N_LAYERS} × LUTLayer({input_size}→{layer_width}, n={N_LUT}) + "
          f"GroupSum(k={num_classes})\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=(EPOCHS//2)-1)  #step_size=14

    # Move tensors to device
    x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)
    x_val,   y_val   = x_val.to(DEVICE),   y_val.to(DEVICE)
    x_test,  y_test  = x_test.to(DEVICE),  y_test.to(DEVICE)

    # ── Training loop ─────────────────────────────────────────────────────────
    def evaluate(x, y):
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        return (pred == y).float().mean().item()

    print("Starting training...\n")
    n_train_samples = x_train.size(0)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        perm = torch.randperm(n_train_samples, device=DEVICE)
        total_loss, correct, total = 0.0, 0, 0

        for i in range(0, n_train_samples, BATCH_SIZE):
            idx      = perm[i : i + BATCH_SIZE]
            bx, by   = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            out  = model(bx)
            loss = cross_entropy(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            correct    += (out.argmax(1) == by).sum().item()
            total      += len(by)

        #scheduler.step()

        train_acc = correct / total
        val_acc   = evaluate(x_val, y_val)
        print(f"Epoch {epoch+1:>3}/{EPOCHS}  "
              f"loss={total_loss/total:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    test_acc = evaluate(x_test, y_test)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    # Per-class breakdown
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1).cpu()
    y_test_cpu = y_test.cpu()
    print("\nPer-class test accuracy:")
    for idx, name in idx_to_name.items():
        mask = y_test_cpu == idx
        if mask.sum() > 0:
            acc = (preds[mask] == idx).float().mean().item()
            print(f"  {name:>12}: {acc:.4f}  ({mask.sum().item()} samples)")


if __name__ == "__main__":
    main()
