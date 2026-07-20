"""
generate_spectrograms.py
========================
Converts raw 1D vibration signals from the CWRU Bearing Dataset (.mat files)
and Gearbox Fault Dataset (.csv files) into 2D spectrogram images using STFT.

The CWRU .mat files contain keys like 'X123_DE_time' (drive-end accelerometer).
The Gearbox .csv files have 4 acceleration columns (a1-a4) with a header row.

Usage:
    python generate_spectrograms.py
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import stft
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

# ── Configuration ────────────────────────────────────────────────────────────
SEGMENT_LENGTH = 1024
STFT_NPERSEG = 256
STFT_NOVERLAP = 128
SAMPLING_RATE = 48000       # CWRU 48k dataset
SAMPLES_PER_CLASS = 300
IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
SPEC_DIR = os.path.join(BASE_DIR, '..', 'data', 'spectrograms')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# CWRU fault mapping: filename prefix -> class name
CWRU_MAP = {
    'Time_Normal': 'normal_bearing',
    'B007': 'ball_fault',
    'B014': 'ball_fault',
    'B021': 'ball_fault',
    'IR007': 'inner_race_fault',
    'IR014': 'inner_race_fault',
    'IR021': 'inner_race_fault',
    'OR007': 'outer_race_fault',
    'OR014': 'outer_race_fault',
    'OR021': 'outer_race_fault',
}


def signal_to_spectrogram(signal_segment, sr=SAMPLING_RATE):
    """Convert a 1D vibration signal segment into a 2D STFT spectrogram."""
    _, _, Zxx = stft(
        signal_segment, fs=sr,
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, nfft=STFT_NPERSEG
    )
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    return magnitude_db


def save_spectrogram_image(spectrogram, output_path):
    """Save a 2D spectrogram array as a 224x224 PNG image."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(spectrogram, cmap='viridis', shading='gouraud')
    ax.set_axis_off()
    fig.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def process_cwru():
    """Process all CWRU .mat files into spectrograms."""
    mat_dir = os.path.join(RAW_DIR, 'cwru', 'raw')
    if not os.path.exists(mat_dir):
        print(f"[SKIP] CWRU directory not found: {mat_dir}")
        return

    print("\n=== Processing CWRU Bearing Dataset ===")
    class_counts = {}

    for filename in sorted(os.listdir(mat_dir)):
        if not filename.endswith('.mat'):
            continue

        # Determine class from filename prefix
        class_name = None
        for prefix, cname in CWRU_MAP.items():
            if filename.startswith(prefix):
                class_name = cname
                break
        if class_name is None:
            continue

        if class_name not in class_counts:
            class_counts[class_name] = 0

        if class_counts[class_name] >= SAMPLES_PER_CLASS:
            continue

        # Load .mat file
        filepath = os.path.join(mat_dir, filename)
        mat_data = loadmat(filepath)

        # Find the drive-end accelerometer key (*_DE_time)
        de_key = None
        for key in mat_data.keys():
            if 'DE_time' in key:
                de_key = key
                break
        if de_key is None:
            continue

        signal = mat_data[de_key].flatten()

        # Create output directory
        class_dir = os.path.join(SPEC_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Segment and generate spectrograms
        step = SEGMENT_LENGTH // 2  # 50% overlap
        for start in range(0, len(signal) - SEGMENT_LENGTH, step):
            if class_counts[class_name] >= SAMPLES_PER_CLASS:
                break

            segment = signal[start:start + SEGMENT_LENGTH]
            spec = signal_to_spectrogram(segment)

            idx = class_counts[class_name]
            img_path = os.path.join(class_dir, f"{class_name}_{idx:04d}.png")
            save_spectrogram_image(spec, img_path)
            class_counts[class_name] += 1

        print(f"  {filename:30s} -> {class_name:20s} (total: {class_counts[class_name]})")

    print("\nCWRU Summary:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:20s}: {count} spectrograms")


def process_gearbox():
    """Process gearbox fault CSV files into spectrograms."""
    gear_dir = os.path.join(RAW_DIR, 'gearbox')
    if not os.path.exists(gear_dir):
        print(f"[SKIP] Gearbox directory not found: {gear_dir}")
        return

    print("\n=== Processing Gearbox Fault Dataset ===")
    condition_map = {
        'Healthy': 'healthy_gear',
        'BrokenTooth': 'broken_tooth',
    }

    for condition, class_name in condition_map.items():
        cond_dir = os.path.join(gear_dir, condition)
        if not os.path.exists(cond_dir):
            print(f"  [SKIP] {cond_dir} not found")
            continue

        class_dir = os.path.join(SPEC_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img_count = 0
        for filename in sorted(os.listdir(cond_dir)):
            if img_count >= SAMPLES_PER_CLASS:
                break
            filepath = os.path.join(cond_dir, filename)
            try:
                df = pd.read_csv(filepath)
                # Use first accelerometer channel (a1)
                signal = df['a1'].values

                step = SEGMENT_LENGTH // 2
                for start in range(0, len(signal) - SEGMENT_LENGTH, step):
                    if img_count >= SAMPLES_PER_CLASS:
                        break
                    segment = signal[start:start + SEGMENT_LENGTH]
                    spec = signal_to_spectrogram(segment)
                    img_path = os.path.join(class_dir, f"{class_name}_{img_count:04d}.png")
                    save_spectrogram_image(spec, img_path)
                    img_count += 1
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

        print(f"  {condition:15s} -> {class_name:15s}: {img_count} spectrograms")


def split_dataset(train_ratio=0.70, val_ratio=0.15):
    """Split spectrograms into train/validation/test sets."""
    print("\n=== Splitting Dataset ===")
    np.random.seed(42)

    classes = [d for d in os.listdir(SPEC_DIR)
               if os.path.isdir(os.path.join(SPEC_DIR, d))]

    for split in ['train', 'validation', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(SPEC_DIR, cls)
        images = sorted([f for f in os.listdir(cls_path) if f.endswith('.png')])
        np.random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': images[:n_train],
            'validation': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            dest_dir = os.path.join(DATA_DIR, split_name, cls)
            for img_name in split_imgs:
                shutil.copy2(
                    os.path.join(cls_path, img_name),
                    os.path.join(dest_dir, img_name)
                )
            print(f"  {cls:20s} | {split_name:10s} | {len(split_imgs):4d} images")


if __name__ == '__main__':
    print("=" * 60)
    print("SPECTROGRAM GENERATION PIPELINE")
    print("=" * 60)

    process_cwru()
    process_gearbox()
    split_dataset()

    print("\n" + "=" * 60)
    print("COMPLETE — Spectrograms saved to:", os.path.abspath(SPEC_DIR))
    print("Train/Val/Test splits saved to:", os.path.abspath(DATA_DIR))
    print("=" * 60)
