"""
generate_noisy_spectrograms.py
==============================
Creates realistic noisy versions of vibration signals by adding
Gaussian noise, periodic interference, and amplitude modulation
to the REAL raw data, then generating STFT spectrograms.

These simulate actual factory sensor degradation.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import stft
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'real_world_samples')

SEGMENT_LENGTH = 1024
STFT_NPERSEG = 256
STFT_NOVERLAP = 128
SAMPLING_RATE = 48000
NOISY_SAMPLES_PER_CLASS = 20   # Generate 20 noisy samples per class


def add_realistic_noise(signal, snr_db=5):
    """Add multi-component noise simulating real factory conditions."""
    np.random.seed(None)  # Different noise each call
    
    # 1. Gaussian white noise (electrical interference)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    gaussian_noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    # 2. Periodic interference (adjacent machinery at random frequency)
    freq = np.random.uniform(50, 500)
    t = np.arange(len(signal)) / SAMPLING_RATE
    periodic_noise = 0.15 * np.std(signal) * np.sin(2 * np.pi * freq * t)
    
    # 3. Random amplitude modulation (variable load)
    mod_freq = np.random.uniform(1, 10)
    amplitude_mod = 1.0 + 0.3 * np.sin(2 * np.pi * mod_freq * t)
    
    noisy = signal * amplitude_mod + gaussian_noise + periodic_noise
    return noisy


def signal_to_spectrogram(signal_segment, sr=SAMPLING_RATE):
    """Convert a 1D vibration segment into a 2D STFT spectrogram."""
    _, _, Zxx = stft(
        signal_segment, fs=sr,
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, nfft=STFT_NPERSEG
    )
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    return magnitude_db


def save_spectrogram_image(spectrogram, output_path):
    """Save spectrogram as 224x224 PNG."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(spectrogram, cmap='viridis', shading='gouraud')
    ax.set_axis_off()
    fig.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60, flush=True)
    print("GENERATING REAL NOISY SPECTROGRAMS", flush=True)
    print("=" * 60, flush=True)
    
    # ── CWRU Bearing Data ──
    cwru_map = {
        'Time_Normal_1_098.mat': 'normal_bearing',
        'B007_1_123.mat': 'ball_fault',
        'IR007_1_110.mat': 'inner_race_fault',
        'OR007_6_1_136.mat': 'outer_race_fault',
    }
    
    mat_dir = os.path.join(RAW_DIR, 'cwru', 'raw')
    
    for filename, class_name in cwru_map.items():
        filepath = os.path.join(mat_dir, filename)
        mat_data = loadmat(filepath)
        de_key = [k for k in mat_data.keys() if 'DE_time' in k][0]
        signal = mat_data[de_key].flatten()
        
        # Create output subdir
        cls_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(NOISY_SAMPLES_PER_CLASS):
            # Random starting position
            max_start = len(signal) - SEGMENT_LENGTH
            start = np.random.randint(0, max_start)
            segment = signal[start:start + SEGMENT_LENGTH]
            
            # Add noise at varying SNR levels (3-10 dB)
            snr = np.random.uniform(3, 10)
            noisy_segment = add_realistic_noise(segment, snr_db=snr)
            
            spec = signal_to_spectrogram(noisy_segment)
            img_path = os.path.join(cls_dir, f'{class_name}_noisy_{i:03d}.png')
            save_spectrogram_image(spec, img_path)
        
        print(f"  {class_name:20s}: {NOISY_SAMPLES_PER_CLASS} noisy spectrograms", flush=True)
    
    # ── Gearbox Data ──
    gear_map = {
        'Healthy': 'healthy_gear',
        'BrokenTooth': 'broken_tooth',
    }
    
    gear_dir = os.path.join(RAW_DIR, 'gearbox')
    
    for condition, class_name in gear_map.items():
        cond_dir = os.path.join(gear_dir, condition)
        files = sorted(os.listdir(cond_dir))
        
        # Concatenate all files for more signal variety
        all_signal = []
        for f in files:
            df = pd.read_csv(os.path.join(cond_dir, f))
            all_signal.append(df['a1'].values)
        signal = np.concatenate(all_signal)
        
        cls_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(NOISY_SAMPLES_PER_CLASS):
            max_start = len(signal) - SEGMENT_LENGTH
            start = np.random.randint(0, max_start)
            segment = signal[start:start + SEGMENT_LENGTH]
            
            snr = np.random.uniform(3, 10)
            noisy_segment = add_realistic_noise(segment, snr_db=snr)
            
            spec = signal_to_spectrogram(noisy_segment)
            img_path = os.path.join(cls_dir, f'{class_name}_noisy_{i:03d}.png')
            save_spectrogram_image(spec, img_path)
        
        print(f"  {class_name:20s}: {NOISY_SAMPLES_PER_CLASS} noisy spectrograms", flush=True)
    
    # Also create single representative images at root level for backward compatibility
    for cls_name in list(cwru_map.values()) + list(gear_map.values()):
        cls_dir = os.path.join(OUTPUT_DIR, cls_name)
        imgs = sorted(os.listdir(cls_dir))
        if imgs:
            import shutil
            src = os.path.join(cls_dir, imgs[0])
            dst = os.path.join(OUTPUT_DIR, f'{cls_name}_noisy.png')
            shutil.copy2(src, dst)
    
    total = NOISY_SAMPLES_PER_CLASS * 6
    print(f"\nTotal: {total} noisy spectrograms generated from REAL vibration data", flush=True)
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}", flush=True)


if __name__ == '__main__':
    generate()
