"""
download_datasets.py
====================
Downloads bearing and gearbox fault datasets from Kaggle.
Tries multiple dataset sources with fallback.
"""
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

def try_download(api, slugs, target_dir, name):
    """Try downloading from multiple Kaggle dataset slugs."""
    os.makedirs(target_dir, exist_ok=True)
    for slug in slugs:
        try:
            print(f"  Trying: {slug}")
            api.dataset_download_files(slug, path=target_dir, unzip=True)
            total = sum(len(f) for _, _, f in os.walk(target_dir))
            print(f"  SUCCESS — Downloaded {total} files")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    print(f"  ERROR: Could not download {name} from any source.")
    return False

def download():
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated.\n")
    
    # 1. CWRU Bearing Dataset — try multiple sources
    cwru_dir = os.path.join(DATA_DIR, 'cwru')
    print(f"[1/2] Downloading CWRU Bearing Dataset to: {cwru_dir}")
    cwru_slugs = [
        'brjapon/cwru-bearing-datasets',
        'astrollama/cwru-case-western-reserve-university-dataset',
        'javadseraj/cwru-bearing-fault-data-set',
        'jinliyekaggle/cwru-bearing-dataset-mat',
        'sufian79/cwru-mat-full-dataset',
    ]
    cwru_ok = try_download(api, cwru_slugs, cwru_dir, "CWRU Bearing Dataset")
    
    # 2. Gearbox Fault Dataset
    gear_dir = os.path.join(DATA_DIR, 'gearbox')
    print(f"\n[2/2] Downloading Gearbox Fault Dataset to: {gear_dir}")
    gear_slugs = [
        'brjapon/gearbox-fault-diagnosis',
        'hetarthchopra/gearbox-fault-detection-dataset-phm-2009-nasa',
    ]
    gear_ok = try_download(api, gear_slugs, gear_dir, "Gearbox Fault Dataset")
    
    print("\n" + "="*50)
    if cwru_ok and gear_ok:
        print("All datasets downloaded successfully!")
    else:
        print("Some datasets failed. Check above for details.")
    
    return cwru_ok, gear_ok

if __name__ == '__main__':
    download()
