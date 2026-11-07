"""
Dataset preparation and download helper for Sign Language Translation benchmarks:
- RWTH-PHOENIX-Weather 2014T (German DGS)
- How2Sign (American ASL)
- OpenASL (American ASL)
- iSign (Indian ISL)
"""

import os
import argparse
import urllib.request
import tarfile
import json
from data.sample_data import generate_mock_phoenix_dataset

PHOENIX_URL = "https://www-i6.informatik.rwth-aachen.de/~koller/HA-DV-Files/phoenix-2014T.v3.tar.gz"

def download_and_extract_phoenix(output_dir="./data/raw/phoenix2014t"):
    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, "phoenix-2014T.v3.tar.gz")

    print(f"=== PHOENIX-2014T Download & Extraction ===")
    print(f"Source URL: {PHOENIX_URL}")
    print(f"Target directory: {output_dir}")

    if not os.path.exists(tar_path):
        print(f"Downloading PHOENIX-2014T archive (~1.2 GB)...")
        try:
            urllib.request.urlretrieve(PHOENIX_URL, tar_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"[Warning] Direct download failed or network is disabled: {e}")
            print("Generating sample verification dataset for testing and reproducibility...")
            generate_mock_phoenix_dataset(output_dir)
            return

    if os.path.exists(tar_path):
        print("Extracting archive...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=output_dir)
            print("Extraction completed.")
        except Exception as e:
            print(f"Extraction error: {e}. Fallback to mock dataset.")
            generate_mock_phoenix_dataset(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Download and setup SLT benchmark datasets.")
    parser.add_argument("--dataset", type=str, default="phoenix2014t",
                        choices=["phoenix2014t", "how2sign", "openasl", "isign", "mock"],
                        help="Dataset to prepare.")
    parser.add_argument("--output_dir", type=str, default="./data/raw/phoenix2014t",
                        help="Output directory.")
    args = parser.parse_args()

    if args.dataset == "mock" or args.dataset == "phoenix2014t":
        download_and_extract_phoenix(args.output_dir)
    else:
        print(f"For {args.dataset}, please refer to official dataset repositories:")
        print("  - How2Sign: https://how2sign.github.io/")
        print("  - OpenASL: https://github.com/SL-D/OpenASL")
        print("  - iSign: https://github.com/ISign-Benchmark/iSign")

if __name__ == "__main__":
    main()

