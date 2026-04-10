import urllib.request
import os
import sys

url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
output = "data/plantvillage_raw.zip"

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rDownloading: {percent:.2f}% ({downloaded / (1024*1024):.2f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.2f} MB (size unknown)")
        sys.stdout.flush()

print(f"Starting download from {url}...")
if not os.path.exists("data"):
    os.makedirs("data")

try:
    urllib.request.urlretrieve(url, output, download_progress)
    print("\nDownload finished successfully!")
except Exception as e:
    print(f"\nError: {e}")
