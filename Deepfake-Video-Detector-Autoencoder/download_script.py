import requests
import shutil
import os

# Using a known public Xception Deepfake model from dubm/ucf-f
url = "https://huggingface.co/dubm/ucf-f/resolve/main/xception-best.pth"
target_path = "models/trained/xception_deepfake.pth"

print(f"Downloading substitute model from {url}...")

try:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    print("Download complete!")
except Exception as e:
    print(f"Error: {e}")
