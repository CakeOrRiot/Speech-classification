import os
import sys
import argparse
import requests
import tarfile
import shutil

file_name = "dev-clean.tar.gz"
url = f"https://openslr.elda.org/resources/60/{file_name}"


def setup():
    download = not os.path.exists(file_name)
    if download:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    os.makedirs("data/raw", exist_ok=True)
    with tarfile.open(file_name) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f)
    shutil.rmtree("data/raw/LibriTTS")
    os.rename("LibriTTS", f"data/raw/LibriTTS")
    os.makedirs("data/processed/audio", exist_ok=True)
    os.makedirs("data/processed/text", exist_ok=True)


if __name__ == "__main__":
    setup()
