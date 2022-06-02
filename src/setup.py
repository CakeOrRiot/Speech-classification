import os
import sys
import argparse
import requests
import tarfile

file_name = 'dev-clean.tar.gz'
url = f'https://openslr.elda.org/resources/60/{file_name}'


def setup():
    download = not os.path.exists(file_name)
    if download:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    os.makedirs('data/raw', exist_ok=True)
    with tarfile.open(file_name) as f:
        f.extractall()
    os.rename('LibriTTS', f'data/raw/LibriTTS')
    os.makedirs('data/processed/audio', exist_ok=True)
    os.makedirs('data/processed/text', exist_ok=True)



if __name__ == '__main__':
    setup()
