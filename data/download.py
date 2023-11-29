import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os
import urllib

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_alzheimer(data_dir):
    full_path = stage_path(data_dir, "alzheimer")
    download_and_extract(
        #'https://drive.google.com/uc?id=19sy6lw5RlQ2Hd-akXNQWfKov_yvj8IfV&export=download',
        'https://drive.google.com/u/0/uc?id=1185kjVWYf_HvjnbJlPQ0p0jIURT9i9yK&export=download',
        os.path.join(full_path, "HCP_YA.zip"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    download_alzheimer(args.data_dir)