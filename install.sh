#!/usr/bin/bash
conda create -n dps python=3.10
conda activate dps
pip install open_clip_torch
pip install wilds
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
pip install braceexpand
pip install webdataset
pip install h5py