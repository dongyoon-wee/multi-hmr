## Overview

This is a fork of the original [Multi-HMR](https://github.com/naver/multi-hmr) repository. The purpose of this fork is to add features and improvements for inferring video.

## Installation
First, you need to clone the repo.

We recommand to use virtual enviroment for running MultiHMR.
Please run the following lines for creating the environment with ```venv```:
```bash
python3.9 -m venv .multihmr
source .multihmr/bin/activate
pip install -r requirements.txt
```

Otherwise you can also create a conda environment.
```bash
conda env create -f conda.yaml
conda activate multihmr
```

The installation has been tested with python3.9 and CUDA 11.7.

Checkpoints will automatically be downloaded to `$HOME/models/multiHMR` the first time you run the demo code.

Besides these files, you also need to download the *SMPLX* model.
You will need the [neutral model](http://smplify.is.tue.mpg.de) for running the demo code.
Please go to the corresponding website and register to get access to the downloads section.
Download the model and place `SMPLX_NEUTRAL.npz` in `./models/smplx/`.

## Run Multi-HMR on images
The following command will run Multi-HMR on all images in the specified `--img_folder`, and save renderings of the reconstructions in `--out_folder`.
The `--model_name` flag specifies the model to use.
The `--extra_views` flags additionally renders the side and bev view of the reconstructed scene, `--save_mesh` saves meshes as in a '.npy' file.
```bash
python3.9 demo.py \
    --img_folder example_data \
    --out_folder demo_out \
    --extra_views 1 \
    --model_name multiHMR_896_L
```
## Run Multi-HMR on a video
The following command will run Multi-HMR on a single video named `--vid` in the specified `--vid_folder`, and save meshes as in a '.npz' file in `--out_folder`.
The `--model_name` flag specifies the model to use.
```bash
python3.9 demo_video.py \
    --vid input_video_name
    --vid_folder video_folder \
    --out_folder demo_out \
    --model_name multiHMR_896_L
```