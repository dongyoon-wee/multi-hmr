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

The installation has been tested with python3.9 and CUDA 12.1.

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

## Pre-trained models
We provide multiple pre-trained checkpoints.
Here is a list of their associated features.
Once downloaded you need to place them into `$HOME/models/multiHMR`.

| modelname                     | training data                     | backbone | resolution | runtime (ms) | PVE-3PDW-test | PVE-EHF | PVE-BEDLAM-val | comment |
|-------------------------------|-----------------------------------|----------|------------|--------------|----------|---------|---------|---------|
| [multiHMR_896_L](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_896_L.pt)    | BEDLAM+AGORA+CUFFS+UBody                      | ViT-L    | 896x896    |    126      | 89.9  | 42.2 | 56.7 | initial ckpt |
| [multiHMR_672_L](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L.pt)    | BEDLAM+AGORA+CUFFS+UBody                      | ViT-L    | 672x672    |     74      | 94.1  | 37.0 | 58.6 | longer training |
| [multiHMR_672_B](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_B.pt)    | BEDLAM+AGORA+CUFFS+UBody                      | ViT-B    | 672x672    |     43      | 94.0  | 43.6 | 67.2 | longer training |
| [multiHMR_672_S](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_S.pt)    | BEDLAM+AGORA+CUFFS+UBody                      | ViT-S    | 672x672    |     29      | 102.4 | 49.3 | 78.9 | longer training |
<!-- | [multiHMR_896_L_universal](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/)    | BEDLAM+AGORA+CUFFS+UBody                      | ViT-L    | 896x896    |    126       | 88.2 | 39.4 | 57.9 | longer training | -->
<!-- | [multiHMR_1288_L_bedlam](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/)    | BEDLAM(train+val)                      | ViT-L    | 1288x1288    |    ?       | ? | ? | ckpt used for BEDLAM leaderboard | -->
<!-- | [multiHMR_1288_L_agora](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/)    | BEDLAM(train+val)+AGORA(train+val)                      | ViT-L    | 1288x1288    |    ?       | ? | ? | ckpt used for AGORA leaderboard | -->

We compute the runtime on GPU V100-32GB.

## Training Multi-HMR
We provide code for training Multi-HMR using a single GPU on BEDLAM-training and evaluating it on BEDLAM-validation, EHF and 3DPW-test.

Activate environnement
```bash
source .multihmr/bin/activate
export PYTHONPATH=`pwd`
```

### Preprocessing BEDLAM
The first thing that you need to do is to download the BEDLAM dataset (6fps version) and place the files into ```data/BEDLAM```
The data structure of the directory should look like this:
```bash
data/BEDLAM
      |
      |---validation
                  |
                  |---20221018_1_250_batch01hand_zoom_suburb_b_6fps
                                                              |
                                                              |---png
                                                                  |
                                                                  |---seq_000000
                                                                              |
                                                                              |---seq_000000_0000.png
                                                                              ...
                                                                              |---seq_000000_0235.png
                                                                  ...
                                                                  |---seq_000249
                  ...
                  |---20221019_3-8_250_highbmihand_orbit_stadium_6fps
      |---training
              |
              |---20221010_3_1000_batch01hand_6fps
              ...
              |---20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps
      |---all_npz_12_training
              |
              |---20221010_3_1000_batch01hand_6fps.npz
              ...
              |---20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.npz
      |---all_npz_12_validation
            |
            |---20221018_1_250_batch01hand_zoom_suburb_b_6fps.npz
            ...
            |---20221019_3-8_250_highbmihand_orbit_stadium_6fps.npz
```

We need to build the annotation files for the training and validation sets. It may takes around 20 minutes for bulding the pkl files depending on your CPU.
```bash
python3.9 datasets/bedlam.py "create_annots(['validation', 'training'])"
```
You will get two files ```data/bedlam_validation.pkl``` and ```data/bedlam_training.pkl```.

### Checking annotations
Visualize the annotation of a specific image.
```bash
python3.9 datasets/bedlam.py "visualize(split='validation', i=1500)"
```
It will create a file ```bedlam_validation_15000.jpg``` where you can see the RGB image on the left side and the RGB image with meshes overlayed on the right side.

### (Optional) Creating jpg files to fast data-loading
BEDLAM is composed of PNG files and loading them could be a bit slow depending our your infrastucture.
The following command will generate one jpg file for each png file with maximal resolution of 1280.
It may take a while because BEDLAM has more than 300k images. You can run the command lines on some specific subdirectories to speed-up the generation of jpg files. You can chose the target size of your choice.
```bash
# Can be slow
python3.9 datasets/bedlam.py "create_jpeg(root_dir='data/BEDLAM', target_size=1280)

# Or parallelize
python3.9 datasets/bedlam.py "create_jpeg(root_dir='data/BEDLAM/validation/20221019_3-8_250_highbmihand_orbit_stadium_6fps', target_size=1280)
...
python3.9 datasets/bedlam.py "create_jpeg(root_dir='data/BEDLAM/training/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps', target_size=1280)
```

### Checking the data-loading time
You can check the quality of your dataloader by running the command above. It will use the png version of BEDLAM.
```bash
python3.9 datasets/bedlam.py "dataloader(split='validation', batch_size=16, num_workers=4, extension='png', img_size=1280, n_iter=100)"
```

### Preprocessing additional validation sets
We also provide code for evaluating on EHF and 3DPW.
Run the command for bulding the annotation fiel for EHF.
```bash
python3.9 datasets/ehf.py "create_annots()"
python3.9 datasets/ehf.py "visualize(i=10)"
```
And for 3DPW. Please download SMPL-male and SMPL-female models, put them into ```models/smpl/SMPL_MALE.pkl``` and ```models/smpl/SMPL_FEMALE.pkl```. And ```smplx2smpl.pkl``` is mandatory for moving from SMPLX to SMPL.
```bash
python3.9 datasets/threedpw.py "create_annots()"
python3.9 datasets/threedpw.py "visualize(i=1011)"
```

### Training on BEDLAM-train
We provide the command for training on BEDLAM-train at resolution 336 on a single GPU.
```bash
# python command
CUDA_VISIBLE_DEVICES=1 python3.9 train.py \
--backbone dinov2_vits14 \
--img_size 336 \
-j 4 \
--batch_size 32 \
-iter 10000 \
--max_iter 500000 \
--name multi-hmr_s_336
```
To decrease data-loading time use ```--extension jpg --res 1280```

### Evaluating BEDLAM-val / EHF-test / 3DPW-test
Above command is for evaluating a pretrained ckpt on validation sets.
```bash
CUDA_VISIBLE_DEVICES=0 python3.9 train.py \
--eval_only 1 \
--backbone dinov2_vitl14 \
--img_size 896 \
--val_data EHF THREEDPW BEDLAM \
--val_split test test validation \
--val_subsample 1 20 25 \
--pretrained models/multiHMR/multiHMR_896_L.pt
```
Either check the log or open the tensorboard for checking the results.

## License
The code is distributed under the CC BY-NC-SA 4.0 License.\
See [Multi-HMR LICENSE](Multi-HMR_License.txt), [Checkpoint LICENSE](Checkpoint_License.txt) and [Example Data LICENSE](Example_Data_License.txt) for more information.

## Citing
If you find this code useful for your research, please consider citing the following paper:
```bibtex
@inproceedings{multi-hmr2024,
    title={Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot},
    author={Baradel*, Fabien and 
            Armando, Matthieu and 
            Galaaoui, Salma and 
            Br{\'e}gier, Romain and 
            Weinzaepfel, Philippe and 
            Rogez, Gr{\'e}gory and
            Lucas*, Thomas
            },
    booktitle={ECCV},
    year={2024}
}
```