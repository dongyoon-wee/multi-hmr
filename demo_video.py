import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import subprocess
import argparse
import torch
import numpy as np
import random
import zipfile
import time
import json
import re

from utils import MEAN_PARAMS, SMPLX_DIR
from demo import load_model, get_camera_parameters, forward_model, open_image
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

def prepare_inference():
    # SMPL-X models
    smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
    if not os.path.isfile(smplx_fn):
        print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
        print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
        print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
        print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
        print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
        assert NotImplementedError
    else:
            print('SMPLX found')
            
    # SMPL mean params download
    if not os.path.isfile(MEAN_PARAMS):
        print('Start to download the SMPL mean params')
        os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
        print('SMPL mean params have been succesfully downloaded')
    else:
        print('SMPL mean params is already here')

    # Loading
    model = load_model(args.model_name)
    return model

def process_video(args):
    vid = args.vid
    vid_name = os.path.splitext(args.vid)[0]
    frame_folder = os.path.join(args.img_folder, vid_name)
    video_path = os.path.join(args.vid_folder, vid)
    os.makedirs(frame_folder, exist_ok=True)

    # extract frames
    command = ['ffmpeg', '-i', video_path]
    if args.init_sec > 0:
        command.extend(['-ss', str(args.init_sec)])
    if args.duration_sec > 0:
        command.extend(['-t', str(args.duration_sec)])
    command.extend(['-vf', f"fps={args.fps}"])
    command.append(f"{frame_folder}/frame%05d.jpg")
    subprocess.run(command, check=True)

    return frame_folder, vid_name

def process_frames(l_frame_paths, out_folder, model, model_name):
    l_duration = []
    start_process_frames = time.time()
    for i, frame_path in enumerate(tqdm(l_frame_paths)):
        save_file_name = os.path.join(out_folder, f"{Path(frame_path).stem}_{model_name}")
        input_path = os.path.join(args.img_folder, frame_path)

        duration, humans = infer_img(input_path, model)
        l_duration.append(duration)

        expand_if_1d = lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) and x.ndim==1 else x
        for i, human in enumerate(humans):
            human_out = map_human(human)
            human_dict = {k: expand_if_1d(v) for k, v in human_out.items()}
            meta_fn = save_file_name+'_'+str(i)+'.npz'
            np.savez(meta_fn, **human_dict)

    print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()}")
    print(f'Total process time={time.time() - start_process_frames}')

    output_zip = out_folder + '.zip'
    zip_npz_files(out_folder, output_zip)

def map_human(human):
    human_out = {
            'global_orient': human['rotvec'][0].cpu().numpy(),
            'body_pose': human['rotvec'][1:22].cpu().numpy(),
            'left_hand_pose': human['rotvec'][22:37].cpu().numpy(),
            'right_hand_pose': human['rotvec'][37:52].cpu().numpy(),
            'jaw_pose': human['rotvec'][52].cpu().numpy(),
            'leye_pose': human['rotvec'][52].cpu().numpy(),
            'reye_pose': human['rotvec'][52].cpu().numpy(),
            'betas': human['shape'].cpu().numpy(),
            'expression': human['expression'].cpu().numpy(),
            'transl': human['transl'].cpu().numpy()
    }
    return human_out

def infer_img(img_path, m):
    # Get input in the right format for the model
    img_size = m.img_size
    x, img_pil_nopad = open_image(img_path, img_size)
    # Get camera parameters
    p_x, p_y = None, None
    K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)
    # Make model predictions
    start = time.time()
    outputs = forward_model(model, x, K,
                            det_thresh=args.det_thresh,
                            nms_kernel_size=args.nms_kernel_size)
    duration = time.time() - start

    return duration, outputs

def zip_npz_files(folder_path, output_zip):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over all the files in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npz'):
                    # Create the full file path
                    file_path = os.path.join(root, file)
                    # Add the file to the zip archive
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

    print(f"All .npz files from {folder_path} have been zipped into {output_zip}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', type=str, help='name of the video (with extension)')
    parser.add_argument('--img_folder', type=str, default='/mnt/input_img')
    parser.add_argument('--vid_folder', type=str, default='/mnt/input_vid')
    parser.add_argument('--out_folder', type=str, default='/mnt/output')
    parser.add_argument('--init_sec', type=int, default=0)
    parser.add_argument('--duration_sec', type=int, default=0)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument("--model_name", type=str, default='multiHMR_896_L')
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=float, default=3)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
    parser.add_argument("--inference_id", type=str)
    args = parser.parse_args()

    dict_args = vars(args)

    assert torch.cuda.is_available()

    # check format
    assert os.path.splitext(args.vid)[1] == '.mp4', 'Only mp4 format is supported'
    
    frame_folder, vid_name = process_video(args)
    print(f'complete to extract {vid_name} / {args.fps} FPS at {frame_folder}')

    # Manage Input/Output 
    suffixes = ('.jpg', '.jpeg', '.png', '.webp')
    list_input_path = [os.path.join(frame_folder, file) for file in os.listdir(frame_folder) if file.endswith(suffixes) and file[0] != '.']
    assert len(list_input_path) > 0, 'No frames to infer'
    print(f'The number of images to infer: {len(list_input_path)}')

    if args.inference_id:
        inference_id = args.inference_id
    else:
        inference_id = vid_name
    out_folder = f'{args.out_folder}/{inference_id}'
    os.makedirs(out_folder, exist_ok=True)

    meta_data = {
        "fps": args.fps
    }
    meta_path = os.path.join(out_folder, 'meta.json')
    with open(meta_path, "w") as meta_file:
        json.dump(meta_data, meta_file)
    
    model = prepare_inference()
    print(f'complete to preparing {args.model_name} inference')

    process_frames(list_input_path, out_folder, model, args.model_name)
    print(f'complete to process {vid_name} at {out_folder}')