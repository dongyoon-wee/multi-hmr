import os

# use gpu rendering if works
os.environ["PYOPENGL_PLATFORM"] = "egl" 
os.environ['EGL_DEVICE_ID'] = '0'
"""
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
"""

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
from demo import load_model, get_camera_parameters, forward_model, open_image, overlay_human_meshes
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from PIL import Image

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

def decode_video(args):
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

def encode_video(frame_folder, output_path, fps):
    command = ['ffmpeg', '-f', fps]
    command.extend(['-i', f"{frame_folder}/frame%05d.jpg"])
    command.extend(['-c:v', f"mpeg4 -pix_fmt yuv420p"])
    command.append(f"{output_path}")
    subprocess.run(command, check=True)

    return 0

def process_frames(l_frame_paths, out_folder, model, visualize, unique_color, fps):
    l_duration = []
    start_process_frames = time.time()

    param_folder = f'{out_folder}/param'
    vis_folder = f'{out_folder}/vis'
    os.makedirs(param_folder, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)

    for i, frame_path in enumerate(tqdm(l_frame_paths)):
        frame_name = f"{Path(frame_path).stem}"
        duration, humans, cam_param, img_pil_nopad = infer_img(frame_path, model)
        l_duration.append(duration)

        expand_if_1d = lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) and x.ndim==1 else x
        for i, human in enumerate(humans):
            human_out = map_human(human)
            human_dict = {k: expand_if_1d(v) for k, v in human_out.items()}

            param_path = os.path.join(param_folder , f"{frame_name}_{i}.npz")
            np.savez(param_path, **human_dict)

        if visualize:
            vis_path = os.path.join(vis_folder, f"{frame_name}.jpg")
            img_array = np.asarray(img_pil_nopad)
            img_pil_visu= Image.fromarray(img_array)
            pred_rend_array, _color = overlay_human_meshes(humans, cam_param, model, img_pil_visu, unique_color)
            Image.fromarray(pred_rend_array).save(vis_path)

    print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()}")
    print(f'Total process time={time.time() - start_process_frames}')

    output_zip = out_folder + '.zip'
    zip_npz_files(out_folder, output_zip)

    if visualize:
        encode_video(vis_folder, out_folder + '.mp4', fps)


def process_frames_animation(l_frame_paths, model):
    l_duration = []
    l_trans = []
    l_poses = []
    l_betas = []

    start_process_frames = time.time()
    for i, frame_path in enumerate(tqdm(l_frame_paths)):
        input_path = os.path.join(args.img_folder, frame_path)

        duration, humans, cam_param, img_pil_nopad = infer_img(input_path, model)
        l_duration.append(duration)

        expand_if_1d = lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) and x.ndim==1 else x
        for j, human in enumerate(humans):
            if j == 0:
                betas = expand_if_1d(human['shape'].cpu().numpy())
                trans = human['transl'].cpu().numpy()
                body_poses = human['rotvec'][:22].cpu().numpy()
                hand_poses = human['rotvec'][22:52].cpu().numpy()
                jaw_pose = expand_if_1d(human['rotvec'][52].cpu().numpy())
                l_betas.append(betas)
                l_poses.append(np.concatenate((body_poses, jaw_pose, jaw_pose, jaw_pose, hand_poses), axis=0))
                l_trans.append(trans)

    print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()}")
    print(f'Total process time={time.time() - start_process_frames}')

    frames_dict = {
        'trans': l_trans,
        'poses': l_poses,
        'betas': l_betas[0]
    }

    return frames_dict

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

    return duration, outputs, K, img_pil_nopad

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
    parser.add_argument("--gender", type=str, default='neutral')
    parser.add_argument("--inference_id", type=str)
    parser.add_argument("--inference_animation", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--unique_color", default=False, action="store_true")
    args = parser.parse_args()

    dict_args = vars(args)

    assert torch.cuda.is_available()

    # check format
    assert os.path.splitext(args.vid)[1] == '.mp4', 'Only mp4 format is supported'
    
    frame_folder, vid_name = decode_video(args)
    print(f'complete to extract {vid_name} / {args.fps} FPS at {frame_folder}')

    # Manage Input/Output 
    suffixes = ('.jpg', '.jpeg', '.png', '.webp')
    list_input_path = [os.path.join(frame_folder, file) for file in os.listdir(frame_folder) if file.endswith(suffixes) and file[0] != '.']
    list_input_path.sort()
    assert len(list_input_path) > 0, 'No frames to infer'
    print(f'The number of images to infer: {len(list_input_path)}')

    if args.inference_id:
        inference_id = args.inference_id
    else:
        current_time = time.localtime()
        inference_id = time.strftime("%Y%m%d%H%M", current_time)
    out_folder = f'{args.out_folder}/{vid_name}_{inference_id}'
    os.makedirs(out_folder, exist_ok=True)

    # Manage Meta
    meta_data = {
        "fps": args.fps
    }
    meta_path = os.path.join(out_folder, 'meta.json')
    with open(meta_path, "w") as meta_file:
        json.dump(meta_data, meta_file)
    
    # Manage inference
    model = prepare_inference()
    print(f'complete to preparing {args.model_name} inference')

    if args.inference_animation:
        frames_dict = process_frames_animation(list_input_path, model)
        frames_dict['mocap_framerate'] = int(args.fps)
        frames_dict['gender'] = args.gender

        save_file_name = os.path.join(out_folder, f"{args.inference_id}_{args.model_name}")
        meta_fn = save_file_name+'.npz'
        np.savez(meta_fn, **frames_dict)
    else:
        process_frames(list_input_path, out_folder, model, args.visualize, args.unique_color, args.fps)

    print(f'complete to process {vid_name} at {out_folder}')