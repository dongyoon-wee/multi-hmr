import os
import cv2
import json
import shutil
import csv

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from pathlib import Path

def create_directory(directory):
    # Create the directory if it doesn't exist
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")

def filter_dataset(json_dir):
    list_files = get_point_json_files(json_dir)
    if len(list_files) > 0:
        print(list_files)

def get_point_json_files(json_directory):
    files_with_points = []
    # Walk through all directories and files under the root directory
    for root, dirs, files in os.walk(json_directory):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                # Open and load the JSON file
                with open(file_path, 'r') as file:
                    try:
                        data = json.load(file)
                        # Check if "points" key is present
                        for d in data["annotations"]:
                            if "points" in d or "point" in d:
                                files_with_points.append(file_path)
                                break
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
                    except UnicodeError:
                        print(f"Error Unicode in file: {file_path}")
    return files_with_points

def check_points(json_file):
    with open(json_file, 'r') as rf:
        try:
            data = json.load(rf)
            for d in data['annotations']:
                if "points" in d or "point" in d:
                    return True
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {json_file}")
        except UnicodeError:
            print(f"Error Unicode in file: {json_file}")
    return False

def update_json(file_path, k, v):
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            data = json.load(rf)
    else:
        data = {}
    data[k] = v
    with open(file_path, 'w') as wf:
        json.dump(wf, file_path, indent=4)

def make_shot_label(label_dir_old, label_dir_new):
    for root, dirs, files in os.walk(label_dir_old):
        for filename in files:
            if filename.endswith('.json'):
                filename_without_ext = os.path.splitext(filename)[0]
                parts = filename_without_ext.split('_')
                shot_idx = parts[-2]
                seq_idx = parts[-1]
                with open(os.path.join(label_dir_old, filename), 'r') as rf_frame:
                    data = json.load(rf_frame)
                    for d in data["annotations"]:
                        if "points" in d or "point" in d:
                            label_file_path = os.path.join(label_dir_new, f'{shot_idx}.json')
                            update_json(label_file_path, seq_idx, d)    


def visualize_keypoint(keypoints, image, radius=5, color=(0,0,255), thickness=-1):
    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i+1]
        cv2.circle(image, (x, y), radius, color, thickness)
    return image

def check_dataset(label_dir, image_dir):
    for root, dirs, files in os.walk(label_dir):
        for directory in dirs:
            label_shot_dir = os.path.join(root, directory)
            image_shot_dir = os.path.join(image_dir, directory)
            list_filenames = os.listdir(label_shot_dir)
            for filename in list_filenames:
                label_path = os.path.join(label_shot_dir, filename)
                with open(label_path, 'r') as rf_label:
                    data = json.load(rf_label)
                for i, d in data.items():
                    print(f'{label_path}')

def copy_images(label_path, image_dir, image_dir_new, visualize_keypoint=False):
    # parse the label path to get image file path
    scene_name = os.path.basename(os.path.dirname(label_path))
    file_name = os.path.basename(label_path)
    shot_name, _ = os.path.splitext(file_name)

    with open(label_path, 'r') as rf_label:
        data = json.load(rf_label)        

    for i, d in data.items():
        image_path_new = os.path.join(image_dir_new, f'{scene_name}_{shot_name}_{i}.jpg')

        try:
            keypoints = d['points']
            image_path = os.path.join(image_dir, f'{scene_name}_{shot_name}_{i}.jpg')
            create_directory(image_dir_new)
            if visualize_keypoint:
                image = cv2.imread(image_path)            
                image_kpt = visualize_keypoint(keypoints, image)
                cv2.imwrite(image_path_new, image_kpt)
            else:
                shutil.copy2(image_path, image_path_new)
            print(f'save the image to {image_path_new}')
        except FileNotFoundError as e:
            print(e)

def count_files_in_folders_by_extension(root_folder, extension):
    folder_file_count = {}
    
    for foldername, subfolders, filenames in os.walk(root_folder):
        filtered_files = [file for file in filenames if file.lower().endswith(extension.lower())]
        file_count = len(filtered_files)
        folder_file_count[foldername] = file_count
    
    for foldername in sorted(folder_file_count.keys(), key=lambda x: x.count(os.sep), reverse=True):
        parent_folder = os.path.dirname(foldername)
        if parent_folder and parent_folder.startswith(root_folder):
            folder_file_count[parent_folder] = folder_file_count.get(parent_folder, 0) + folder_file_count[foldername]
    
    return folder_file_count

def save_to_csv(file_counts, output_file):
    # Save the file counts to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Folder', 'File Count'])
        for folder, count in file_counts.items():
            writer.writerow([folder, count])

def manage_images(scene, label_dir, image_dir, image_dir_new):
    new_image_scene_dir = os.path.join(image_dir_new, scene)
    new_meta_scene_dir = os.path.join(meta_dir_new, scene)
    create_directory(new_image_scene_dir)
    create_directory(new_meta_scene_dir)
    for root, dirs, files in os.walk(label_dir):
        for directory in dirs:
            label_shot_dir = os.path.join(root, directory)
            new_image_shot_dir = os.path.join(new_image_scene_dir, directory)
            list_filenames = os.listdir(label_shot_dir)
            image_shot_dir = os.path.join(image_dir, directory)
            for filename in list_filenames:
                label_path = os.path.join(label_shot_dir, filename)
                print(f'{label_path}')
                copy_images(label_path, image_shot_dir, new_image_shot_dir)

def manage_labels(scene, label_dir, ):
    new_label_scene_dir = os.path.join(label_dir_new, scene)
    new_image_scene_dir = os.path.join(image_dir_new, scene)
    create_directory(new_label_scene_dir)
    create_directory(new_image_scene_dir)

    for root, dirs, files in os.walk(label_dir):
        for directory in dirs:
            new_label_shot_dir = os.path.join(new_label_scene_dir, directory)
            new_image_shot_dir = os.path.join(new_image_scene_dir, directory)
            image_shot_dir = os.path.join(image_dir, directory)
            create_directory(new_label_shot_dir)
            create_directory(new_image_shot_dir)

            label_shot_dir = os.path.join(label_dir, directory)
            for root, dirs, files in os.walk(label_shot_dir):
                for filename in files:
                    if filename.endswith('.json'):
                        filename_without_ext = os.path.splitext(filename)[0]
                        parts = filename_without_ext.split('_')
                        shot_idx = parts[-2]
                        seq_idx = parts[-1]
                        label_path = os.path.join(label_shot_dir, filename)
                        print(f'{label_path}')
                        with open(os.path.join(label_shot_dir, filename), 'r') as rf:
                            data = json.load(rf)
                            for d in data["annotations"]:
                                if "points" in d or "point" in d:
                                    shot_json_path = os.path.join(new_label_shot_dir, f'{shot_idx}.json')

                                    if os.path.exists(shot_json_path):
                                        with open(shot_json_path, 'r') as shot_file:
                                            shot_data = json.load(shot_file)
                                    else:
                                        shot_data = {}
                                    
                                    shot_data[seq_idx] = d

                                    with open(shot_json_path, 'w') as shot_file:
                                        json.dump(shot_data, shot_file, indent=4)
                                    print(f'updated or created {shot_json_path}.')

def check_dataset():
    extension = ".jpg"  # Set the desired file extension
    file_counts = count_files_in_folders_by_extension(new_image_scene_dir, ".jpg")
    output_csv = os.path.join(new_meta_scene_dir, 'image_count.csv')

    for folder, count in file_counts.items():
        print(f"Folder: {folder} contains {count} '{extension}' files (including files from child folders)")

    #save_to_csv(file_counts, output_csv)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--train_data', type=str, default='BEDLAM')
    parser.add_argument('--train_split', type=str, default='training')
    parser.add_argument('--root_dir', type=str, default='/mnt/c/Users/USER/Downloads/스포츠 사람 동작 영상(골프)')
    parser.add_argument('--image_dir_new', type=str, default='/home/dywee/data/aihub_golf/train/image')
    parser.add_argument('--label_dir_new', type=str, default='/home/dywee/data/aihub_golf/train/label')
    parser.add_argument('--meta_dir_new', type=str, default='/home/dywee/data/aihub_golf/meta')
    args = parser.parse_args()

    root_dir = args.root_dir
    label_dir_new = args.label_dir_new
    image_dir_new = args.image_dir_new
    meta_dir_new = args.meta_dir_new

    data_dir = os.path.join(root_dir, 'Training', 'Association', 'female')
    list_scene = ['swing_18']

    manage_labels = True
    manage_images = True

    for scene in list_scene:
        image_dir = os.path.join(data_dir, f'[원천]{scene}')
        if manage_labels:
            label_dir = os.path.join(data_dir, f'[라벨]{scene}')
            manage_images(scene, label_dir, image_dir)

        if manage_images:
            label_dir = os.path.join(label_dir_new, f'{scene}')
            manage_images(scene, label_dir, image_dir)

    """
    for scene in list_scene:
        label_dir = os.path.join(data_dir, f'[라벨]{scene}')
        image_dir = os.path.join(data_dir, f'[원천]{scene}')

        new_label_scene_dir = os.path.join(label_dir_new, scene)
        new_image_scene_dir = os.path.join(image_dir_new, scene)
        create_directory(new_label_scene_dir)
        create_directory(new_image_scene_dir)

        for root, dirs, files in os.walk(label_dir):
            for directory in dirs:
                new_label_shot_dir = os.path.join(new_label_scene_dir, directory)
                new_image_shot_dir = os.path.join(new_image_scene_dir, directory)
                image_shot_dir = os.path.join(image_dir, directory)
                create_directory(new_label_shot_dir)
                create_directory(new_image_shot_dir)

                label_shot_dir = os.path.join(label_dir, directory)
                for root, dirs, files in os.walk(label_shot_dir):
                    for filename in files:
                        if filename.endswith('.json'):
                            filename_without_ext = os.path.splitext(filename)[0]
                            parts = filename_without_ext.split('_')
                            shot_idx = parts[-2]
                            seq_idx = parts[-1]
                            label_path = os.path.join(label_shot_dir, filename)
                            print(f'{label_path}')
                            visualize_label(label_path, image_shot_dir, new_image_shot_dir)
                            with open(os.path.join(label_shot_dir, filename), 'r') as rf:
                                data = json.load(rf)
                                for d in data["annotations"]:
                                    if "points" in d or "point" in d:
                                        shot_json_path = os.path.join(new_label_shot_dir, f'{shot_idx}.json')

                                        if os.path.exists(shot_json_path):
                                            with open(shot_json_path, 'r') as shot_file:
                                                shot_data = json.load(shot_file)
                                        else:
                                            shot_data = {}
                                        
                                        shot_data[seq_idx] = d

                                        with open(shot_json_path, 'w') as shot_file:
                                            json.dump(shot_data, shot_file, indent=4)
                                        print(f'updated or created {shot_json_path}.')
                            """
