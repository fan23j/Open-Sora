import os
import subprocess
import argparse

def create_remote_directory(remote_path):
    try:
        subprocess.run(['ssh', remote_path.split(':')[0], f'mkdir -p {remote_path.split(":")[1]}'], check=True)
        print(f"Ensured remote path {remote_path} exists.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating remote directory {remote_path}: {e}")
        raise

def find_model_subfolders(base_path):
    model_folders = []
    for subfolder in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(subfolder_path):
            model_path = os.path.join(subfolder_path, 'model')
            if os.path.isdir(model_path):
                model_folders.append((subfolder, model_path))
    return model_folders

def rsync_folders(folders, remote_path):
    for parent_folder, model_folder in folders:
        remote_folder_path = os.path.join(remote_path, parent_folder)
        try:
            print(f"Syncing {model_folder} to {remote_folder_path}")
            subprocess.run(['rsync', '-avz', model_folder, remote_folder_path], check=True)
            print(f"Successfully synced {model_folder}")
        except subprocess.CalledProcessError as e:
            print(f"Error syncing {model_folder}: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for wandb configuration.")
    parser.add_argument('--base-path', type=str, default="",
                        help="Path to the ckpt folder on your local machine.")
    parser.add_argument('--remote-path', type=str, default="",
                        help="Path to the ckpt folder on your remote machine.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    create_remote_directory(args.remote_path)

    # Find all the sub-folders in base_path that contains model weights
    subfolders_with_model = find_model_subfolders(args.base_path)

    # rsync on the remote server (skip all the folders & files that are not for model weights)
    rsync_folders(subfolders_with_model, args.remote_path)
