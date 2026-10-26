import os
from datetime import datetime

def create_timestamped_directory(base_dir, folder_name):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    full_path = os.path.join(base_dir, folder_name, timestamp)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def create_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)
