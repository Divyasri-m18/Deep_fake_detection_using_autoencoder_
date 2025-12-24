import kagglehub
import shutil
import os
import yaml

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_and_setup():
    print("Downloading 140k-real-and-fake-faces dataset using kagglehub...")
    print("This might take a while depending on your internet connection.")
    
    # Download latest version
    try:
        path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        print("Dataset downloaded to cache at:", path)
        
        config = load_config()
        target_dir = config['paths']['kaggle_data_path'] # data/kaggle_140k
        
        # Determine strict absolute path for safety
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Handle relative path in config
        if not os.path.isabs(target_dir):
            target_dir = os.path.join(project_root, target_dir)
            
        print(f"Moving/Copying dataset to project folder: {target_dir}")
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        # Move logic: The cached path is usually a directory containing the files
        # We want to copy the contents of 'path' into 'target_dir'
        # Using shutil.copytree is tricky if destination exists.
        # Let's iterate and move/copy.
        
        # Simple copy fallback
        import distutils.dir_util
        distutils.dir_util.copy_tree(path, target_dir)
        
        print("Dataset setup complete!")
        
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please ensure you have authenticated with Kaggle if required, or download manually.")

if __name__ == "__main__":
    download_and_setup()
