import os
import sys
import zipfile
import requests
from tqdm import tqdm
import kagglehub
from pathlib import Path

# ====================================
# CONFIGURATION
# ====================================
DATASET_NAME = "tristanzhang32/ai-generated-images-vs-real-images"
LOCAL_DATASET_DIR = "data/raw"
KAGGLE_CONFIG_DIR = os.path.expanduser("~/.kaggle")
KAGGLE_CONFIG_FILE = os.path.join(KAGGLE_CONFIG_DIR, "kaggle.json")

# ====================================
# UTILITY FUNCTIONS
# ====================================
def check_kaggle_credentials():
    """Check if Kaggle API credentials are properly configured."""
    if not os.path.exists(KAGGLE_CONFIG_FILE):
        print("Kaggle credentials not found!")
        print("\nTo download datasets, you need to:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place kaggle.json in:", KAGGLE_CONFIG_DIR)
        print("4. Make sure the file has proper permissions (chmod 600)")
        return False
    return True

def create_directories():
    """Create necessary directories for dataset storage."""
    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
    print(f"Created directory: {LOCAL_DATASET_DIR}")

def download_with_progress(url, filepath, chunk_size=8192):
    """Download file with progress bar using requests."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = file.write(chunk)
                    pbar.update(size)
        
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def extract_with_progress(zip_path, extract_to):
    """Extract ZIP file with progress bar."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.infolist()
            
            with tqdm(
                total=len(file_list),
                desc="Extracting files",
                unit="file"
            ) as pbar:
                for file_info in file_list:
                    zip_ref.extract(file_info, extract_to)
                    pbar.update(1)
        
        print(f"Extraction completed to: {extract_to}")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def get_dataset_info():
    """Display information about the dataset."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION DATASET")
    print("="*60)
    print(f"Dataset: {DATASET_NAME}")
    print("Description: AI Generated Images vs Real Images")
    print("Classes: Fake (AI-generated) vs Real images")
    print("Purpose: Training deepfake detection models")
    print("="*60)

def validate_download(dataset_path):
    """Validate that the dataset was downloaded correctly."""
    if not os.path.exists(dataset_path):
        return False
    
    total_files = sum([len(files) for r, d, files in os.walk(dataset_path)])
    
    if total_files == 0:
        print("Warning: Dataset directory is empty")
        return False
    
    print(f"Dataset validation successful!")
    print(f"Location: {dataset_path}")
    print(f"Total files found: {total_files}")
    
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if subdirs:
        print("Folder structure:")
        for subdir in subdirs[:5]:
            subdir_path = os.path.join(dataset_path, subdir)
            file_count = sum([len(files) for r, d, files in os.walk(subdir_path)])
            print(f"   └── {subdir}/ ({file_count} files)")
        
        if len(subdirs) > 5:
            print(f"   └── ... and {len(subdirs) - 5} more folders")
    
    return True

# ====================================
# MAIN DOWNLOAD FUNCTION
# ====================================
def download_dataset():
    """Download the deepfake detection dataset with progress tracking."""
    
    print("Starting dataset download process...")
    
    if not check_kaggle_credentials():
        return False
    
    create_directories()
    
    get_dataset_info()
    
    print("\nInitiating download...")
    print("This may take several minutes depending on your connection...")
    
    try:
        print(f"Downloading {DATASET_NAME}...")
        
        import threading
        import time
        
        download_complete = False
        
        def progress_indicator():
            chars = "|/-\\"
            idx = 0
            while not download_complete:
                print(f"\rDownloading... {chars[idx % len(chars)]}", end="", flush=True)
                idx += 1
                time.sleep(0.1)
        
        progress_thread = threading.Thread(target=progress_indicator)
        progress_thread.daemon = True
        progress_thread.start()
        
        dataset_path = kagglehub.dataset_download(DATASET_NAME)
        download_complete = True
        
        print(f"\rDownload completed!                    ")
        print(f"Dataset downloaded to: {dataset_path}")
        
        if validate_download(dataset_path):
            print("\nDataset ready for preprocessing!")
            print("\nNext steps:")
            print("1. Run the preprocessing script to prepare train/val/test splits")
            print("2. Start training your deepfake detection models")
            print("3. Use the inference script to test on new images")
            return True
        else:
            print("Dataset validation failed")
            return False
            
    except Exception as e:
        download_complete = True
        print(f"\rDownload failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify Kaggle API credentials")
        print("3. Ensure you have accepted the dataset's terms on Kaggle website")
        return False

# ====================================
# MAIN EXECUTION
# ====================================
if __name__ == "__main__":
    print("DEEPFAKE DETECTION DATASET DOWNLOADER")
    print("=" * 50)
    
    success = download_dataset()
    
    if success:
        print("\nAll done! Dataset is ready for use.")
        sys.exit(0)
    else:
        print("\nDownload process failed. Please check the errors above.")
        sys.exit(1)
