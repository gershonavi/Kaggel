import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Config
COMPETITION = 'house-prices-advanced-regression-techniques'
DATA_DIR = 'House_Prices_data'

def download_data():
    # Setup Kaggle API
    # Assumes kaggle.json is in project root or configured via env var
    # We will set the env var just in case the script is run standalone without the wrapper logic
    # But ideally this script assumes the environment is set up.
    # For robustness, we'll look for kaggle.json in the git root if not found.
    
    # Simple direct download
    api = KaggleApi()
    api.authenticate()
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Downloading {COMPETITION} files...")
    api.competition_download_files(COMPETITION, path=DATA_DIR)
    
    # Extract
    for item in os.listdir(DATA_DIR):
        if item.endswith('.zip'):
            print(f"Extracting {item}...")
            zip_ref = zipfile.ZipFile(os.path.join(DATA_DIR, item), 'r')
            zip_ref.extractall(DATA_DIR)
            zip_ref.close()
            os.remove(os.path.join(DATA_DIR, item))

    print("Download complete.")

if __name__ == "__main__":
    download_data()
