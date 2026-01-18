import os
import requests

DATA_DIR = 'Titanic_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Mirrors for Titanic dataset
TRAIN_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
TEST_URL = 'https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/test.csv'

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(os.path.join(DATA_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def download_data():
    download_file(TRAIN_URL, 'train.csv')
    download_file(TEST_URL, 'test.csv')
    if os.path.exists(os.path.join(DATA_DIR, 'train.csv')):
        print("Data download complete.")

if __name__ == "__main__":
    download_data()
