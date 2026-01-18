import os
import requests
import zipfile
import io

# We will try to download from a mirror or just placeholder for manual if mirrors fail.
# Spaceship Titanic is newer, might not be on the same simple mirrors.
# Let's try to verify if we can find a raw CSV mirror.
# Often these are on GitHub.
# User: https://github.com/vbookshelf/spaceship-titanic-dataset/raw/master/spaceship_titanic.zip (Example, need to verify or try generic)

# Actually, rely on manual download if these fail, but let's try a known one or create empty placeholders with instructions.
# Found working mirrors via search
DATA_URL = 'https://raw.githubusercontent.com/You-sha/Spaceship-Titanic/master/train.csv'
TEST_URL_GIT = 'https://raw.githubusercontent.com/You-sha/Spaceship-Titanic/master/test.csv'

DATA_DIR = 'Spaceship_Titanic_data'
os.makedirs(DATA_DIR, exist_ok=True)

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

if __name__ == "__main__":
    download_file(DATA_URL, 'train.csv')
    download_file(TEST_URL_GIT, 'test.csv')
