import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
# Update these paths for your specific submission
SUBMISSION_FILE = r"Kaggel_Solutions\Titanic\04_Ashish_Patel_Inspired\submission_ashish_patel_inspired.csv"
MESSAGE = "Submission 04 - Ashish Patel Inspired"
COMPETITION = "titanic"

def submit():
    # Set config dir to current directory so it finds kaggle.json here
    cwd = os.getcwd()
    os.environ['KAGGLE_CONFIG_DIR'] = cwd
    
    print(f"Using kaggle.json from: {cwd}")
    print(f"Submitting file: {SUBMISSION_FILE}")
    
    # Authenticate
    api = KaggleApi()
    try:
        api.authenticate()
        print(f"Authenticated as: {api.get_config_value('username')}")
    except Exception as e:
        print(f"Authentication Failed: {e}")
        print("Ensure 'kaggle.json' is in this folder and has correct credentials.")
        return

    # Submit
    try:
        api.competition_submit(SUBMISSION_FILE, MESSAGE, COMPETITION)
        print("Submission command sent successfully!")
    except Exception as e:
        print(f"Submission Failed: {e}")

if __name__ == "__main__":
    submit()
