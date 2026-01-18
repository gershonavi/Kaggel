# Kaggle Solutions Repository
Collection of different Kaggle challenges.
python to use : 
$env:KAGGLE_CONFIG_DIR='C:\Users\avig\.kaggle'; C:\Users\avig\anaconda3\python.exe submit_kaggle.py

## AI Agent Instructions
When adding new challenges or solutions, you **MUST** follow these strict naming conventions:

1.  **Prefix Naming**: Never use generic names like `data`, `solutions`, or `utils`. Always prefix them with the project/challenge name.
    *   **Bad**: `data/`, `solution.py`
    *   **Good**: `Titanic_data/`, `Titanic_solution.py`, `Spaceship_Titanic_data/`

2.  **Structure**:
    *   `Kaggel_Solutions/<Challenge_Name>/<Version_Folder>/`
    *   Inside the version folder:
        *   `<Challenge_Name>_data/` (Directory for datasets)
        *   `<Challenge_Name>_solution.py` (Main script)
        *   `<Challenge_Name>_download_data.py` (Download script)

3.  **Execution**:
    *   Scripts must rely on relative paths or the prefixed folder names, not hardcoded "data".

4.  **Global Configuration Isolation**:
    *   Do **NOT** place project-wide configurations (e.g., `kaggle.json`, submission tokens, or generic submission scripts) inside specific solution folders.
    *   These must remain in the project root or use the User's global environment (e.g., `~/.kaggle/`).

Maintain this consistency to prevents conflicts and ambiguity in a multi-project environment.

## Usage & Recovery (User Guide)
In case of power failure or a fresh environment start, follow these steps to resume work or run solutions.
use conda
### 1. Python Environment Setup
To run the advanced solutions (Stacking, Boosting), you need the following packages installed.

**Recommended Installation Command:**
```bash
pip install pandas numpy scikit-learn kaggle xgboost lightgbm catboost
```

**Key Libraries:**
*   **Core**: `pandas`, `numpy` (Data Manipluation)
*   **Models**: `scikit-learn` (RandomForest, Linear), `xgboost`, `lightgbm`, `catboost` (Gradient Boosting SOTA)
*   **API**: `kaggle` (Data download/submission)

### 2. Kaggle API Setup (Required for Submission)
To automate downloading data and submitting predictions, we use a local `kaggle.json`.
### Kaggle Submission Commands

**Titanic:**
```bash
kaggle competitions submit -c titanic -f submission.csv -m "Titanic Submission"
```

**House Prices:**
```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "House Prices Submission"
```

**Spaceship Titanic:**
```bash
kaggle competitions submit -c spaceship-titanic -f submission.csv -m "Spaceship Titanic Submission"
```

3.  **Security**:
    *   **No Hardcoded Keys**: The `kaggle.json` file is ignored by git to prevent accidental exposure of credentials.
    *   **Setup**:
        1.  Rename `kaggle.json.example` to `kaggle.json`.
        2.  Add your Kaggle username and key to the file.
        3.  Alternatively, place your `kaggle.json` in `~/.kaggle/` (default system location).
    *   **File Content** (`kaggle.json.example`):
    ```json
    {
      "username": "<KEY>",
      "key": "<KEY>"
    }
    ```
    *   The `.gitignore` in this project is configured to exclude `kaggle.json`.
### 4. Automated Submission Workflow (Reliable Method)
Since `kaggle.exe` path issues are common on Windows, a dedicated Python script is included to handle submission using the local credentials.

**Option A: Universal Submission Script (Recommended)**
1.  Open `manual_submit.py` and ensure the `SUBMISSION_FILE` path points to your desired CSV.
2.  Run:
    ```bash
    python manual_submit.py
    ```

**Option B: CLI Direct Command**
If you prefer the command line, use `python -m kaggle` to avoid path issues:
```bash
# Set Config Path (PowerShell)
$env:KAGGLE_CONFIG_DIR = "c:\Users\avig\Documents\GitHub\Kaggel_chalenge_1"

# Submit
python -m kaggle competitions submit -c <competition-name> -f <path-to-csv> -m "Your Message"
```

### 4. Project Structure Overview
*   **`Kaggel_Solutions/`**: Root for all challenges.
    *   **`<Challenge>/`**: (e.g., `Titanic`, `Spaceship_Titanic`)
        *   **`<Version>/`**: (e.g., `v1_baseline`, `03_Stacking_SOTA`)
            *   `*_solution.py`: The logic.
            *   `*_download_data.py`: Data fetcher.

### 5. Backward Compatibility & Isolation
*   Each version folder (e.g., `v1_baseline`, `v2_advanced`) is **self-contained**.
*   They may share the `*_data` folder definition, but they contain their own logic and download scripts if necessary.
*   **Do not overwrite** older versions. Always create a new version folder (e.g., `v3_experimental`) for new attempts.


