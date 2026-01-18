import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Paths
TRAIN_PATH = 'House_Prices_data/train.csv'
TEST_PATH = 'House_Prices_data/test.csv'
SUBMISSION_PATH = 'submission.csv'

def load_data():
    if not os.path.exists(TRAIN_PATH):
        print("Data not found. Running download script...")
        import House_Prices_download_data
        House_Prices_download_data.download_data()
        
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def preprocess(df, train_set=None):
    # Basic Preprocessing for Baseline
    # 1. Drop usually high-missing columns or ID
    df = df.copy()
    
    # 2. Fill Missing (Simple)
    # Numerical -> Median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
        
    # Categorical -> Mode or "Missing"
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")
        # Label Encode
        le = LabelEncoder()
        # Fit on train set column if provided to handle unseen labels? or just simple concatenation
        # For baseline, simple fit_transform on whole column (ignoring leakage roughly) or just concat
        # Let's do simple categorical codes
        df[c] = df[c].astype('category').cat.codes
        
    return df

def main():
    print("Loading Data...")
    import os
    if not os.path.exists('House_Prices_data'):
         import House_Prices_download_data
         House_Prices_download_data.download_data()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    test_ids = test['Id']
    
    # Combine for simple encoding consistency (Baseline hack)
    n_train = len(train)
    y = np.log1p(train['SalePrice']) # Log transform target is standard for this metric (RMSE of logs)
    train = train.drop('SalePrice', axis=1)
    
    all_data = pd.concat([train, test], axis=0)
    all_data = preprocess(all_data)
    
    X = all_data.iloc[:n_train]
    X_test = all_data.iloc[n_train:]
    
    # Drop IDs
    X = X.drop('Id', axis=1)
    X_test = X_test.drop('Id', axis=1)
    
    # Model
    print("Training Baseline Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
    print(f"Baseline CV Score (RMSE): {-scores.mean():.4f}")
    
    # Fit & Predict
    model.fit(X, y)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log) # Revert log
    
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
