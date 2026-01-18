import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import skew

# Paths
TRAIN_PATH = 'House_Prices_data/train.csv'
TEST_PATH = 'House_Prices_data/test.csv'
SUBMISSION_PATH = 'submission_v5_community.csv'

def load_data():
    import os
    if not os.path.exists(TRAIN_PATH):
        import House_Prices_download_data
        House_Prices_download_data.download_data()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class CommunityFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_temp = X.copy()
        self.numeric_features = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identify Skewed Features
        # Using 0.75 threshold common in top kernels
        self.skewed_feats = []
        for c in self.numeric_features:
            # Impute temp for skew calc
            vals = X_temp[c].fillna(X_temp[c].median())
            if abs(skew(vals)) > 0.75:
                self.skewed_feats.append(c)
        
        # Manifold (V2/V4 Logic kept because it works)
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        
        self.kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        self.pca = PCA(n_components=3, random_state=42)
        
        # Handle negatives/zeros for log before clustering
        data_for_cluster = X_temp[cluster_cols].fillna(0)
        data_for_cluster = np.log1p(data_for_cluster)
        
        self.kmeans.fit(data_for_cluster)
        self.pca.fit(X_temp[self.numeric_features].fillna(0))
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Strict Ordinal Encoding (The "Community Insight")
        # Map qualities to numbers so linear models see the progression
        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None': 0}
        ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        
        for c in ordinal_cols:
            if c in X.columns:
                X[c] = X[c].fillna('None').map(qual_map).fillna(0) 
        
        # 2. Golden Features (Interaction Terms)
        # TotalSF is already a known good one
        X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
        
        # Interaction: Quality * Area (Very strong predictor)
        if 'OverallQual' in X.columns and 'TotalSF' in X.columns:
            X['Qual_TotalSF'] = X['OverallQual'] * X['TotalSF']
        
        if 'OverallQual' in X.columns and 'GrLivArea' in X.columns:
            X['Qual_GrLivArea'] = X['OverallQual'] * X['GrLivArea']
            
        # 3. Global Deskewing
        for c in self.skewed_feats:
            if c in X.columns:
                # Log1p is safe and effective
                X[c] = np.log1p(X[c].fillna(X[c].median()))
                
        # 4. Manifold
        X_nums = X[self.numeric_features].fillna(0)
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        data_for_cluster = np.log1p(X_nums[cluster_cols])
        
        X['ClusterID'] = self.kmeans.predict(data_for_cluster)
        pca_feats = self.pca.transform(X_nums)
        X['PCA1'] = pca_feats[:, 0]
        X['PCA2'] = pca_feats[:, 1]
        
        return X

def get_pipeline(model):
    from sklearn.compose import make_column_selector
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', cat_transformer, make_column_selector(dtype_include=object))
        ])
    
    return Pipeline(steps=[
        ('fe', CommunityFeatureEngineer()),
        ('pre', preprocessor),
        ('model', model)
    ])

def main():
    print("Loading Data...")
    train, test = load_data()
    test_ids = test['Id']
    
    # Cleaning (Standard V2/V4)
    train = train[train['GrLivArea'] < 4500]
    
    y = np.log1p(train['SalePrice'])
    X = train.drop(['SalePrice', 'Id'], axis=1)
    X_test = test.drop('Id', axis=1)
    
    # --- V5 Ensemble (V4 Stack + Encoded Features) ---
    
    # Tuned interactions will be picked up better by Linear models now
    lasso = Lasso(alpha=0.0005, random_state=1)
    ridge = Ridge(alpha=10)
    
    gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, 
                                   max_features='sqrt', min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=42)
    
    hgb = HistGradientBoostingRegressor(loss='squared_error', max_iter=500, random_state=42)
    
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4, 
                       subsample=0.7, colsample_bytree=0.7, random_state=42)
    
    lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, verbose=-1, random_state=42)

    estimators = [
        ('gb', get_pipeline(gb)),
        ('lasso', get_pipeline(lasso)),
        ('ridge', get_pipeline(ridge)),
        ('hgb', get_pipeline(hgb)),
        ('xgb', get_pipeline(xgb)),
        ('lgbm', get_pipeline(lgbm))
    ]
    
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    print("Training V5 Community Insight Ensemble...")
    stack.fit(X, y)
    
    print("Predicting...")
    preds_log = stack.predict(X_test)
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
