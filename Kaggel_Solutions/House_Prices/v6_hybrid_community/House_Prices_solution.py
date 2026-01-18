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

# Paths
TRAIN_PATH = 'House_Prices_data/train.csv'
TEST_PATH = 'House_Prices_data/test.csv'
SUBMISSION_PATH = 'submission_v6_hybrid_community.csv'

def load_data():
    import os
    if not os.path.exists(TRAIN_PATH):
        import House_Prices_download_data
        House_Prices_download_data.download_data()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class HybridCommunityFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_temp = X.copy()
        
        # 1. Manifold Fitting (V4 SOTA Logic)
        self.numeric_features = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        for c in self.numeric_features:
            X_temp[c] = X_temp[c].fillna(X_temp[c].median())
            
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        
        self.kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        self.pca = PCA(n_components=3, random_state=42)
        
        data_for_cluster = np.log1p(X_temp[cluster_cols].fillna(0))
        self.kmeans.fit(data_for_cluster)
        self.pca.fit(X_temp[self.numeric_features].fillna(0))
        
        # 2. Ordinal Mapping Setup (V5 Insight)
        self.qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None': 0}
        
        # 3. Winsorization Learning (V4 SOTA Logic)
        # We need to compute features first to learn caps for new features too!
        # This is strictly hard to do in one-pass sklearn fit without leakage, 
        # so we'll learn caps on the RAW numerics now, and hard-code specific interaction caps if needed 
        # or just clip interactions dynamically.
        # Let's stick to V4 strategy: Cap RAW columns.
        self.caps = {}
        for col in self.numeric_features:
            self.caps[col] = X_temp[col].quantile(0.99)
            
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Custom Cleaning (V2/V4)
        none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                     'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                     'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
        for c in none_cols:
            if c in X.columns: X[c] = X[c].fillna('None')

        # 2. Ordinal Encoding (V5 Insight - The "Community" Addition)
        ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'OverallQual', 'OverallCond']
        # Note: OverallQual is numeric, but we can treat it as is.
        # The others are strings.
        for c in ordinal_cols:
            if c in X.columns and X[c].dtype == object:
                X[c] = X[c].fillna('None').map(self.qual_map).fillna(0)

        # 3. Winsorization (V4 SOTA Restoration)
        # Clip RAW numerics
        for c in self.numeric_features:
             if c in X.columns and c in self.caps:
                 X[c] = X[c].clip(upper=self.caps[c])
        
        # 4. Feature Creation (V2 + V5 Golden)
        X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
        
        # V5 Golden Features
        if 'OverallQual' in X.columns and 'TotalSF' in X.columns:
            X['Qual_TotalSF'] = X['OverallQual'] * X['TotalSF']
        if 'OverallQual' in X.columns and 'GrLivArea' in X.columns:
            X['Qual_GrLivArea'] = X['OverallQual'] * X['GrLivArea']
        if 'OverallQual' in X.columns and 'YearBuilt' in X.columns:
             X['Qual_YearBuilt'] = X['OverallQual'] * X['YearBuilt']
        
        # Clip the NEW Golden Features (Critical Step to fix V5 regression)
        # Quantile 0.99 logic dynamically applied
        for c in ['Qual_TotalSF', 'Qual_GrLivArea', 'Qual_YearBuilt']:
            if c in X.columns:
                cap = X[c].quantile(0.99) # This might leak test data slightly if transform is called on test batch, 
                                        # but in this script we transform whole train/test usually or safely ignore for stability.
                                        # Ideally we should learn this in fit, but for now we clamp roughly.
                                        # Let's hard clamp to reasonable bounds instead to avoid leakage.
                                        # Qual (5) * SF (4000) = 20000. 
                X[c] = X[c].clip(upper=35000) # Safety Clamp

        X['YearBuilt_Age'] = 2010 - X['YearBuilt']
        
        # 5. Manifold (V4 SOTA)
        X_nums = X[self.numeric_features].fillna(0)
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        data_for_cluster = np.log1p(X_nums[cluster_cols])
        
        X['ClusterID'] = self.kmeans.predict(data_for_cluster)
        pca_feats = self.pca.transform(X_nums)
        X['PCA1'] = pca_feats[:, 0]
        X['PCA2'] = pca_feats[:, 1]
        X['PCA3'] = pca_feats[:, 2]
        
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
        ('fe', HybridCommunityFeatureEngineer()),
        ('pre', preprocessor),
        ('model', model)
    ])

def main():
    print("Loading Data...")
    train, test = load_data()
    test_ids = test['Id']
    
    # Aggressive Outlier Removal (V4 SOTA)
    train = train[train['GrLivArea'] < 4500]
    
    y = np.log1p(train['SalePrice'])
    X = train.drop(['SalePrice', 'Id'], axis=1)
    X_test = test.drop('Id', axis=1)
    
    # --- V6 Stack (V4 Hybrid + New Features) ---
    
    # 1. Gradient Boost
    gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, 
                                   max_features='sqrt', min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=42)
    # 2. Linear - Will LOVE the new Interaction features
    lasso = Lasso(alpha=0.0005, random_state=1)
    ridge = Ridge(alpha=10)
    
    # 3. HistGradient
    hgb = HistGradientBoostingRegressor(loss='squared_error', max_iter=500, random_state=42)
    
    # 4. XGBoost
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4, 
                       subsample=0.7, colsample_bytree=0.7, random_state=42)
    
    # 5. LightGBM
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
    
    print("Training V6 Hybrid Community Stack (V4 + Ordinal/Interaction w/ Clipping)...")
    stack.fit(X, y)
    
    print("Predicting...")
    preds_log = stack.predict(X_test)
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
