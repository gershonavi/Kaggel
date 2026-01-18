import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge

# Paths
TRAIN_PATH = 'House_Prices_data/train.csv'
TEST_PATH = 'House_Prices_data/test.csv'
SUBMISSION_PATH = 'submission_v3_optuna.csv'

def load_data():
    import os
    if not os.path.exists(TRAIN_PATH):
        print("Data not found. Running download script...")
        import House_Prices_download_data
        House_Prices_download_data.download_data()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_temp = X.copy()
        self.numeric_features = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        for c in self.numeric_features:
            X_temp[c] = X_temp[c].fillna(X_temp[c].median())
        
        # Clustering
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        
        self.kmeans = KMeans(n_clusters=8, random_state=42, n_init=10) # 8 optimized for housing types
        self.pca = PCA(n_components=5, random_state=42) # More components
        
        data_for_cluster = np.log1p(X_temp[cluster_cols])
        self.kmeans.fit(data_for_cluster)
        self.pca.fit(X_temp[self.numeric_features])
        return self

    def transform(self, X):
        X = X.copy()
        
        # Custom Cleaning
        none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageType', 'PoolQC']
        for c in none_cols:
            if c in X.columns: X[c] = X[c].fillna('None')
                
        # Feature Creation
        X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
        X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
        X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        X['Has2ndFloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        
        # Manifold
        X_nums = X[self.numeric_features].fillna(X[self.numeric_features].median())
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        data_for_cluster = np.log1p(X_nums[cluster_cols])
        
        X['ClusterID'] = self.kmeans.predict(data_for_cluster)
        pca_feats = self.pca.transform(X_nums)
        for i in range(5):
            X[f'PCA{i+1}'] = pca_feats[:, i]
        
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
        ('fe', AdvancedFeatureEngineer()),
        ('pre', preprocessor),
        ('model', model)
    ])

def main():
    print("Loading Data...")
    train, test = load_data()
    test_ids = test['Id']
    
    # Aggressive Outlier Removal
    train = train[train['GrLivArea'] < 4500]
    train = train[train['LotArea'] < 100000] # Remove giant lots which skew
    
    y = np.log1p(train['SalePrice'])
    X = train.drop(['SalePrice', 'Id'], axis=1)
    X_test = test.drop('Id', axis=1)
    
    # --- Tuned Models (Hypothetical Optuna Best Params) ---
    
    # XGBoost: Known strong params for Housing
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4, 
                       subsample=0.7, colsample_bytree=0.7, random_state=42)
    
    # LightGBM
    lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, 
                         bagging_fraction=0.8, feature_fraction=0.8, random_state=42, verbose=-1)
    
    # Kernel Ridge (Captures non-linear polynomial interactions well)
    krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    
    # ElasticNet/Lasso for robust linear baseline
    lasso = LassoCV(max_iter=10000, random_state=42)
    
    estimators = [
        ('xgb', get_pipeline(xgb)),
        ('lgbm', get_pipeline(lgbm)),
        ('krr', get_pipeline(krr)),
        ('lasso', get_pipeline(lasso))
    ]
    
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(), # Meta learner
        cv=5,
        n_jobs=-1
    )
    
    print("Training V3 Optuna-Style Ensemble...")
    stack.fit(X, y)
    
    print("Predicting...")
    preds_log = stack.predict(X_test)
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
