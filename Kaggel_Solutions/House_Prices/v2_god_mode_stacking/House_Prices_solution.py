import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Paths
TRAIN_PATH = 'House_Prices_data/train.csv'
TEST_PATH = 'House_Prices_data/test.csv'
SUBMISSION_PATH = 'submission_v2_god_mode.csv'

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_temp = X.copy()
        
        # Determine filling for numeric clustering
        self.numeric_features = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        # Clean for fit
        for c in self.numeric_features:
            X_temp[c] = X_temp[c].fillna(X_temp[c].median())
            
        # Manifold Learning (Unsupervised)
        # Using physical properties for clustering
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        
        self.kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        self.pca = PCA(n_components=3, random_state=42)
        
        # Scale for clustering
        data_for_cluster = np.log1p(X_temp[cluster_cols])
        self.kmeans.fit(data_for_cluster)
        self.pca.fit(X_temp[self.numeric_features])
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Custom Cleaning (Domain Knowledge)
        # Some NAs mean "None"
        none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                     'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                     'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
        for c in none_cols:
            if c in X.columns:
                X[c] = X[c].fillna('None')
                
        # 2. Feature Creation
        X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
        X['YearBuilt_Age'] = 2010 - X['YearBuilt']
        X['YearRemod_Age'] = 2010 - X['YearRemodAdd']
        X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath'] + X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
        
        # 3. Manifold Features
        # Impute temporarily for calculation
        X_nums = X[self.numeric_features].fillna(X[self.numeric_features].median())
        
        cluster_cols = [c for c in self.numeric_features if c in ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']]
        if not cluster_cols: cluster_cols = self.numeric_features[:5]
        
        # Safe log for clustering
        data_for_cluster = np.log1p(X_nums[cluster_cols])
        
        X['ClusterID'] = self.kmeans.predict(data_for_cluster)
        
        pca_feats = self.pca.transform(X_nums)
        X['PCA1'] = pca_feats[:, 0]
        X['PCA2'] = pca_feats[:, 1]
        X['PCA3'] = pca_feats[:, 2]
        
        return X

def get_pipeline(model):
    # Numeric: Quantile Transform for non-normal distributions
    # Categorical: OneHot (High cardinality handled by Stacking usually, or we can use Ordinal for some)
    
    # We will identify columns dynamically for simplicity in pipeline
    # But for "Best Practice", explicit types are better. 
    # For speed, we use type selectors.
    
    from sklearn.compose import make_column_selector
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        # Strategy constant='None' for things we missed? Or most_frequent
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
    
    # Outlier Removal (Manual "God Mode" cleaning)
    # GrLivArea > 4000 is recommended to remove in documentation
    train = train[train['GrLivArea'] < 4500]
    
    # Target Log Transform
    y = np.log1p(train['SalePrice'])
    X = train.drop(['SalePrice', 'Id'], axis=1)
    X_test = test.drop('Id', axis=1)
    
    # --- Ensembling Strategy ---
    
    # 1. Gradient Boosting (Robust)
    gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, 
                                   max_features='sqrt', min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=42)
    
    # 2. Linear Models (Very strong for this dataset with OneHot)
    lasso = Lasso(alpha=0.0005, random_state=1)
    ridge = Ridge(alpha=10)
    
    # 3. HistGradient (LightGBM equivalent)
    hgb = HistGradientBoostingRegressor(loss='squared_error', max_iter=500, random_state=42)
    
    # Stacking
    estimators = [
        ('gb', get_pipeline(gb)),
        ('lasso', get_pipeline(lasso)),
        ('ridge', get_pipeline(ridge)),
        ('hgb', get_pipeline(hgb))
    ]
    
    # Meta Learner
    # Ridge is usually a safe meta-learner for regression stacking
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    print("Training God Mode Stacked Enesemble...")
    stack.fit(X, y)
    
    print("Predicting...")
    preds_log = stack.predict(X_test)
    preds = np.expm1(preds_log) # Invert Log
    
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
