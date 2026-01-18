import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_v10_outlier.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

# --- V9 Manifold + Social Logic ---
class ManifoldFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_temp = X.copy()
        X_temp['Cabin'] = X_temp['Cabin'].fillna('T/0/P')
        self.cabin_counts = X_temp['Cabin'].value_counts().to_dict()
        X_temp['Name'] = X_temp['Name'].fillna('Unknown Unknown')
        X_temp['Surname'] = X_temp['Name'].apply(lambda x: x.split(' ')[-1])
        self.surname_counts = X_temp['Surname'].value_counts().to_dict()
        X_temp['Deck'] = X_temp['Cabin'].apply(lambda x: x.split('/')[0])
        X_temp['Num'] = X_temp['Cabin'].apply(lambda x: float(x.split('/')[1]))
        self.max_num_per_deck = X_temp.groupby('Deck')['Num'].max().to_dict()
        
        # Unsupervised Fitting
        self.kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
        self.pca = PCA(n_components=2, random_state=42)
        numeric_matrix = self._get_numeric_matrix(X_temp)
        self.kmeans.fit(numeric_matrix)
        self.pca.fit(numeric_matrix)
        
        # --- Percentile Learning for Winsorization (New in V10) ---
        self.caps = {}
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend']
        # We need TotalSpend in X_temp first
        spend_cols_raw = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X_temp['TotalSpend'] = X_temp[spend_cols_raw].sum(axis=1) # NaNs already handled in _get_numeric implicitly or not?
        # Let's do a quick robust calc
        for c in spend_cols_raw:
            X_temp[c] = X_temp[c].fillna(0)
        X_temp['TotalSpend'] = X_temp[spend_cols_raw].sum(axis=1)
        
        for col in spend_feats:
            if col in X_temp.columns:
                self.caps[col] = X_temp[col].quantile(0.99) # Cap at 99th percentile
        
        return self

    def _get_numeric_matrix(self, X_input):
        X = X_input.copy()
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
        X[spend_feats] = X[spend_feats].fillna(0)
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        X['Age'] = X['Age'].fillna(X['Age'].median())
        for col in spend_feats + ['TotalSpend']:
            X[col] = np.log1p(X[col])
        return X[spend_feats + ['TotalSpend', 'Age']]

    def transform(self, X):
        X = X.copy()
        
        # --- Winsorization (Clipping Outliers) ---
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        # Fill zero temporarily for logic
        for c in spend_feats:
            if c in self.caps:
                X[c] = X[c].fillna(0).clip(upper=self.caps[c])

        # Recalculate TotalSpend AFTER clipping
        X['TotalSpend'] = X[spend_feats].sum(axis=1)

        # --- Manifold Features ---
        numeric_matrix = self._get_numeric_matrix(X)
        X['ClusterID'] = self.kmeans.predict(numeric_matrix)
        pca_comps = self.pca.transform(numeric_matrix)
        X['PCA1'] = pca_comps[:, 0]
        X['PCA2'] = pca_comps[:, 1]
        
        # --- Social & Spatial ---
        X['PassengerGroup'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('PassengerGroup')['PassengerGroup'].transform('count')
        X['Name'] = X['Name'].fillna('Unknown Unknown')
        X['Surname'] = X['Name'].apply(lambda x: x.split(' ')[-1])
        X['FamilySizeSurname'] = X['Surname'].map(self.surname_counts).fillna(1)
        
        X['Cabin'] = X['Cabin'].fillna('T/0/P')
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Num'] = X['Cabin'].apply(lambda x: float(x.split('/')[1]))
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        X['CabinRegion'] = pd.cut(X['Num'], bins=[-1, 300, 600, 1000, 1500, 2500], labels=[0, 1, 2, 3, 4], right=True).astype(float).fillna(0)
        X['CabinOccupancy'] = X['Cabin'].map(self.cabin_counts).fillna(1)
        X['RelCabinPos'] = X.apply(lambda row: row['Num'] / self.max_num_per_deck.get(row['Deck'], 2000), axis=1)

        # --- Physics ---
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0 # Force consistency
        X['ZeroSpend'] = (X['TotalSpend'] == 0).astype(int)
        
        # Log Transforms (Post-Clipping)
        for col in spend_feats + ['TotalSpend']:
            X[col] = np.log1p(X[col])

        drop_cols = ['Transported', 'PassengerId', 'Name', 'Surname', 'Cabin', 'PassengerGroup', 'Num']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_pipeline(model):
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'CabinRegion', 'ClusterID'] 
    
    num_cols = ['Age', 'TotalSpend', 'RelCabinPos', 'GroupSize', 'FamilySizeSurname', 'CabinOccupancy', 
                'ZeroSpend', 'PCA1', 'PCA2']
    num_cols += ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=15)),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42)) 
    ])
    
    from sklearn.preprocessing import OrdinalEncoder
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    return Pipeline(steps=[('fe', ManifoldFeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', model)])

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    # Global Fit FE
    fe_global = ManifoldFeatureEngineer()
    all_data = pd.concat([train.drop('Transported', axis=1), test], axis=0)
    fe_global.fit(all_data)
    
    # Prepare Data
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # --- STEP 1: Feature Transformation First (for Outlier Detection) ---
    # We need to transform X into numerical features to run IsolationForest
    # We'll use a temporary pipeline just for FE -> Preprocessing
    print("Transforming data for Outlier Detection...")
    temp_pipeline = get_pipeline(None) # Just FE + Pre
    # Pipeline ends with model=None, so steps[:-1] is ('fe', 'pre')
    pre_steps = Pipeline(temp_pipeline.steps[:-1])
    
    X_processed = pre_steps.fit_transform(X, y) # ManifoldFE will refit locally, that's fine/consistent
    
    # --- STEP 2: Isolation Forest Filtering ---
    # Assume ~2% contamination
    print("Running Isolation Forest...")
    iso = IsolationForest(contamination=0.02, random_state=42)
    outliers = iso.fit_predict(X_processed) 
    # -1 is outlier, 1 is inlier
    
    mask = outliers != -1
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Original samples: {len(X)}. Outliers removed: {sum(outliers == -1)}. Clean samples: {len(X_clean)}.")
    
    # --- STEP 3: Train Ensemble on Clean Data ---
    
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, l2_regularization=2.0, max_leaf_nodes=50, random_state=42)
    xt = ExtraTreesClassifier(n_estimators=450, min_samples_leaf=3, random_state=99)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, subsample=0.8, random_state=77)

    # Bagged HGB
    bag_hgb = BaggingClassifier(hgb, n_estimators=10, max_samples=0.75, random_state=55)

    final_ensemble = VotingClassifier(
        estimators=[
            ('bag', get_pipeline(bag_hgb)),
            ('xt', get_pipeline(xt)),
            ('gb', get_pipeline(gb))
        ],
        voting='soft',
        weights=[3, 1, 1]
    )
    
    print("Training Outlier-Cleaned Ensemble V10...")
    final_ensemble.fit(X_clean, y_clean)
    
    print("Predicting...")
    preds = final_ensemble.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
