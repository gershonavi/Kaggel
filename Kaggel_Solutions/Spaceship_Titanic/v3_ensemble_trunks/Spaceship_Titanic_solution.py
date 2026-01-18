import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_trunks.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Total Spend & Interactions
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        # Intelligent fill: if CryoSleep is True, Spend is 0.
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        X['NoSpending'] = (X['TotalSpend'] == 0).astype(int)
        
        # Log transform skewed money
        for col in spend_feats + ['TotalSpend']:
            X[col] = np.log1p(X[col].fillna(0))

        # 2. Cabin Decoding
        X['Cabin'] = X['Cabin'].fillna('T/0/P') 
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        X['DeckSide'] = X['Deck'] + X['Side'] # Interaction feature
        
        # 3. Group Logic
        X['Group'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('Group')['Group'].transform('count')
        
        # Drop
        drop_cols = ['Transported', 'PassengerId', 'Name', 'Cabin', 'Group']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_base_pipeline(model):
    # Columns
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'DeckSide']
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'GroupSize']
    
    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    
    # Ordinal for trees to preserve some separation
    from sklearn.preprocessing import OrdinalEncoder
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Handle unknown categories in test
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    return Pipeline(steps=[('fe', AdvancedFeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', model)])

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # --- Strategy: Forced Diversity "Trunks" ---
    
    # Trunk 1: Bagging HistGradient (LightGBM-style) - subsample 0.7 to force variance
    # "Bagging with different subsets"
    hgb = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, random_state=42)
    trunk_1 = BaggingClassifier(
        estimator=hgb, 
        n_estimators=10, # 10 different HistGradient models on different subsets
        max_samples=0.7, 
        bootstrap=True, 
        random_state=1
    )
    
    # Trunk 2: ExtraTrees (Randomized splits) - naturally diverse
    # Subsampled features per split
    trunk_2 = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=2)
    
    # Trunk 3: GradientBoosting (Standard) - different architecture/loss
    trunk_3 = GradientBoostingClassifier(n_estimators=200, subsample=0.8, random_state=3)
    
    # Meta-Ensemble: Soft Voting
    final_ensemble = VotingClassifier(
        estimators=[
            ('bag_hgb', get_base_pipeline(trunk_1)),
            ('xt', get_base_pipeline(trunk_2)), # Extra Trees often very strong on tabular
            ('gb', get_base_pipeline(trunk_3))
        ],
        voting='soft',
        weights=[2, 1, 1] # Giving more weight to the Bagged HGB trunk
    )
    
    print("Training Truncated Bagging Ensemble...")
    final_ensemble.fit(X, y)
    
    print("Predicting...")
    preds = final_ensemble.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
