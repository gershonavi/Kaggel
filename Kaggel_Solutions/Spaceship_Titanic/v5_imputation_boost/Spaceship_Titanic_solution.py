import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_imputation.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class AdvancedImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # --- Smart Imputation (Group & Physics Based) ---
        X['Group'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('Group')['Group'].transform('count')
        
        # 1. Fill CryoSleep based on Spend
        # If Spend metrics > 0, CryoSleep MUST be False
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['TotalSpend_Raw'] = X[spend_cols].sum(axis=1)
        X.loc[(X['CryoSleep'].isna()) & (X['TotalSpend_Raw'] > 0), 'CryoSleep'] = False
        
        # 2. Fill Missing HomePlanet based on Group
        # People in same group usually come from same planet
        # Map Group -> HomePlanet mode
        group_planet_map = X.groupby('Group')['HomePlanet'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        X['HomePlanet'] = X['HomePlanet'].fillna(X['Group'].map(group_planet_map))
        
        # 3. Fill Cabin features based on Group?
        # Often groups are in same cabin or adjacent. 
        # But Cabin is Deck/Num/Side.
        # Let's verify if Group predicts Deck.
        X['Cabin'] = X['Cabin'].fillna('T/0/P') # Temp fill for splitting
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])

        # Restore NaNs where we filled temp 'T/0/P' if original was NaN? 
        # Actually proper imputation is complex here.
        # Let's do simple Group-Deck propogation if missing
        # If Cabin was missing, Deck is 'T' (from our fillna). 
        # Check if we can improve this.
        
        # (For V5 we stick to the strong model + some smart Cryo/Planet filling which is high yield)
        
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Recalculate basic physics features on top of cleaned data
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Entropy Rule: Cryo = True -> Spend = 0
        for col in spend_cols:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_cols].sum(axis=1)
        X['LuxurySpend'] = X['Spa'] + X['VRDeck'] + X['RoomService']
        X['SustenanceSpend'] = X['FoodCourt'] + X['ShoppingMall']
        
        # Log transforms
        for col in spend_cols + ['TotalSpend', 'LuxurySpend', 'SustenanceSpend']:
            X[col] = np.log1p(X[col].fillna(0))
            
        # Cabin Parsing (Refined)
        X['CabinNum'] = X['Cabin'].apply(lambda x: float(x.split('/')[1]) if x != 'T/0/P' else 0)
        
        # Deck Mapping (Ordinal can be useful: A,B,C...T)
        deck_order = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}
        X['Deck_Ord'] = X['Deck'].map(deck_order).fillna(8)
        
        # Drop
        drop_cols = ['Transported', 'PassengerId', 'Name', 'Cabin', 'Group', 'TotalSpend_Raw']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_base_pipeline(model):
    # Columns
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    num_cols = ['Age', 'TotalSpend', 'LuxurySpend', 'SustenanceSpend', 'CabinNum', 'Deck_Ord', 'GroupSize']
    # Add raw individual spend logs
    num_cols += ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=15)), # Increased neighbors for robustness
        ('scaler', RobustScaler())
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
    
    return Pipeline(steps=[('imp', AdvancedImputer()),
                           ('fe', FeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', model)])

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # --- Ensemble Strategy V5 ---
    # Stronger heavy HGB + Random Forest (good for categorical interactions) + Logistic Stacking/Voting
    
    # HGB 1: High variance
    hgb1 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=60, l2_regularization=0.5, random_state=42)
    # HGB 2: Low variance
    hgb2 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=43)
    
    # Extra Trees (Geometric)
    xt = ExtraTreesClassifier(n_estimators=400, min_samples_split=4, random_state=44)
    
    final_ensemble = VotingClassifier(
        estimators=[
            ('hgb_deep', get_base_pipeline(hgb1)),
            ('hgb_fast', get_base_pipeline(hgb2)),
            ('xt', get_base_pipeline(xt))
        ],
        voting='soft',
        weights=[3, 1, 2] 
    )
    
    print("Training Imputation-Boosted Ensemble...")
    final_ensemble.fit(X, y)
    
    print("Predicting...")
    preds = final_ensemble.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
