import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_hybrid.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

# --- STEP 1: V5 Smart Imputation Logic ---
class HybridImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Group Helper
        X['Group'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        
        # 2. Fill CryoSleep using Spend (Hard Rule)
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['TotalSpend_Raw'] = X[spend_cols].sum(axis=1)
        # If they spent money, they are NOT in CryoSleep
        X.loc[(X['CryoSleep'].isna()) & (X['TotalSpend_Raw'] > 0), 'CryoSleep'] = False
        
        # 3. Fill HomePlanet using Group Mode (The "Group Logic" winner)
        group_planet_map = X.groupby('Group')['HomePlanet'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        X['HomePlanet'] = X['HomePlanet'].fillna(X['Group'].map(group_planet_map))
        
        return X

# --- STEP 2: V4 Physics Feature Engineering ---
class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate global max cabin num per deck for normalization
        X_temp = X.copy()
        X_temp['Cabin'] = X_temp['Cabin'].fillna('T/0/P')
        X_temp['Deck'] = X_temp['Cabin'].apply(lambda x: x.split('/')[0])
        X_temp['Num'] = X_temp['Cabin'].apply(lambda x: int(x.split('/')[1]))
        
        self.max_num_per_deck = X_temp.groupby('Deck')['Num'].max().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Thermodynamic/Spend Features
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        
        # Physics Rule: Locked systems (Cryo) have 0 entropy/spend
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        
        # Unitless Ratios (From V4 - The 0.801 booster)
        X['LuxurySpend'] = X['Spa'] + X['VRDeck'] + X['RoomService']
        X['LuxuryRatio'] = X['LuxurySpend'] / (X['TotalSpend'] + 1)
        
        X['SustenanceSpend'] = X['FoodCourt'] + X['ShoppingMall']
        X['SustenanceRatio'] = X['SustenanceSpend'] / (X['TotalSpend'] + 1)
        
        X['Age_Imp'] = X['Age'].fillna(X['Age'].median()).clip(lower=1)
        X['SpendPower'] = X['TotalSpend'] / X['Age_Imp']

        # Log transform
        for col in spend_feats + ['TotalSpend', 'LuxurySpend', 'SustenanceSpend', 'SpendPower']:
            X[col] = np.log1p(X[col].fillna(0))

        # 2. Geometric Features (RelCabinPos)
        X['Cabin'] = X['Cabin'].fillna('T/0/P') 
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Num'] = X['Cabin'].apply(lambda x: float(x.split('/')[1]))
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        
        # Normalized Longitudinal Position (From V4)
        X['RelCabinPos'] = X.apply(lambda row: row['Num'] / self.max_num_per_deck.get(row['Deck'], 2000), axis=1)
        
        # 3. Group features
        if 'Group' not in X.columns:
            X['Group'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('Group')['Group'].transform('count')
        
        # Drop
        drop_cols = ['Transported', 'PassengerId', 'Name', 'Cabin', 'Group', 'Num', 'Age_Imp', 'TotalSpend_Raw']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_base_pipeline(model):
    # Columns
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    # Combined Physics + Spend + Group Size
    num_cols = ['Age', 'TotalSpend', 'LuxuryRatio', 'SustenanceRatio', 'SpendPower', 'RelCabinPos', 'GroupSize']
    # Keep raw logs too
    num_cols += ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=9)),
        ('scaler', QuantileTransformer(output_distribution='normal')) # V4's scaler (worked best)
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
    
    return Pipeline(steps=[('imp', HybridImputer()), # V5 Logic
                           ('fe', PhysicsFeatureEngineer()), # V4 Logic
                           ('pre', preprocessor),
                           ('model', model)])

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # --- Ensemble Strategy (V4 Architecture) ---
    # We switch back to the V4 Architecture (Trunk Bagging) because it scored higher (0.801) than V5's.
    # But we feed it better data (V5 Imputation + V4 Physics Features).
    
    # Trunk 1: Bagging HGB
    trunk_1 = BaggingClassifier(
        estimator=HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=40, random_state=42),
        n_estimators=10, 
        max_samples=0.65, 
        bootstrap=True, 
        random_state=101
    )
    
    # Trunk 2: ExtraTrees
    trunk_2 = ExtraTreesClassifier(n_estimators=350, min_samples_leaf=3, random_state=102)
    
    # Trunk 3: GradientBoosting
    trunk_3 = GradientBoostingClassifier(n_estimators=250, subsample=0.75, learning_rate=0.05, random_state=103)

    final_ensemble = VotingClassifier(
        estimators=[
            ('bag_hgb', get_base_pipeline(trunk_1)),
            ('xt', get_base_pipeline(trunk_2)), # V4/V3 structure
            ('gb', get_base_pipeline(trunk_3))
        ],
        voting='soft',
        weights=[3, 1, 1] 
    )
    
    print("Training Hybrid V6 (Best of Both)...")
    final_ensemble.fit(X, y)
    
    print("Predicting...")
    preds = final_ensemble.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
