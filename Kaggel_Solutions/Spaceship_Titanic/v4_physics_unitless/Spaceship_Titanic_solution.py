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
SUBMISSION_PATH = 'submission_physics.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

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
        
        # --- 1. Thermodynamic/Spend Features ---
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        
        # Physics Rule: Locked systems (Cryo) have 0 entropy/spend
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        
        # Unitless Ratios (Distributions)
        # Avoid division by zero by adding 1 to denominator
        # "Luxury Fraction": (Spa + VR + RoomService) / Total
        X['LuxurySpend'] = X['Spa'] + X['VRDeck'] + X['RoomService']
        X['LuxuryRatio'] = X['LuxurySpend'] / (X['TotalSpend'] + 1)
        
        # "Sustenance Fraction": (Food + Mall) / Total
        X['SustenanceSpend'] = X['FoodCourt'] + X['ShoppingMall']
        X['SustenanceRatio'] = X['SustenanceSpend'] / (X['TotalSpend'] + 1)
        
        # "Spend Density" (Spend per year of age) -> Power output
        # Age 0 needs imputation for division
        X['Age_Imp'] = X['Age'].fillna(X['Age'].median()).clip(lower=1)
        X['SpendPower'] = X['TotalSpend'] / X['Age_Imp']

        # Log transform magnitudes for models
        for col in spend_feats + ['TotalSpend', 'LuxurySpend', 'SustenanceSpend', 'SpendPower']:
            X[col] = np.log1p(X[col].fillna(0))

        # --- 2. Spatial/Geometric Features ---
        X['Cabin'] = X['Cabin'].fillna('T/0/P') 
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Num'] = X['Cabin'].apply(lambda x: float(x.split('/')[1]))
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        
        # Normalized Longitudinal Position (0.0 to 1.0 along the ship)
        # Uses fit-time max values to be consistent
        X['RelCabinPos'] = X.apply(lambda row: row['Num'] / self.max_num_per_deck.get(row['Deck'], 2000), axis=1)
        
        # --- 3. Social Physics ---
        X['Group'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('Group')['Group'].transform('count')
        
        # Unitless Group Ratio: Rank in group / Group Size? 
        X['PassengerNum'] = X['PassengerId'].apply(lambda x: int(x.split('_')[1]))
        X['GroupRankNorm'] = X['PassengerNum'] / X['GroupSize']

        # Drop raw high card
        drop_cols = ['Transported', 'PassengerId', 'Name', 'Cabin', 'Group', 'Num', 'Age_Imp']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_base_pipeline(model):
    # Columns
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    # Add new physics columns
    num_cols = ['Age', 'TotalSpend', 'LuxuryRatio', 'SustenanceRatio', 'SpendPower', 'RelCabinPos', 'GroupSize', 'GroupRankNorm']
    # Also individual spend logs
    for c in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        if c in num_cols: pass # check if needed
    num_cols += ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'] # Add them back as they are useful raw logs

    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=9)),
        ('scaler', QuantileTransformer(output_distribution='normal')) # Force Gaussian for "Physics" variables
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
    
    return Pipeline(steps=[('fe', PhysicsFeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', model)])

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # --- Ensemble Strategy ---
    # Keeping the strong Trunk Bagging structure but with the new Feature Set
    
    # Trunk 1: HGB (The heavy lifter)
    trunk_1 = BaggingClassifier(
        estimator=HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=40, random_state=42),
        n_estimators=10, 
        max_samples=0.65, 
        bootstrap=True, 
        random_state=101
    )
    
    # Trunk 2: ExtraTrees (Geometry lover)
    trunk_2 = ExtraTreesClassifier(n_estimators=350, min_samples_leaf=3, random_state=102)
    
    # Trunk 3: GradientBoosting (Classic)
    trunk_3 = GradientBoostingClassifier(n_estimators=250, subsample=0.75, learning_rate=0.05, random_state=103)

    final_ensemble = VotingClassifier(
        estimators=[
            ('bag_hgb', get_base_pipeline(trunk_1)),
            ('xt', get_base_pipeline(trunk_2)),
            ('gb', get_base_pipeline(trunk_3))
        ],
        voting='soft',
        weights=[3, 1, 1] 
    )
    
    print("Training Physics-Enriched Ensemble...")
    final_ensemble.fit(X, y)
    
    print("Predicting...")
    preds = final_ensemble.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
