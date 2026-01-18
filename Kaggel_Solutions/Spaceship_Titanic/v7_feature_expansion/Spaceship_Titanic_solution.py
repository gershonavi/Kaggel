import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_v7.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class DeepFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Learn global distributions for scaling or counts
        X_temp = X.copy()
        
        # Cabin Counts (Occupancy)
        X_temp['Cabin'] = X_temp['Cabin'].fillna('T/0/P')
        self.cabin_counts = X_temp['Cabin'].value_counts().to_dict()
        
        # Surname Counts (Family Size proxy)
        X_temp['Name'] = X_temp['Name'].fillna('Unknown Unknown')
        X_temp['Surname'] = X_temp['Name'].apply(lambda x: x.split(' ')[-1])
        self.surname_counts = X_temp['Surname'].value_counts().to_dict()
        
        # Max Num per Deck (for relative pos)
        X_temp['Deck'] = X_temp['Cabin'].apply(lambda x: x.split('/')[0])
        X_temp['Num'] = X_temp['Cabin'].apply(lambda x: float(x.split('/')[1]))
        self.max_num_per_deck = X_temp.groupby('Deck')['Num'].max().to_dict()
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # --- 1. Social Features (The New Stuff) ---
        X['PassengerGroup'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('PassengerGroup')['PassengerGroup'].transform('count')
        
        # Surname Logic
        X['Name'] = X['Name'].fillna('Unknown Unknown')
        X['Surname'] = X['Name'].apply(lambda x: x.split(' ')[-1])
        # Map fitted counts (handles train/test split consistency if fitted on combined, but here fitted on batch)
        # Note: Ideally fit on Train+Test combined for full family picture, but strictly fit on X here to be pipeline compliant.
        # But if X is just Test, we miss Train families.
        # ALLOWANCE: For Kaggle, usually safe to use global maps if calculated carefully, but let's stick to pipeline fit
        # Self-mapped counts:
        X['FamilySizeSurname'] = X['Surname'].map(self.surname_counts).fillna(1)
        
        # --- 2. Spatial / Cabin Features ---
        X['Cabin'] = X['Cabin'].fillna('T/0/P')
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Num'] = X['Cabin'].apply(lambda x: float(x.split('/')[1]))
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        
        # Cabin Occupancy (How many people in this specific cabin)
        X['CabinOccupancy'] = X['Cabin'].map(self.cabin_counts).fillna(1)
        
        # Relative Position
        X['RelCabinPos'] = X.apply(lambda row: row['Num'] / self.max_num_per_deck.get(row['Deck'], 2000), axis=1)

        # --- 3. Physics / Spend ---
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        X['ZeroSpend'] = (X['TotalSpend'] == 0).astype(int)
        
        # Ratios
        X['LuxuryRatio'] = (X['Spa'] + X['VRDeck'] + X['RoomService']) / (X['TotalSpend'] + 1)
        X['FoodRatio'] = (X['FoodCourt'] + X['ShoppingMall']) / (X['TotalSpend'] + 1)
        
        # --- 4. Age Logic ---
        # IsChild is strong predictor (Children don't pay, higher survival often)
        X['Age'] = X['Age'].fillna(X['Age'].median())
        X['IsChild'] = (X['Age'] < 13).astype(int)
        X['IsElder'] = (X['Age'] > 60).astype(int)
        
        # Log Transforms
        for col in spend_feats + ['TotalSpend']:
            X[col] = np.log1p(X[col])

        # --- 5. Shadow Features (Missing Flags) ---
        # Sometimes missing data IS data (e.g. lost record = catastrophic failure?)
        # X['MissingName'] = (X['Name'] == 'Unknown Unknown').astype(int)
        
        # Drop
        drop_cols = ['Transported', 'PassengerId', 'Name', 'Surname', 'Cabin', 'PassengerGroup', 'Num']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_pipeline(model):
    # Columns
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    # Expanded Numerical Set
    num_cols = ['Age', 'TotalSpend', 'LuxuryRatio', 'FoodRatio', 'RelCabinPos', 'GroupSize', 
                'FamilySizeSurname', 'CabinOccupancy', 'IsChild', 'IsElder', 'ZeroSpend']
    num_cols += ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Preprocessing - Using Quantile again (proved best in V4)
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=15)),
        # Normal distribution forces spread for nice gradients
        ('scaler', QuantileTransformer(output_distribution='normal')) 
    ])
    
    from sklearn.preprocessing import OrdinalEncoder
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Ordinal encoding handles tree splits well
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    return Pipeline(steps=[('fe', DeepFeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', model)])

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    # Fit the Feature Engineer globally to capture full counts?
    # Actually, let's create a concatenated DF for fitting the FE to get proper global counts
    # (Surname counts are better if we know Test surnames exist in Train families)
    fe_global = DeepFeatureEngineer()
    all_data = pd.concat([train.drop('Transported', axis=1), test], axis=0)
    fe_global.fit(all_data)
    
    # But wait, Scikit pipeline expects fit during train. 
    # To use global knowledge safely without leakage, we pass the fitted state.
    # We will hack the pipeline to use this pre-fitted FE.
    
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # --- Ensemble Strategy ---
    # Diversity is key.
    
    # 1. HGB (High Reg): 
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, l2_regularization=3.0, random_state=42)
    
    # 2. ExtraTrees (Geometry):
    xt = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=4, bootstrap=False, random_state=99)
    
    # 3. XGBoost Simulant (GradientBoosting with deeper trees)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, subsample=0.8, random_state=77)

    # 4. Bagging (V4's secret sauce) - wrap HGB
    bag_hgb = BaggingClassifier(hgb, n_estimators=12, max_samples=0.7, random_state=55)

    final_ensemble = VotingClassifier(
        estimators=[
            ('bag', get_pipeline(bag_hgb)),
            ('xt', get_pipeline(xt)),
            ('gb', get_pipeline(gb))
        ],
        voting='soft',
        weights=[3, 1, 1]
    )
    
    # Inject the global FE into pipelines? 
    # Actually, simpler: Pass the global FE data to the specific instances or rely on the class logic.
    # The class logic fits on X passed to it. If we pass Train+Test to fit, it works.
    # For now, let's stick to fitting on Train inside the pipeline to avoid data leakage accusations,
    # though "Family Size" is technically legit to infer from Test set existence.
    # We will let the pipeline fit on Train X. Test X will use Train counts (which is safe/conservative).
    
    print("Training Deep Feature Ensemble V7...")
    final_ensemble.fit(X, y)
    
    print("Predicting...")
    preds = final_ensemble.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
