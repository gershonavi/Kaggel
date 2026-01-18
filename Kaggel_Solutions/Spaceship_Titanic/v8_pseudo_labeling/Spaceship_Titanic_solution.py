import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_v8_pseudo.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

# --- V7 Logic Re-used (The "Best" Logic so far) ---
class DeepFeatureEngineer(BaseEstimator, TransformerMixin):
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
        return self

    def transform(self, X):
        X = X.copy()
        
        # Social
        X['PassengerGroup'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('PassengerGroup')['PassengerGroup'].transform('count')
        X['Name'] = X['Name'].fillna('Unknown Unknown')
        X['Surname'] = X['Name'].apply(lambda x: x.split(' ')[-1])
        X['FamilySizeSurname'] = X['Surname'].map(self.surname_counts).fillna(1)
        
        # Spatial
        X['Cabin'] = X['Cabin'].fillna('T/0/P')
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Num'] = X['Cabin'].apply(lambda x: float(x.split('/')[1]))
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        X['CabinOccupancy'] = X['Cabin'].map(self.cabin_counts).fillna(1)
        X['RelCabinPos'] = X.apply(lambda row: row['Num'] / self.max_num_per_deck.get(row['Deck'], 2000), axis=1)

        # Physics
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool)
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        X['ZeroSpend'] = (X['TotalSpend'] == 0).astype(int)
        X['LuxuryRatio'] = (X['Spa'] + X['VRDeck'] + X['RoomService']) / (X['TotalSpend'] + 1)
        X['FoodRatio'] = (X['FoodCourt'] + X['ShoppingMall']) / (X['TotalSpend'] + 1)
        
        # Age
        X['Age'] = X['Age'].fillna(X['Age'].median())
        X['IsChild'] = (X['Age'] < 13).astype(int)
        X['IsElder'] = (X['Age'] > 60).astype(int)
        
        for col in spend_feats + ['TotalSpend']:
            X[col] = np.log1p(X[col])

        drop_cols = ['Transported', 'PassengerId', 'Name', 'Surname', 'Cabin', 'PassengerGroup', 'Num']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_pipeline(model):
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    num_cols = ['Age', 'TotalSpend', 'LuxuryRatio', 'FoodRatio', 'RelCabinPos', 'GroupSize', 
                'FamilySizeSurname', 'CabinOccupancy', 'IsChild', 'IsElder', 'ZeroSpend']
    num_cols += ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=15)),
        ('scaler', QuantileTransformer(output_distribution='normal')) 
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
    
    return Pipeline(steps=[('fe', DeepFeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', model)])

def get_ensemble():
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, l2_regularization=3.0, random_state=42)
    xt = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=4, bootstrap=False, random_state=99)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, subsample=0.8, random_state=77)
    bag_hgb = BaggingClassifier(hgb, n_estimators=12, max_samples=0.7, random_state=55)

    return VotingClassifier(
        estimators=[
            ('bag', get_pipeline(bag_hgb)),
            ('xt', get_pipeline(xt)),
            ('gb', get_pipeline(gb))
        ],
        voting='soft',
        weights=[3, 1, 1]
    )

def main():
    train, test = load_data()
    test_ids = test['PassengerId']
    
    # Pre-fit FE on everything for counts (Safe count trick)
    fe_global = DeepFeatureEngineer()
    all_data = pd.concat([train.drop('Transported', axis=1), test], axis=0)
    fe_global.fit(all_data)
    
    # 1. First Pass: Train on pure Train
    X_train = train.drop('Transported', axis=1)
    y_train = train['Transported'].astype(int)
    
    print("Training Initial Model...")
    model = get_ensemble()
    model.fit(X_train, y_train)
    
    # 2. Pseudo-Labeling Step
    print("Generating Pseudo-Labels...")
    probs = model.predict_proba(test)
    probs_max = probs.max(axis=1)
    
    # Confidence Threshold (High confidence only)
    THRESHOLD = 0.95
    idx_pseudo = np.where(probs_max > THRESHOLD)[0]
    
    print(f"Found {len(idx_pseudo)} confident test samples ({(len(idx_pseudo)/len(test))*100:.2f}%)")
    
    if len(idx_pseudo) > 50: # Only proceed if significant
        X_pseudo = test.iloc[idx_pseudo].copy()
        # Get the predicted label
        y_pseudo = model.predict(X_pseudo)
        
        # Augment Training Data
        X_augmented = pd.concat([X_train, X_pseudo], axis=0)
        y_augmented = np.concatenate([y_train, y_pseudo], axis=0)
        
        # 3. Retrain on Augmented Data
        print("Retraining on Augmented Data (Self-Training)...")
        model_final = get_ensemble()
        model_final.fit(X_augmented, y_augmented)
        
        # Final Predict
        print("Final Prediction...")
        preds = model_final.predict(test)
    else:
        print("Not enough confident samples. Using initial model.")
        preds = model.predict(test)
        
    preds_bool = preds.astype(bool)
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
