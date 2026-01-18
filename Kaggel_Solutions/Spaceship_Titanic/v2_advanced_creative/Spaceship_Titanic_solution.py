import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Paths
TRAIN_PATH = 'Spaceship_Titanic_data/train.csv'
TEST_PATH = 'Spaceship_Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_creative.csv'

def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

class CreativeFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Total Spend
        spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        # Fill None with 0 for total calc (we will impute individual later, but 0 is safe assumption for sum if missing often means 0)
        # Actually, let's keep NaNs propagate so we don't skew, OR rely on the fact that missing often equals 0 interactions.
        # Strategically: If CryoSleep is True, Spend IS 0.
        
        # CryoSleep Handling
        X['CryoSleep'] = X['CryoSleep'].fillna(False).astype(bool) # Assume awake if missing for safer impute? Or Mode.
        # Allow Imputer to handle Cryo missing properly later, but for Spend rule:
        
        for col in spend_feats:
            X.loc[X['CryoSleep'] == True, col] = 0.0
            
        X['TotalSpend'] = X[spend_feats].sum(axis=1)
        X['NoSpending'] = (X['TotalSpend'] == 0).astype(int)
        
        # 2. Cabin Decoding: Deck/Num/Side
        # Cabin format: Deck/Num/Side
        # Handle missing cabins
        X['Cabin'] = X['Cabin'].fillna('T/0/P') # T is rare, 0 is null, P is placeholder
        
        X['Deck'] = X['Cabin'].apply(lambda x: x.split('/')[0])
        X['Side'] = X['Cabin'].apply(lambda x: x.split('/')[2])
        # Deck mapping: combine small decks? Keep 'T' separate or merge? T is very small.
        # Side mapping: P (Port), S (Starboard)
        
        # 3. Group Logic (PassengerId is gggg_pp)
        X['Group'] = X['PassengerId'].apply(lambda x: x.split('_')[0])
        X['GroupSize'] = X.groupby('Group')['Group'].transform('count')
        X['IsSolo'] = (X['GroupSize'] == 1).astype(int)
        
        # 4. Age Binning (Optional, trees handle raw well, but interaction helps)
        # 5. Name - Extract Surname for potential family grouping (Too complex for this step? Skip for now)
        
        # Drop high cardinality or unused
        drop_cols = ['Transported', 'PassengerId', 'Name', 'Cabin', 'Group']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], axis=1)
        
        return X

def get_pipeline():
    # Columns
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'GroupSize']
    
    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder()) # Ordinal/Label encoding for Trees is better often than OneHot for high cardinality, but Sklearn standard is OHE.
        # HistGradientBoosting handles NaNs locally, but we need pipeline compatibility.
        # Let's use OneHot for compatibility with VotingClassifier (RF needs it) or Ordinal if using specific tree logic.
        # For simplicity and robustness with diverse models:
    ])
    
    # Custom Ordinal Encoding helper since Sklearn pipelines can be tricky with LabelEncoder on multiple columns
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
    
    # Ensemble Model
    # combining HistGradient (LightGBM-like) + RandomForest + ExtraTrees
    # HistGradient is SOTA for tabular in sklearn
    
    clf1 = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
    # n_jobs=-1 for RF
    
    ensemble = VotingClassifier(
        estimators=[('hgb', clf1), ('rf', clf2)],
        voting='soft'
    )
    
    return Pipeline(steps=[('fe', CreativeFeatureEngineer()),
                           ('pre', preprocessor),
                           ('model', ensemble)])

def main():
    train, test = load_data()
    
    # PassengerId for submission
    test_ids = test['PassengerId']
    
    X = train.drop('Transported', axis=1)
    y = train['Transported'].astype(int)
    
    # Model
    model = get_pipeline()
    
    print("Training Creative Ensemble...")
    model.fit(X, y)
    
    print("Predicting...")
    preds = model.predict(test)
    preds_bool = preds.astype(bool)
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
