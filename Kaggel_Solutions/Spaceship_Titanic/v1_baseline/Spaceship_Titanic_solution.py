import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Data mappings
# HomePlanet, CryoSleep, Cabin, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Name, Transported

def preprocess(df):
    df = df.copy()
    
    # Drop Name, Cabin for baseline
    df = df.drop(['Name', 'Cabin'], axis=1)
    
    # Impute missing categories with mode
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
        
    # Impute missing numericals with median
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for n in num_cols:
        df[n] = df[n].fillna(df[n].median())
        
    # Encode categories
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = df[c].astype(str)
        df[c] = le.fit_transform(df[c])
        
    return df

def main():
    print("Loading Data...")
    try:
        train = pd.read_csv('Spaceship_Titanic_data/train.csv')
        test = pd.read_csv('Spaceship_Titanic_data/test.csv')
    except FileNotFoundError:
        print("Data not found. Please ensure data is in 'data/' folder.")
        return

    # PassengerId specific to test for submission
    test_ids = test['PassengerId']
    
    print("Preprocessing...")
    train = preprocess(train)
    test = preprocess(test)
    
    # Drop PassengerId from train features
    X = train.drop(['PassengerId', 'Transported'], axis=1)
    y = train['Transported'].astype(int) # True/False to 1/0
    
    X_test = test.drop('PassengerId', axis=1)
    
    print("Training Baseline RF...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    print("Predicting...")
    preds = clf.predict(X_test)
    preds_bool = preds.astype(bool) # Submission needs True/False
    
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': preds_bool})
    sub.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    main()
