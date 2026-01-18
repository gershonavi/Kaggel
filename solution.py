import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Paths
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'submission.csv'

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def preprocess(df):
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Fill missing Age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing Fare (only happens in test usually)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Encode Embarked
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Drop unused
    # minimal features for "simple solution"
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Handle any remaining NaNs? (e.g. in other cols if we used them, but we aren't)
    return df[features]

def main():
    print("Loading data...")
    train_raw, test_raw = load_data()
    
    print("Preprocessing...")
    X_train = preprocess(train_raw)
    y_train = train_raw['Survived']
    
    X_test = preprocess(test_raw)
    
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Predicting...")
    predictions = model.predict(X_test)
    
    print("Saving submission...")
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Survived': predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission.head())

if __name__ == "__main__":
    main()
