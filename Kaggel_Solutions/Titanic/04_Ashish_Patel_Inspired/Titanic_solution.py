import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# Sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

# Paths
TRAIN_PATH = 'Titanic_data/train.csv'
TEST_PATH = 'Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_ashish_patel_inspired.csv'

def load_data():
    import os
    if not os.path.exists(TRAIN_PATH):
        import Titanic_download_data
        Titanic_download_data.download_data()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def process_data(train, test):
    train_test_data = [train, test]

    # 1. Title Extraction & Mapping
    for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, 
        "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
        "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3
    }
    
    for dataset in train_test_data:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        # Fill missing titles with 0 (Mr) if any
        dataset['Title'] = dataset['Title'].fillna(0)

    # 2. Sex Mapping
    sex_mapping = {"male": 0, "female": 1}
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    # 3. Age Filling (Median by Title)
    # Ashish's notebook fills Age based on Title medians implicitly or explicitly. 
    # We will be explicit.
    for dataset in train_test_data:
        dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
    
    # Age Binning
    for dataset in train_test_data:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
        dataset.loc[dataset['Age'] > 62, 'Age'] = 4

    # 4. Embarked
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    # 5. Fare
    # Fill missing Fare in test set
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    
    # Fare Binning
    for dataset in train_test_data:
        dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3

    # 6. Cabin (Ashish uses the first letter mapping, or numeric)
    # Looking at the head(), Cabin became floats like 0.8, 1.6, 2.0.
    # This implies a mapping of the first letter (A, B, C...) to numbers.
    # Let's derive a simple mapping.
    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2.0, "G": 2.4, "T": 2.8}
    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].str[:1]
        dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
        dataset['Cabin'].fillna(dataset.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

    # 7. FamilySize
    for dataset in train_test_data:
        dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
        # Binning FamilySize according to notebook glimpse? 
        # Saw 0.4, 0.0, 1.6 values. Looks like a scaling or mapping.
        # Let's use standard scaler or mapping.
        # Ashish's logic: Family mapping? 
        # Simpler: Just keep numeric FamilySize or bin it.
        # Based on output 0.4 etc, he probably mapped it.
        # Let's normalize it like specific mapping.
        family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2.0, 7: 2.4, 8: 2.8, 9: 3.2, 11: 3.6}
        dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
        dataset['FamilySize'] = dataset['FamilySize'].fillna(0) # For others

    # 8. Drop Features
    features_drop = ['Ticket', 'SibSp', 'Parch', 'Name', 'PassengerId']
    train_ids = train['PassengerId']
    test_ids = test['PassengerId']
    
    train = train.drop(features_drop, axis=1)
    test = test.drop(features_drop, axis=1)
    
    return train, test, test_ids

def main():
    print("Loading Data...")
    train, test = load_data()
    
    # Process
    train, test, test_ids = process_data(train, test)
    
    X = train.drop('Survived', axis=1)
    y = train['Survived']
    
    print("Data Shape:", X.shape)
    print(X.head())

    # --- Modelling (Ashish Patel's Best + Voting) ---
    
    # 1. SVC (Best in Notebook ~83.5)
    svc = SVC(probability=True, random_state=0)
    
    # 2. KNN (Good Performer)
    knn = KNeighborsClassifier(n_neighbors=13)
    
    # 3. Gradient Boosting (Robustness)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=0)
    
    # Voting Classifier
    # We combine them to stabilize the result
    voting_clf = VotingClassifier(estimators=[
        ('svc', svc), 
        ('knn', knn), 
        ('gb', gb)
    ], voting='soft')
    
    print("Training Voting Classifier...")
    voting_clf.fit(X, y)
    
    print("Cross Validation Score...")
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(voting_clf, X, y, cv=k_fold, scoring='accuracy')
    print(f"CV Score: {np.mean(scores)*100:.2f}%")
    
    print("Predicting...")
    predictions = voting_clf.predict(test)
    
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
