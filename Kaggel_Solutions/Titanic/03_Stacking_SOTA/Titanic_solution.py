import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Paths
TRAIN_PATH = 'Titanic_data/train.csv'
TEST_PATH = 'Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_sota.csv'

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def preprocess_features(df_train, df_test):
    # Combine for consistent engineering
    n_train = len(df_train)
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # 1. Title Extraction
    df_all['Title'] = df_all['Name'].apply(get_title)
    df_all['Title'] = df_all['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_all['Title'] = df_all['Title'].replace('Mlle', 'Miss')
    df_all['Title'] = df_all['Title'].replace('Ms', 'Miss')
    df_all['Title'] = df_all['Title'].replace('Mme', 'Mrs')
    
    # 2. Family Features
    df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
    df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)
    
    # 3. Deck (from Cabin) - extract first letter, map nans to 'M' (Missing)
    df_all['Deck'] = df_all['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')
    # Group rare decks
    df_all['Deck'] = df_all['Deck'].replace(['T', 'G'], 'M') 

    # 4. Fill Embarked
    df_all['Embarked'] = df_all['Embarked'].fillna(df_all['Embarked'].mode()[0])
    
    # 5. Fill Fare (use Pclass specific median)
    df_all['Fare'] = df_all['Fare'].fillna(df_all.groupby('Pclass')['Fare'].transform('median'))
    df_all['FareLog'] = np.log1p(df_all['Fare']) # Log transform
    
    # 6. Drop non-useful
    df_all = df_all.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1) # Drop original Fare, use Log
    
    # 7. Convert Sex to binary
    df_all['Sex'] = df_all['Sex'].map({'male': 0, 'female': 1})
    
    # 8. Encode Categoricals (Title, Embarked, Deck)
    # We will use Pandas get_dummies or OneHot inside pipeline? 
    # Let's use get_dummies for simplicity before imputation to handle structure
    categorical_cols = ['Title', 'Embarked', 'Deck']
    df_all = pd.get_dummies(df_all, columns=categorical_cols, drop_first=True)
    
    # 9. KNN Imputation for Age
    # We run imputation on the whole set (train+test) to leverage all data structure
    imputer = KNNImputer(n_neighbors=5)
    # Isolate targets if present (Survived) - don't impute using Target? 
    # Actually simpler to drop Survived for imputation then add back.
    if 'Survived' in df_all.columns:
        targets = df_all['Survived']
        features = df_all.drop('Survived', axis=1)
        features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
        features_imputed['Survived'] = targets.values # Add back
        df_all = features_imputed
    else:
        df_all = pd.DataFrame(imputer.fit_transform(df_all), columns=df_all.columns)

    # 10. Split back
    train_proc = df_all[:n_train].copy()
    test_proc = df_all[n_train:].copy()
    
    if 'Survived' in test_proc.columns:
        test_proc = test_proc.drop('Survived', axis=1)
        
    return train_proc, test_proc

def main():
    print("Loading data...")
    train, test = load_data()
    
    print("Feature Engineering & Imputation...")
    train_proc, test_proc = preprocess_features(train, test)
    
    X = train_proc.drop('Survived', axis=1)
    y = train_proc['Survived']
    X_sub = test_proc
    
    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_sub_scaled = scaler.transform(X_sub)
    
    print("Training Stacking Ensemble...")
    
    # Base Estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
        ('svc', SVC(probability=True, kernel='rbf', C=10, gamma=0.1, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=10)),
        ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
    ]
    
    # Stacking
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # CV Check
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Stacking CV Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}")
    
    clf.fit(X_scaled, y)
    
    print("Predicting...")
    predictions = clf.predict(X_sub_scaled).astype(int)
    
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved SOTA submission to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
