import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
TRAIN_PATH = 'Titanic_data/train.csv'
TEST_PATH = 'Titanic_data/test.csv'
SUBMISSION_PATH = 'submission_advanced.csv'

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

class FeatureEngineer:
    def __init__(self):
        self.age_medians = {}
        self.fare_median = 0
        self.title_encoder = LabelEncoder()
        self.embarked_mode = 'S'
    
    def get_title(self, name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""

    def simplify_title(self, title):
        if title in ['Mr']: return 0
        if title in ['Miss', 'Mlle', 'Ms']: return 1
        if title in ['Mrs', 'Mme']: return 2
        if title in ['Master']: return 3
        return 4 # Rare/Other

    def fit(self, df):
        # Embarked mode
        self.embarked_mode = df['Embarked'].mode()[0]
        
        # Fare median
        self.fare_median = df['Fare'].median()
        
        # Age medians by Pclass and Sex (simple but effective)
        self.age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()

    def transform(self, df):
        df = df.copy()
        
        # Feature: Title
        df['Title'] = df['Name'].apply(self.get_title)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Map Title to integer
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna(0)

        # Feature: Sex
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
        
        # Feature: Embarked
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        
        # Feature: FamilySize
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Feature: IsAlone
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        
        # Feature: Fare - Handle missing and bin/scale
        df['Fare'] = df['Fare'].fillna(self.fare_median)
        # Log transform fare to reduce skew
        df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
        
        # Feature: Age - Impute
        # We'll use a crude lookup
        for idx, row in df.iterrows():
            if np.isnan(row['Age']):
                key = (row['Pclass'], 0 if row['Sex']==0 else 1) # Sex is already mapped? Wait, sex mapped above.
                # Actually let's just use the Pclass alone if Sex is tricky or re-map.
                # Let's rely on the grouped median we calc'd strictly.
                # Re-map sex back for lookup or just use numerical? 
                # My fit used strings.
                s_str = 'male' if row['Sex'] == 0 else 'female'
                df.at[idx, 'Age'] = self.age_medians.get((row['Pclass'], s_str), 28)

        # Drop unused
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        if 'Survived' in df.columns:
            drop_cols.append('Survived')
        df = df.drop(drop_cols, axis=1)
        
        return df

def main():
    print("Loading data...")
    train_raw, test_raw = load_data()
    
    print("Feature Engineering...")
    fe = FeatureEngineer()
    fe.fit(train_raw)
    
    X_train = fe.transform(train_raw)
    y_train = train_raw['Survived']
    X_test = fe.transform(test_raw)
    
    # Scale data for SVC
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Ensemble Model...")
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=3)
    
    # 2. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3)
    
    # 3. SVC (requires scaling) - we will put it in a pipeline if we were strict, 
    # but for simplicity we will just use Tree based here to avoid scaling mess in Voting without Pipeline.
    # Actually, let's stick to Tree-based ensemble for robustness against scaling.
    # RF + GB is a strong combo.
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    # Cross Validation
    scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}")
    
    ensemble.fit(X_train, y_train)
    
    print("Predicting...")
    predictions = ensemble.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Survived': predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved advanced submission to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
