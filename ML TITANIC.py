import pandas as pd
import  seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

data = sns.load_dataset("titanic")
print("First 5 rows:")  
print(data.head())

data = data[
    ["survived", "pclass", "sex", "age", "sibsp",
     "parch", "fare", "embarked"]
]

data.columns = [
    "Survived", "Pclass", "Sex", "Age",
    "SibSp", "Parch", "Fare", "Embarked"
]

print("\n column names:")
print(data.columns)


print("\n missing value:")
print(data.isnull().sum())


print("\n zero value :")
print((data == 0).sum())

data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

data["Sex"] = data["Sex"].map({
    "male": 0,
    "female": 1
})

data["Embarked"] = data["Embarked"].map({
    "C": 0,
    "Q": 1,
    "S": 2
})

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n accuracy:")
print(accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\n confusion matrix:")
print(cm)


print("\n classification report:")
print(classification_report(y_test, y_pred))

y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("ROC-AUC Score:")
print(roc_auc)

print("\n feature importance:")

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

importance = model.feature_importances_

for name, value in zip(features, importance):
    print(name, ":", value)