import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pd.read_csv("diabetes.csv")

print("first 5 rows:")
print(data.head())
data.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

print("\n column names:")
print(data.columns)

print("\n missing values:")
print(data.isnull().sum())

print("\n zero values:")
print((data == 0).sum())

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns:
    data[column] = data[column].replace(0, data[column].median())

print("\n zero values after the handling:")
print((data == 0).sum())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print("\n accuracy:")
print(accuracy)

cm = confusion_matrix(y_test, y_prediction)
print("\n confusion matrix:")
print(cm)

precision = precision_score(y_test, y_prediction)
print("\n precision:")
print(precision)

recall = recall_score(y_test, y_prediction)
print("\n recall:")
print(recall)


f1 = f1_score(y_test, y_prediction)
print("\n f1-score:")
print(f1)

model2 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model2.fit(X_train, y_train)

y_prediction2 = model2.predict(X_test)

accuracy2 = accuracy_score(y_test, y_prediction2)
precision2 = precision_score(y_test, y_prediction2)
recall2 = recall_score(y_test, y_prediction2)
f12 = f1_score(y_test, y_prediction2)


print("\n-Restricted decision tree-")

print("Accuracy:", accuracy2)
print("Precision:", precision2)
print("recall:", recall2)
print("f1-score:", f12)

print("\n-Model Comparison-")

print("Normal decision tree accuracy:", accuracy)
print("Restricted decision tree accuracy:", accuracy2)

print("\nfeature Importance:")

features = X.columns
importance = model.feature_importances_

for name, value in zip(features, importance):
    print(name, ":", value)