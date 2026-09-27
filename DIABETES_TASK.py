import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


data = pd.read_csv("diabetes.csv")

print("First 5 rows:")
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

print("\n column name:")
print(data.columns)


print("\n missing value:")
print(data.isnull().sum())


print("\n zero value:")
print((data == 0).sum())


X = data.drop("Outcome", axis=1)
y = data["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = LogisticRegression()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\n confusion matrix:")
print(cm)


print("\n classification report:")
print(classification_report(y_test, y_pred))


y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("roc-auc score:", roc_auc)

print("\n model coefficients:")

for name, value in zip(data.columns[:-1], model.coef_[0]):
    print(name, ":", value)