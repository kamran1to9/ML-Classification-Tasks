from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print(data.data.shape)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test ,y_pred)
print("Accuracy is :", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix is :\n", cm)


print("\nPrecision, Recall and F1-Score:")
print(classification_report(y_test, y_pred))

y_probability = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probability)
print("Roc - auc score is :", roc_auc)