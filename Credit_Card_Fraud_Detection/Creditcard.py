import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

df = pd.read_csv("creditcard.csv")

print(df.head())

normal = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

print("Normal Transactions :", len(normal))
print("Fraud Transactions  :", len(fraud))

normal_sample = resample(
    normal,
    replace=False,
    n_samples=len(fraud),
    random_state=42
)

balanced_df = pd.concat([normal_sample, fraud])

X = balanced_df.drop("Class", axis=1)
y = balanced_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))