import pandas as pd
import mlflow
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/train.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

mlflow.set_experiment("iris")

with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "model/model.pkl")

print("train done")