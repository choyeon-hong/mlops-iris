from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=["f1","f2","f3","f4"]
)

df["target"] = iris.target

train, new = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv("data/train.csv", index=False)
new.to_csv("data/new_data.csv", index=False)

print("dataset created")