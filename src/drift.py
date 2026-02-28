import pandas as pd
import json

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/train.csv")
new = pd.read_csv("data/new_data.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=new)

result = report.as_dict()

drift = result["metrics"][0]["result"]["dataset_drift"]

print("drift:", drift)

if drift:
    with open("drift_flag.txt", "w") as f:
        f.write("drift")