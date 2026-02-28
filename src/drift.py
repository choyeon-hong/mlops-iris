import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/train.csv")
new = pd.read_csv("data/new_data.csv")

ref = ref.drop(columns=["target"])
new = new.drop(columns=["target"])

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=new)

result = report.as_dict()

dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
feature_metrics = result["metrics"][0]["result"]["drift_by_features"]

print("dataset drift:", dataset_drift)

drift_detected = False

for k, v in feature_metrics.items():
    print(k, v["drift_detected"])
    if v["drift_detected"]:
        drift_detected = True

if drift_detected:
    exit(1)