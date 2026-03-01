import joblib
import numpy as np
from fastapi import FastAPI, Form

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def health():
    return {"status": "ok"}

# @app.post("/predict")
# def predict(data: list):

#     arr = np.array(data).reshape(1, -1)
#     pred = model.predict(arr)

#     return {"prediction": int(pred[0])}
@app.post("/predict")
def predict(data: str = Form(...)):

    data = list(map(float, data.split(",")))
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)

    return {"prediction": int(pred[0])}