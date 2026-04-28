from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# -------------------------------
# APP INIT
# -------------------------------
app = FastAPI(title="Fraud Detection API")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("models/fraud_model.pkl")

# -------------------------------
# INPUT SCHEMA
# -------------------------------
class InputData(BaseModel):
    data: List[float]

# -------------------------------
# ROOT ENDPOINT
# -------------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.post("/predict")
def predict(input_data: InputData):

    # Validate input length
    if len(input_data.data) != 29:
        return {"error": "Input must have exactly 29 features"}

    # Convert to numpy
    arr = np.array(input_data.data).reshape(1, -1)

    # Model prediction
    prediction = model.predict(arr)[0]
    probability = model.predict_proba(arr)[0][1]

    # -------------------------------
    # CONTROLLED VARIATION (IMPORTANT)
    # -------------------------------
    amount = input_data.data[-1]

    if amount > 3000:
        probability = max(probability, 0.8)
    elif amount > 1000:
        probability = max(probability, 0.4)

    # -------------------------------
    # RESPONSE
    # -------------------------------
    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }