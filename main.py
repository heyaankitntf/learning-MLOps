from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="Demand Forecasting API")

# Load the model
model = joblib.load("demand_forecasting_pipeline.pkl")

TRAINING_START_DATE = pd.Timestamp('2022-01-01')


# 1. Define RAW inputs (What a user actually knows)
class DemandInput(BaseModel):
    Date: str               # e.g., "2023-10-25"
    Price: float
    Discount: float
    Competitor_Pricing: float
    Region: str
    Weather_Condition: str
    Category: str
    Epidemic: int
    Promotion: int


@app.post("/predict")
def predict_demand(data: DemandInput):
    
    # 2. Convert string date to a Pandas Timestamp
    current_date = pd.to_datetime(data.Date)
    
    # 3. Calculate the engineered features automatically!
    month = current_date.month
    day_of_week = current_date.dayofweek
    
    time_step = (current_date - TRAINING_START_DATE).days
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dayofweek_sin = np.sin(2 * np.pi * day_of_week / 7)
    dayofweek_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Calculate Price Gap (Handling division by zero just in case)
    price_gap = 0.0
    if data.Competitor_Pricing != 0:
        price_gap = ((data.Price - data.Competitor_Pricing) / data.Competitor_Pricing) * 100

    # 4. Build the dictionary EXACTLY how the model expects it
    input_dict = {
        'Price': data.Price,
        'Discount': data.Discount,
        'Competitor Pricing': data.Competitor_Pricing, # Note the space, not underscore!
        'Time_Step': time_step,
        'Price_Gap': price_gap,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'DayOfWeek_Sin': dayofweek_sin,
        'DayOfWeek_Cos': dayofweek_cos,
        'Region': data.Region,
        'Weather Condition': data.Weather_Condition, # Note the space!
        'Category': data.Category,
        'Epidemic': data.Epidemic,
        'Promotion': data.Promotion
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # 5. Predict
    prediction = model.predict(df)
    
    return {
        "predicted_demand": float(prediction[0])
    }