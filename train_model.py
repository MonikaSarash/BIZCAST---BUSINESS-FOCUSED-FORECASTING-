import pandas as pd
from prophet import Prophet
import joblib

# Load dataset
df = pd.read_excel("sALES.xlsx")

# Rename columns for Prophet
df = df.rename(columns={"date": "ds", "sales": "y"})

# Convert date column
df["ds"] = pd.to_datetime(df["ds"])

print("Dataset loaded:", df.shape)

# Train Prophet model
model = Prophet()
model.fit(df)

# Save trained model
joblib.dump(model, "prophet_demand_model.pkl")

print("Model trained successfully and saved!")
