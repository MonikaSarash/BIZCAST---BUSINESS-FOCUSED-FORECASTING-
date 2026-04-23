from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("prophet_demand_model.pkl")

@app.route("/")
def home():
    return "Backend is running successfully"

@app.route("/predict_excel", methods=["POST"])
def predict_excel():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    df = pd.read_excel(file)

    # Column names MUST be date, sales
    df = df.rename(columns={"date": "ds", "sales": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    last_date = df["ds"].max()

    future = pd.DataFrame({
        "ds": pd.date_range(
            start=last_date + timedelta(days=1),
            periods=30
        )
    })

    forecast = model.predict(future)
    total_prediction = forecast["yhat"].sum()

    return jsonify({
        "next_month_total_product": float(total_prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
