from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model, scaler, and feature names
with open("heart_random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Expected input features
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        required_raw_fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                               "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        for field in required_raw_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "required_fields": required_raw_fields
                }), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Rename for consistency
        input_df.rename(columns={
            "trestbps": "resting_bp",
            "chol": "cholesterol",
            "thalach": "max_heart_rate",
            "oldpeak": "depression"
        }, inplace=True)

        # Feature Engineering
        input_df["cholesterol_age_ratio"] = input_df["cholesterol"] / input_df["age"]
        input_df["heart_stress"] = input_df["max_heart_rate"] - input_df["resting_bp"]

        age = input_df["age"].values[0]
        if age < 40:
            age_group = "young"
        elif age < 55:
            age_group = "middle"
        elif age < 65:
            age_group = "old"
        else:
            age_group = "very_old"
        input_df[f"age_group_{age_group}"] = 1

        # One-hot encoding
        categorical_features = {
            "sex": {1: "male", 0: "female"},
            "cp": {0: "B", 1: "C", 2: "D", 3: "E"},
            "fbs": {1: "A", 0: "B"},
            "restecg": {0: "B", 1: "C", 2: "D"},
            "exang": {1: "A", 0: "B"},
            "slope": {0: "B", 1: "C", 2: "D"},
            "ca": {0: "B", 1: "C", 2: "D", 3: "E"},
            "thal": {0: "B", 1: "C", 2: "D", 3: "E"}
        }

        for feature, mapping in categorical_features.items():
            value = data[feature]
            mapped = mapping.get(value)
            if mapped:
                input_df[f"{feature}_{mapped}"] = 1

        # Add missing columns to match training
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure correct order of columns
        input_df = input_df[feature_names]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][int(prediction)]

        result_text = "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"
        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability, 2),
            "result": result_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return "Heart Disease Prediction API is live!"


if __name__ == '__main__':
    app.run(debug=True)
