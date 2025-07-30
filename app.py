from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model, scaler, RFE selector, and all features used before RFE
rfe = joblib.load("rfe_selector.pkl")
all_features = joblib.load("all_features_before_rfe.pkl")  # complete feature list before RFE
scaler = joblib.load("heart_scaler.pkl")
model = joblib.load("heart_random_forest_model.pkl")

# Transform input to match the training pipeline
def transform_input(data):
    df = pd.DataFrame([data])

    # Derived features
    df['cholesterol_age_ratio'] = df['cholesterol'] / df['age']
    df['heart_stress'] = df['resting_bp'] / df['depression'].replace(0, 1)  # avoid division by zero

    # Age groups
    df['age_group_middle'] = ((df['age'] >= 40) & (df['age'] < 55)).astype(int)
    df['age_group_old'] = ((df['age'] >= 55) & (df['age'] < 65)).astype(int)
    df['age_group_very_old'] = (df['age'] >= 65).astype(int)

    # One-hot encoding for categorical fields
    encoded = pd.DataFrame()

    encoded['sex_male'] = [1 if df['sex'][0].lower() == 'male' else 0]

    # Define categorical columns and their expected one-hot values
    categories = {
        'chest_pain_type': ['B', 'C', 'D', 'E'],
        'fasting_blood_sugar': ['B', 'C', 'D', 'E'],
        'resting_ecg': ['B', 'C', 'D', 'E'],
        'exercise_angina': ['B', 'C', 'D', 'E'],
        'slope': ['B', 'C', 'D', 'E'],
        'major_vessels': ['B', 'C', 'D', 'E'],
        'thal': ['B', 'C', 'D', 'E']
    }

    for col, options in categories.items():
        val = df[col][0].upper()
        for opt in options:
            encoded[f"{col}_{opt}"] = [1 if val == opt else 0]

    # Add numeric and derived columns
    encoded['age'] = df['age']
    encoded['resting_bp'] = df['resting_bp']
    encoded['cholesterol'] = df['cholesterol']
    encoded['max_heart_rate'] = df['max_heart_rate']
    encoded['depression'] = df['depression']
    encoded['cholesterol_age_ratio'] = df['cholesterol_age_ratio']
    encoded['heart_stress'] = df['heart_stress']
    encoded['age_group_middle'] = df['age_group_middle']
    encoded['age_group_old'] = df['age_group_old']
    encoded['age_group_very_old'] = df['age_group_very_old']

    # Add missing columns with 0
    for col in all_features:
        if col not in encoded:
            encoded[col] = 0

    # Reorder the columns to match training order
    encoded = encoded.reindex(columns=all_features)

    # Scale and apply RFE
    encoded_scaled = scaler.transform(encoded)
    encoded_rfe = rfe.transform(encoded_scaled)

    return encoded_rfe

@app.route('/', methods=['GET'])
def home():
    return "Heart Disease Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        transformed = transform_input(user_input)

        pred = model.predict(transformed)[0]
        prob = model.predict_proba(transformed)[0][int(pred)]

        return jsonify({
            "prediction": int(pred),
            "probability": round(prob, 2),
            "result": "Positive for Heart Disease" if pred == 1 else "Negative for Heart Disease"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
