from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("heart_random_forest_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Heart Disease Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features in the correct order
        input_features = [data.get(feature) for feature in feature_names]

        # Check if any feature is missing
        if None in input_features:
            return jsonify({
                "error": "Missing one or more required features.",
                "required_features": feature_names
            }), 400

        # Scale the input
        input_scaled = scaler.transform([input_features])  # shape (1, n_features)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
