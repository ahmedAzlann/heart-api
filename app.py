from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("heart_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Heart Disease Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Receieved data:", data)

        # Ensure all expected features are present
        input_features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            input_features.append(data[feature])

        # Scale input
        scaled_input = scaler.transform([input_features])

        # Predict
        prediction = model.predict(scaled_input)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        print("🔴 Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
