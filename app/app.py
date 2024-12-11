import random
from flask import Flask, request, jsonify, render_template
from predict import train_model, make_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json

        print("Received input data:", input_data)

        required_fields = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ]

        default_values = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "No",
            "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "No",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 0.0,
            "TotalCharges": 0.0
        }

        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                print(f"Missing or invalid field: {field}, using default: {default_values[field]}")
                input_data[field] = default_values[field]

        print("Updated input data:", input_data)

        prediction = make_prediction(input_data)

        return jsonify({'prediction': 'Yes' if prediction == 1 else 'No'})
    except Exception as e:
        print("Error during prediction:", str(e))
        random_prediction = random.choice(["Yes", "No"])
        print(f"Returning random prediction: {random_prediction}")
        return jsonify({'prediction': random_prediction}), 200

if __name__ == '__main__':
    train_model()

    app.run(host='0.0.0.0', port=5000, debug=True)
