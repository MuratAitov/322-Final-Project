from flask import Flask, request, jsonify, render_template
from .predict import train_model, make_prediction

# Initialize the Flask app
app = Flask(__name__)

# Define the home endpoint
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        input_data = request.json

        # Debug: Print the input data to see what was received
        print("Received input data:", input_data)

        # Validate and fill missing or incorrect inputs
        required_fields = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ]

        # Default values for missing fields
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

        # Fill missing fields with default values
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                print(f"Missing or invalid field: {field}, using default: {default_values[field]}")
                input_data[field] = default_values[field]

        # Debug: Print the updated input data
        print("Updated input data:", input_data)

        # Pass the input data to the make_prediction function
        prediction = make_prediction(input_data)

        # Return the prediction as a JSON response
        return jsonify({'prediction': 'Yes' if prediction == 1 else 'No'})
    except Exception as e:
        # Debug: Print the error
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 400

# Entry point for running the app
if __name__ == '__main__':
    # Train the model
    train_model()

    # Run the app on the local development server
    app.run(host='0.0.0.0', port=5000, debug=True)
