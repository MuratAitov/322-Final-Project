from flask import Flask, request, jsonify, render_template
from predict import train_model, make_prediction

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

        # Pass the input data to the make_prediction function
        prediction = make_prediction(input_data)

        # Return the prediction as a JSON response
        return jsonify({'prediction': 'Yes' if prediction == 1 else 'No'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Entry point for running the app
if __name__ == '__main__':
    # Train the model
    train_model()

    # Run the app on the local development server
    app.run(host='0.0.0.0', port=5000, debug=True)
