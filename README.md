# Telecom Customer Churn Prediction

## Project Overview
This project aims to build a machine learning model to predict customer churn in the telecom industry. Using the Telco Customer Churn dataset from Kaggle, the project involves data preprocessing, feature engineering, model training, and deploying a predictive web application using Flask.

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Project Structure](#project-structure)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Deployment](#deployment)
7. [Acknowledgements](#acknowledgements)

## Dataset Description
The dataset includes 7,043 records with 21 attributes, such as:
- **Gender**: Male/Female
- **SeniorCitizen**: Binary (0/1)
- **Tenure**: Number of months a customer has stayed
- **Contract**: Month-to-month, one year, two years
- **Churn**: Yes/No (Target Variable)

This dataset provides comprehensive details to help build predictive models for customer churn.

## Project Structure


## Technologies Used
- **Python**: Core programming language
- **Flask**: Web framework for deployment
- **Scikit-learn**: Machine learning
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **LabelEncoder/MinMaxScaler**: Data preprocessing
- **cURL**: Testing API endpoints

## Installation

### Prerequisites
1. Python 3.8+ installed on your system.
2. A terminal or command prompt.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/churn-prediction.git
   cd churn-prediction

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt


python app/app.py

curl -X POST -H "Content-Type: application/json" -d '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}' http://localhost:5000/predict


{
    "prediction": "Yes"
}

Deployment
The app has been deployed on Render (or other hosting service). You can access the deployed application using the following link:

Deployed Application

Deployment Steps
Create a free account on Render.
Link your GitHub repository to Render.
Add environment variables or configuration if required.
Specify the build command (e.g., pip install -r requirements.txt).
Specify the start command:
python app/app.py


Acknowledgements
Kaggle for the Telco Customer Churn dataset.
Scikit-learn Documentation for providing clear guidance on machine learning.
Render for free app hosting.


---

This `README.md` provides clear instructions for other developers to understand and use your project. Replace `https://your-app-link-here.com` with the link to your deployed app once you’ve hosted it.

Let me know if you need further assistance!

