import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyNaiveBayesClassifier

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
table = MyPyTable()
table.load_from_file(data_path)

categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "Tenure_c", "MonthlyCharges_c", "TotalCharges_c"
]

# Functions for discretization
def categorize_tenure(value):
    if value <= 12:
        return "New"
    elif value <= 24:
        return "Short-term"
    elif value <= 48:
        return "Mid-term"
    else:
        return "Long-term"

def categorize_monthly_charges(value):
    if value <= 40:
        return "Low"
    elif value <= 80:
        return "Medium"
    else:
        return "High"

def categorize_total_charges(value):
    try:
        value = float(value)
        if value <= 2000:
            return "Low"
        elif value <= 5000:
            return "Medium"
        else:
            return "High"
    except ValueError:
        return "Unknown"  # Or handle as appropriate for your model

def discretize_data(table):
    tenure_data = table.get_column("tenure", include_missing_values=False)
    monthly_charges_data = table.get_column("MonthlyCharges", include_missing_values=False)
    total_charges_data = table.get_column("TotalCharges", include_missing_values=False)

    tenure_c = [categorize_tenure(float(value)) for value in tenure_data]
    monthly_charges_c = [categorize_monthly_charges(float(value)) for value in monthly_charges_data]
    total_charges_c = [categorize_total_charges(value) for value in total_charges_data]


    table.column_names.extend(["Tenure_c", "MonthlyCharges_c", "TotalCharges_c"])
    for i, row in enumerate(table.data):
        row.extend([tenure_c[i], monthly_charges_c[i], total_charges_c[i]])

# Encode categorical data
def encode_categorical(table, columns):
    encoders = {}
    for col in columns:
        column_data = table.get_column(col)
        unique_values = list(set(column_data))
        encoders[col] = {value: idx for idx, value in enumerate(unique_values)}
        table.update_column(col, [encoders[col][value] for value in column_data])
    return encoders

naive_bayes_model = None
encoders = None

def preprocess_data(table):
    discretize_data(table)
    encoders = encode_categorical(table, categorical_columns)
    return encoders
def train_model():
    global naive_bayes_model, encoders, feature_names

    # Preprocess the data
    processed_table = MyPyTable(column_names=table.column_names, data=[row[:] for row in table.data])
    encoders = preprocess_data(processed_table)

    # Drop irrelevant columns
    X_table = processed_table.drop_columns(["Churn", "customerID", "tenure", "MonthlyCharges", "TotalCharges"])
    X = X_table.data
    y = [1 if label == "Yes" else 0 for label in table.get_column("Churn")]

    # Store feature names
    feature_names = X_table.column_names

    # Train the Naive Bayes classifier
    naive_bayes_model = MyNaiveBayesClassifier()
    naive_bayes_model.fit(X, y)

def make_prediction(input_data):
    global naive_bayes_model, encoders, feature_names

    # Apply discretization
    input_data["Tenure_c"] = categorize_tenure(input_data["tenure"])
    input_data["MonthlyCharges_c"] = categorize_monthly_charges(input_data["MonthlyCharges"])
    input_data["TotalCharges_c"] = categorize_total_charges(input_data["TotalCharges"])

    # Encode input data
    for col, encoder in encoders.items():
        input_data[col] = encoder[input_data[col]]

    # Prepare the input row based on feature names
    input_row = [input_data[col] for col in feature_names]
    return naive_bayes_model.predict([input_row])[0]

if __name__ == "__main__":
    train_model()

    test_input = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    prediction = make_prediction(test_input)
    print(f"Prediction for test input: {'Yes' if prediction == 1 else 'No'}")