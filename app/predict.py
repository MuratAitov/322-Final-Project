import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
table = MyPyTable()
table.load_from_file(data_path)

categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

knn_model = None
encoders = None
min_max_scalers = None


def encode_categorical(table, columns):
    encoders = {}
    for col in columns:
        column_data = table.get_column(col)
        unique_values = list(set(column_data))
        encoders[col] = {value: idx for idx, value in enumerate(unique_values)}
        table.update_column(col, [encoders[col][value] for value in column_data])
    return encoders


def normalize_numerical(table, columns):
    min_max_values = {}
    for col in columns:
        column_data = table.get_column(col)

        numeric_data = []
        for value in column_data:
            try:
                numeric_data.append(float(value))
            except ValueError:
                numeric_data.append(0.0)

        min_val, max_val = min(numeric_data), max(numeric_data)
        min_max_values[col] = (min_val, max_val)

        normalized_values = [
            (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            for value in numeric_data
        ]
        table.update_column(col, normalized_values)
    return min_max_values

def preprocess_data(table):
    encoders = encode_categorical(table, categorical_columns)
    min_max_scalers = normalize_numerical(table, numeric_columns)
    return encoders, min_max_scalers


# Train the KNN model
def train_model():
    global knn_model, encoders, min_max_scalers

    processed_table = MyPyTable(column_names=table.column_names, data=[row[:] for row in table.data])
    encoders, min_max_scalers = preprocess_data(processed_table)

    # Remove the "Churn" and "customerID" columns
    X_table = processed_table.drop_columns(["Churn", "customerID"])
    X = X_table.data
    y = [1 if label == "Yes" else 0 for label in table.get_column("Churn")]

    knn_model = MyKNeighborsClassifier(n_neighbors=3)
    knn_model.feature_names = X_table.column_names
    knn_model.fit(X, y)


def make_prediction(input_data):
    global knn_model, encoders, min_max_scalers

    for col, encoder in encoders.items():
        input_data[col] = encoder[input_data[col]]

    for col, (min_val, max_val) in min_max_scalers.items():
        input_data[col] = (input_data[col] - min_val) / (max_val - min_val)

    input_row = [input_data[col] for col in knn_model.feature_names]
    return knn_model.predict([input_row])[0]


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
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    prediction = make_prediction(test_input)
    print(f"Prediction for test input: {'Yes' if prediction == 1 else 'No'}")

    categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
    ]

    # Get unique values for each categorical column
    unique_values = {col: set(table.get_column(col)) for col in categorical_columns}
    print(unique_values)
