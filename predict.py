import sys
import os

# Add the path to the mysklearn folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mysklearn")))

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from mysklearn.myclassifiers import MyKNeighborsClassifier

# Load the data
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

label_encoders = {}
scaler = None
model = None

def preprocess_data(data):
    global label_encoders, scaler

    categorical_columns = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    for col in categorical_columns:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
        else:
            data[col] = label_encoders[col].transform(data[col])

    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    if not scaler:
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    else:
        data[numeric_columns] = scaler.transform(data[numeric_columns])

    return data

def train_model():
    global model
    df = preprocess_data(data.copy())
    X = df.drop(columns=["Churn", "customerID"])
    y = LabelEncoder().fit_transform(df["Churn"])

    model = MyKNeighborsClassifier(n_neighbors=3)
    model.fit(X.values.tolist(), y)

def make_prediction(input_data):
    input_df = preprocess_data(pd.DataFrame([input_data]))
    return model.predict(input_df.values.tolist())[0]
