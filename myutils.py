def categorical_distance(val1, val2):
    return 0 if val1 == val2 else 1

def encode_categorical_features(X):
    """Encodes categorical features in the dataset into numerical values."""
    encoded_X = []
    for row in X:
        encoded_row = [hash(value) if isinstance(value, str) else value for value in row]
        encoded_X.append(encoded_row)
    return encoded_X


def detect_and_remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile
    Q3 = df[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1                   # Interquartile range

    # Calculate the bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Outlier removal for {column}:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

    # Count outliers before removal
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Number of outliers in {column}: {len(outliers)}")

    # Filter the dataframe
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]