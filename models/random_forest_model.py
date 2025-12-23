import sys
sys.path.append('.')  # Ensure current directory is in Python path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from csv_dataloader import CSVDataLoader

def train_and_evaluate_random_forest(data_path):
    # Load data
    loader = CSVDataLoader(data_path)
    data = next(iter(loader))

    print(data.columns)

    # Drop all unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Apply one-hot encoding to non-numeric columns before splitting the data
    data = pd.get_dummies(data, drop_first=True)

    # Prepare features and target after encoding
    target = 'PJME_MW'
    features = [col for col in data.columns if col != target]
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict and evaluate Random Forest model
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest - Mean Squared Error: {mse}")
    print(f"Random Forest - R^2 Score: {r2}")

# Example usage
if __name__ == "__main__":
    train_and_evaluate_random_forest('pjm_processed.csv')