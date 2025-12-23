import pandas as pd
from sklearn.model_selection import train_test_split
from models.random_forest_model import train_and_evaluate_random_forest
from models.arima_model import train_and_evaluate_arima
from models.mlp_model import train_and_evaluate_mlp
from models.lstm_model import train_and_evaluate_lstm
from models.transformer_model import train_and_evaluate_transformer

# Load data
data = pd.read_csv('pjm_processed.csv')

# Prepare features and target
target = 'PJME_MW'
features = [col for col in data.columns if col != target]
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Random Forest model
train_and_evaluate_random_forest('pjm_processed.csv')

# ARIMA model implementation
# Ensure data is sorted by time (assuming a 'Date' column exists)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Prepare time-series data
time_series = data[target]

# Split data into train and test
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Fit ARIMA model
arima_model = ARIMA(train, order=(5, 1, 0))  # Example order, adjust as needed
arima_model_fit = arima_model.fit()

# Predict and evaluate ARIMA model
arima_predictions = arima_model_fit.forecast(steps=len(test))
arima_mse = mean_squared_error(test, arima_predictions)

print(f"ARIMA - Mean Squared Error: {arima_mse}")
print("ARIMA Model Summary:")
print(arima_model_fit.summary())

# Example: Predict future values with ARIMA
# future_steps = 10
# future_predictions = arima_model_fit.forecast(steps=future_steps)
# print(f"Future Predictions: {future_predictions}")

# Run MLP model
train_and_evaluate_mlp('pjm_processed.csv')

# Run LSTM model
train_and_evaluate_lstm('pjm_processed.csv')

# Run Transformer model
train_and_evaluate_transformer('pjm_processed.csv')
