# Stock Market Price Prediction

## Overview
This project focuses on predicting stock market closing prices using a **Linear Regression model**. The model is trained using historical stock data, applying **log transformation** to handle skewness and improve prediction accuracy. The experiment tracking and model logging are handled using **MLflow**.

## Data Preprocessing
1. **Handling Missing Values**: Checked and handled any missing data.
2. **Dropping Unwanted Categorical Columns**: Removed irrelevant categorical features.
3. **Exploratory Data Analysis (EDA)**: Analyzed distributions, correlations, and trends.
4. **Outlier Handling**:
   - Used **IQR (Interquartile Range)** and **Z-score** to detect outliers.
   - Clipped extreme outliers to ensure robust predictions.
5. **Feature Selection**: Chose only **numerical columns** for training.

## Features
- **Open**: Opening price of the stock.
- **High**: Highest price during the trading period.
- **Low**: Lowest price during the trading period.
- **Volume**: Number of shares traded.
- **Company Encoding**: One-hot encoding for companies (Amazon, Apple, etc.).
- **Close (Target Variable)**: Closing price of the stock.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn mlflow
```

## Usage
### **1. Training the Model**
Run the training script to preprocess data, train the model, and log metrics with MLflow:

```python
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow.sklearn

# Apply log transformation
X_train_log = X_train.apply(np.log1p)
X_test_log = X_test.apply(np.log1p)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

with mlflow.start_run(run_name="linear_regression_model"):
    model = LinearRegression()
    model.fit(X_train_log, y_train_log)
    y_pred_log = model.predict(X_test_log)
    y_pred = np.expm1(y_pred_log)  # Reverse log transformation
    y_test_actual = np.expm1(y_test_log)

    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2 Score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "linear_regression_model")
```

### **2. Making Predictions**
To predict stock prices using a trained model:

```python
with open("log_transformed_model.pkl", "rb") as f:
    model = pickle.load(f)

X_new_log = np.log1p(X_new)  # Apply log transformation
y_pred_log = model.predict(X_new_log)
y_pred = np.expm1(y_pred_log)  # Convert back to actual prices

print("Predicted stock price:", y_pred)
```

## Model Evaluation
- **MAE (Mean Absolute Error):** Measures average error in absolute terms.
- **RMSE (Root Mean Squared Error):** Measures error with higher penalties on large deviations.
- **R² Score:** Determines how well the model fits the data.

## MLflow Tracking
To view logged experiments, run:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Future Improvements
- Experimenting with **Random Forest and LightGBM**.
- Adding **technical indicators** (e.g., moving averages, RSI).
- Handling market volatility with **time-series models** (LSTM, ARIMA).

## License
This project is open-source under the **MIT License**.

## Author
- **GitHub**: [REBIN SCIENTIST](https://github.com/REBINFERNART)
- **Email**: krebinfernart@gmail.com
- **HUGGINGFACE:https://huggingface.co/spaces/REBIN007/fanng_stock_predict

---

🚀 **Happy Coding!** Let me know if you need modifications!

🏃
