





# import pickle
# import mlflow.pyfunc
# import streamlit as st
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Load the model
# model = mlflow.pyfunc.load_model("models:/linear1/1")

# # Load the one-hot encoding mapping
# with open("C:/Users/krebi/OneDrive/Desktop/stock predict/faang/company_mapping.pkl", "rb") as f:
#     company_mapping = pickle.load(f)

# # Load the trained scaler
# with open("C:/Users/krebi/OneDrive/Desktop/stock predict/faang/scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# st.write("# Stock Price Prediction")

# # Input fields
# company = st.selectbox("Select a company", ["Amazon", "Apple", "Facebook", "Google", "Netflix"])
# open_val = st.number_input("Open", format="%.8f")
# high_val = st.number_input("High", format="%.8f")
# low_val = st.number_input("Low", format="%.8f")
# volume_val = st.number_input("Volume", format="%.8f")



# # Create input dictionary
# input_dict = {"Open": open_val, "High": high_val, "Low": low_val, "Volume": volume_val}
# input_df = pd.DataFrame([input_dict])

# # Add one-hot encoded company columns
# for col in company_mapping.values():
#     input_df[col] = 0

# selected_column = company_mapping.get(company, None)
# if selected_column:
#     input_df[selected_column] = 1

# # Transform input data using the pre-fitted scaler
# scaled_features = scaler.transform(input_df)

# if st.button("Predict"):
#     try:
#         # Predict using the model
#         prediction = model.predict(scaled_features)
#         st.write(f"The predicted stock price is: {prediction[0]}")
#     except ValueError as e:
#         st.error(f"Prediction failed: {e}")


import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the log-transformed data
with open("log_transformed_scaler.pkl", "rb") as f:
    X_log_transformed, y_log_transformed = pickle.load(f)

# Company Encoding Mapping
company_mapping = {
    'Amazon': 'Company_Amazon',
    'Apple': 'Company_Apple',
    'Facebook': 'Company_Facebook',
    'Google': 'Company_Google',
    'Netflix': 'Company_Netflix'
}

# Train a Linear Regression model (for demonstration purposes)
# Replace this with your actual trained model if available
model = LinearRegression()
model.fit(X_log_transformed, y_log_transformed)

# Streamlit App
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction Dashboard")

st.sidebar.header("User Input Features")

# Input fields
company = st.sidebar.selectbox("Select a company", list(company_mapping.keys()))
open_val = st.sidebar.number_input("Open", format="%.8f")
high_val = st.sidebar.number_input("High", format="%.8f")
low_val = st.sidebar.number_input("Low", format="%.8f")
volume_val = st.sidebar.number_input("Volume", format="%.8f")

# Create input dictionary
input_dict = {"Open": open_val, "High": high_val, "Low": low_val, "Volume": volume_val}
input_df = pd.DataFrame([input_dict])

# Add one-hot encoded company columns
for col in company_mapping.values():
    input_df[col] = 0

selected_column = company_mapping.get(company, None)
if selected_column:
    input_df[selected_column] = 1

# Log-transform the input data
input_df_log_transformed = input_df.apply(np.log1p)

if st.sidebar.button("Predict"):
    try:
        # Predict using the model
        prediction_log = model.predict(input_df_log_transformed)
        prediction = np.expm1(prediction_log)  # Reverse log transformation
        st.success(f"🚀 The predicted stock price is: **${prediction[0]:.2f}**")
        
        # Display historical data visualization
        st.subheader("📊 Historical Stock Data (Log Transformed)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=X_log_transformed, x=X_log_transformed.index, y=y_log_transformed, ax=ax)
        ax.set_title(f"Historical Close Prices (Log Transformed) for {company}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Transformed Close Price")
        st.pyplot(fig)
        
        # Display prediction distribution
        st.subheader("📉 Prediction Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(y_log_transformed, kde=True, ax=ax)
        ax.axvline(prediction_log[0], color='r', linestyle='--', label=f'Predicted Price (Log): {prediction_log[0]:.2f}')
        ax.set_title("Distribution of Predicted Stock Prices (Log Transformed)")
        ax.set_xlabel("Log Transformed Stock Price")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Additional Information
st.sidebar.markdown("""
### About
This app predicts stock prices using a Linear Regression model trained on historical data from FAANG companies (Facebook, Apple, Amazon, Netflix, Google).
""")

st.sidebar.markdown("""
### Instructions
1. Select a company from the dropdown.
2. Enter the stock market features (Open, High, Low, Volume).
3. Click the **Predict** button to see the predicted stock price.
""")

st.sidebar.markdown("""
### Data Source
The data used for training the model is sourced from publicly available stock market data.
""")