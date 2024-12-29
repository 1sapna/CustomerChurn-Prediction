import streamlit as st
import pickle
import pandas as pd

# Load the saved model and encoders
with open('customer_churn_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open("encoders.pkl", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)

# Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details below to predict the likelihood of churn:")

# Collect user inputs
gender = st.selectbox("Gender:", ['Male', 'Female'])
senior_citizen = st.selectbox("Senior Citizen:", [0, 1])
partner = st.selectbox("Partner:", ['Yes', 'No'])
dependents = st.selectbox("Dependents:", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months):", min_value=0, max_value=100, value=1)
phone_service = st.selectbox("Phone Service:", ['Yes', 'No'])
multiple_lines = st.selectbox("Multiple Lines:", ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox("Internet Service:", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security:", ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox("Online Backup:", ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox("Device Protection:", ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox("Tech Support:", ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV:", ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies:", ['Yes', 'No', 'No internet service'])
contract = st.selectbox("Contract:", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing:", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method:", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges:", min_value=0.0, max_value=1000.0, value=29.85)
total_charges = st.number_input("Total Charges:", min_value=0.0, value=29.85)

# Create input data dictionary
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert input data to DataFrame
input_data_df = pd.DataFrame([input_data])

# Encode categorical features using the saved encoders
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Make prediction
if st.button("Predict"):
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    # Display results
    if prediction[0] == 1:
        st.error(f"Prediction: Churn. Probability: {pred_prob[0][1]:.2f}")
    else:
        st.success(f"Prediction: No Churn. Probability: {pred_prob[0][0]:.2f}")
