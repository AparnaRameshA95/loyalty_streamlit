import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained models and encoders
with open('robust_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('target_encoder.pkl', 'rb') as file:
    target_encoder = pickle.load(file)

with open('rfe_model.pkl', 'rb') as file:
    rfe_model = pickle.load(file)

with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Feature list for proper input mapping
selected_features = ['Age', 'Items Purchased', 'Total Spent', 'Discount (%)',
                     'Satisfaction Score', 'Revenue']

# Streamlit app
st.title("Loyalty Category Prediction")

# Input fields
age = st.number_input("Age", min_value=0, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
visit_time = st.selectbox("Preferred Visit Time", ["Morning", "Afternoon", "Evening"])
items_purchased = st.number_input("Items Purchased", min_value=0, step=1)
total_spent = st.number_input("Total Spent", min_value=0.0, step=0.01)
discount = st.number_input("Discount (%)", min_value=0.0, step=0.01)
revenue = st.number_input("Revenue", min_value=0.0, step=0.01)
product_category = st.selectbox(
    "Product Category", ["Accessories", "Laptop", "Tablet", "Television", "Mobile"]
)
payment_method = st.selectbox(
    "Payment Method", ["Cash", "Credit Card", "Debit Card", "UPI", "Net Banking"]
)
satisfaction_score = st.number_input(
    "Satisfaction Score", min_value=0, step=1
)

# Prediction logic
if st.button("Predict Loyalty Teir"):
    try:
        # Prepare input data
        data = {
            "Age": age,
            "Gender": gender,
            "Region": region,
            "Preferred Visit Time": visit_time,
            "Items Purchased": items_purchased,
            "Total Spent": total_spent,
            "Discount (%)": discount,
            "Revenue": revenue,
            "Product Category": product_category,
            "Payment Method": payment_method,
            "Satisfaction Score": satisfaction_score,
        }

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Scale numerical features
        input_data[['Total Spent', 'Discount (%)', 'Revenue']] = scaler.transform(
            input_data[['Total Spent', 'Discount (%)', 'Revenue']]
        )

        # One-hot encode categorical features
        categorical_data = input_data[['Gender', 'Region', 'Product Category', 'Preferred Visit Time', 'Payment Method']]
        encoded_categorical = encoder.transform(categorical_data)
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=encoder.get_feature_names_out(categorical_data.columns)
        )

        # Combine scaled and encoded features
        input_data = pd.concat([input_data.drop(columns=categorical_data.columns), encoded_categorical_df], axis=1)

        # Select RFE-selected features
        input_data = input_data[selected_features]

        # Make prediction
        prediction = best_model.predict(input_data)
        category = target_encoder.inverse_transform(prediction)[0]

        # Display result
        st.success(f"Loyalty Teir is: {category}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
