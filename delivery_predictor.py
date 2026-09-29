import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
# Ensure the model file is in the same directory as this script or provide the full path
model = joblib.load('logistic_regression_model.sav')

# Define the feature columns used during training (X_train.columns from your notebook)
# This list must exactly match the columns used to train the model, including the order
# You might need to manually copy this from your notebook's X_train.columns.tolist()
feature_columns = ['Cost_of_the_Product', 'Prior_purchases', 'Discount_offered', 'Weight_in_gms',
                   'Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F',
                   'Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship',
                   'Product_importance_low', 'Product_importance_medium', 'Gender_M']

st.set_page_config(page_title="Delivery On-Time Predictor", layout="centered")
st.title("🚚 Delivery On-Time Predictor")
st.markdown("Enter the details of the delivery to predict if it will arrive on time.")

# Input fields for user
st.header("Delivery Details")

c_cost = st.number_input("Cost of the Product", min_value=50, max_value=500, value=200)
c_prior = st.number_input("Number of Prior Purchases", min_value=0, max_value=10, value=3)
c_discount = st.number_input("Discount Offered", min_value=0, max_value=70, value=10)
c_weight = st.number_input("Weight in Grams", min_value=100, max_value=8000, value=3000)

warehouse_block_options = ['A', 'B', 'C', 'D', 'F']
c_warehouse = st.selectbox("Warehouse Block", warehouse_block_options)

shipment_mode_options = ['Flight', 'Road', 'Ship']
c_shipment = st.selectbox("Mode of Shipment", shipment_mode_options)

product_importance_options = ['low', 'medium', 'high']
c_product_importance = st.selectbox("Product Importance", product_importance_options)

c_gender = st.radio("Gender of Customer", ['M', 'F'])

if st.button("Predict Delivery Status"):    
    # Create a DataFrame from the inputs, matching the original dataframe structure for preprocessing
    input_df = pd.DataFrame([{
        'Cost_of_the_Product': c_cost,
        'Prior_purchases': c_prior,
        'Discount_offered': c_discount,
        'Weight_in_gms': c_weight,
        'Warehouse_block': c_warehouse,
        'Mode_of_Shipment': c_shipment,
        'Product_importance': c_product_importance,
        'Gender': c_gender
    }])

    # Apply one-hot encoding, matching training process
    categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    input_df_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure all feature columns from training are present and in the correct order
    final_input = pd.DataFrame(columns=feature_columns) # Create an empty DF with all expected columns
    for col in feature_columns:
        if col in input_df_processed.columns:
            final_input[col] = input_df_processed[col]
        else:
            final_input[col] = 0 # Add missing columns as 0 (for one-hot encoded cols not present)

    # Make prediction
    prediction = model.predict(final_input)
    prediction_proba = model.predict_proba(final_input)[:, 1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"The delivery is predicted to be **On Time**! (Probability: {prediction_proba[0]:.2f})")
    else:
        st.error(f"The delivery is predicted to be **Not On Time**. (Probability: {prediction_proba[0]:.2f})")

    st.markdown("--- ")
    st.info("This prediction is based on a Logistic Regression model trained on historical delivery data.")
