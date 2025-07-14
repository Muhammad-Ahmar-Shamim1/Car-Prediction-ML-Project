import streamlit as st
import numpy as np
import pickle

# Load the model
import pickle

# Open the file in read-binary mode
with open("car_model.pkl", "rb") as file:
    model = pickle.load(file)


# UI
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Predictor")
st.markdown("### Enter details to predict car price")

# Inputs
model_year = st.number_input("Model Year", min_value=2000, max_value=2025, value=2015)
milage = st.number_input("Mileage (in km)", min_value=0, value=50000)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[model_year, milage]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Car Price: Rs -/{prediction[0]:,.2f}")
