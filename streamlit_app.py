import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("models/random_forest_wine_model.pkl")

# Streamlit UI
st.title("Wine Quality Predictor 🍷")
st.write("Enter the following characteristics of the wine:")

# Create inputs for all 11 features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

# Predict button
if st.button("Predict Quality"):
    # Create input data with all 11 features
    # Assuming 'quality' was wrongly included in training, we give a dummy 0 value here
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, pH, sulphates, alcohol, 0]])  # Dummy 'quality' value

    
    st.write("Input shape:", input_data.shape)  # Debugging help

    prediction = model.predict(input_data)[0]

    # Display result
    if prediction >= 7:
        st.success(f"The wine is of **GOOD** quality (Predicted Score: {prediction}) 🍾")
    elif prediction >= 5:
        st.warning(f"The wine is of **AVERAGE** quality (Predicted Score: {prediction}) 🍷")
    else:
        st.error(f"The wine is of **POOR** quality (Predicted Score: {prediction}) 🧃")
