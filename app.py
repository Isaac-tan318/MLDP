import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl') # Remove this line if you didn't use scaling

# 2. App Title and Description
st.title("Diabetes Risk Predictor")
st.write("Enter your health metrics below to assess diabetes risk.")

# 3. Sidebar for User Inputs
st.sidebar.header("User Input Features")

def user_input_features():
    # Numerical Inputs (Adjust ranges based on your real data)
    bmi = st.sidebar.slider('BMI', 10.0, 95.0, 25.0)
    age = st.sidebar.slider('Age Category (1-13)', 1, 13, 5)
    gen_hlth = st.sidebar.slider('General Health (1-5)', 1, 5, 3)
    ment_hlth = st.sidebar.slider('Days of Poor Mental Health (0-30)', 0, 30, 0)
    phys_hlth = st.sidebar.slider('Days of Poor Physical Health (0-30)', 0, 30, 0)
    
    # Binary Inputs (0 or 1)
    high_bp = st.sidebar.checkbox('High Blood Pressure?')
    high_chol = st.sidebar.checkbox('High Cholesterol?')
    smoker = st.sidebar.checkbox('Smoker?')
    stroke = st.sidebar.checkbox('History of Stroke?')
    heart_disease = st.sidebar.checkbox('Heart Disease?')
    phys_activity = st.sidebar.checkbox('Physically Active?')
    fruits = st.sidebar.checkbox('Eats Fruits Daily?')
    veggies = st.sidebar.checkbox('Eats Veggies Daily?')
    hvy_alcohol = st.sidebar.checkbox('Heavy Alcohol Consumer?')
    diff_walk = st.sidebar.checkbox('Difficulty Walking?')
    sex = st.sidebar.radio('Sex', ('Female', 'Male'))
    
    # Convert inputs to match model format (1 for True/Male, 0 for False/Female)
    data = {
        'HighBP': int(high_bp),
        'HighChol': int(high_chol),
        'CholCheck': 1, # Assuming they had a check to know the result
        'BMI': bmi,
        'Smoker': int(smoker),
        'Stroke': int(stroke),
        'HeartDiseaseorAttack': int(heart_disease),
        'PhysActivity': int(phys_activity),
        'Fruits': int(fruits),
        'Veggies': int(veggies),
        'HvyAlcoholConsump': int(hvy_alcohol),
        'AnyHealthcare': 1, # Default assumption or add input
        'NoDocbcCost': 0,   # Default assumption or add input
        'GenHlth': gen_hlth,
        'MentHlth': ment_hlth,
        'PhysHlth': phys_hlth,
        'DiffWalk': int(diff_walk),
        'Sex': 1 if sex == 'Male' else 0,
        'Age': age,
        'Education': 4, # Default or add slider
        'Income': 5     # Default or add slider
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 4. Display User Inputs
st.subheader('Patient Data')
st.write(input_df)

# 5. Make Prediction
if st.button('Predict Risk'):
    # Apply Scaling (Ensure columns match training order!)
    # input_df_scaled = scaler.transform(input_df) 
    # Use the line above if you scaled. If not, use input_df directly:
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1] # Probability of Class 1

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes (Probability: {probability:.2%})")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {probability:.2%})")