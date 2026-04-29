import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #c0392b;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(231, 76, 60, 0.2);
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 style='text-align: center; color: #e74c3c; padding-top: 2rem;'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;'>A machine learning application to assess heart disease risk based on clinical parameters.</p>", unsafe_allow_html=True)

# Sections: Intro and Scope
with st.expander("ℹ️ Project Introduction & Scope", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌟 Project Introduction")
        st.markdown("""
        Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection and management are crucial in preventing severe complications. 
        This web application leverages **Machine Learning (Logistic Regression)** to predict the likelihood of a patient having heart disease based on various medical attributes.
        It uses a dataset containing patient metrics like age, blood pressure, cholesterol levels, and exercise-induced symptoms to make real-time predictions.
        """)
    with col2:
        st.markdown("### 🎯 Scope of the Project")
        st.markdown("""
        - **Interactive UI**: An intuitive interface for users and medical professionals to input patient data.
        - **Real-time Prediction**: Instantaneous risk assessment using a trained Logistic Regression model.
        - **Data Preprocessing**: Handles missing values and scales features using standard scaling techniques to ensure accurate predictions.
        - **Model Deployment**: Demonstrates the transition from a Jupyter Notebook analysis to a fully functional, deployable web application.
        """)

st.markdown("---")

# Load and prepare the model
@st.cache_resource
def load_data_and_train_model():
    try:
        # Load dataset
        df = pd.read_csv('heart.csv')
        
        # Preprocessing
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # Features and Target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        return model, scaler, accuracy, df.columns.drop('target')
    except Exception as e:
        return None, None, None, str(e)

model, scaler, accuracy, feature_names = load_data_and_train_model()

if model is None:
    st.error(f"Error loading data or training model: {feature_names}")
    st.stop()

st.sidebar.markdown("### 📊 Model Information")
st.sidebar.info(f"**Logistic Regression**\\n\\n**Test Accuracy:** {accuracy*100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Features Info")
st.sidebar.markdown("""
- **Age**: Age in years
- **Sex**: 1 = male; 0 = female
- **Chest Pain**: 0-3 scale
- **Resting BP**: in mm Hg
- **Cholesterol**: mg/dl
- **Fasting Blood Sugar**: > 120 mg/dl
- **Resting ECG**: 0-2 scale
- **Max Heart Rate**: achieved
- **Exercise Angina**: 1 = yes; 0 = no
- **Oldpeak**: ST depression
- **Slope**: ST segment
- **Major Vessels (CA)**: 0-4 scale
- **Thal**: Thalassemia status
""")

st.markdown("### 🩺 Patient Clinical Parameters")
st.markdown("Please enter the patient's medical details below to generate a prediction.")

# Form for user input
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        thal = st.selectbox("Thalassemia (Thal)", options=[0, 1, 2, 3], format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversable Defect"][x] if x in [0,1,2,3] else x)

    with col2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150)
        slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        
    with col3:
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ca = st.selectbox("Number of Major Vessels (CA)", options=[0, 1, 2, 3, 4], format_func=lambda x: str(x))

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🩺 Predict Heart Disease Risk")

if submit_button:
    # Prepare input data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=feature_names)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.markdown("### 📋 Risk Assessment Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease**")
            st.markdown("The model predicts that this patient is **likely** to have heart disease.")
        else:
            st.success("✅ **Low Risk of Heart Disease**")
            st.markdown("The model predicts that this patient is **unlikely** to have heart disease.")
            
    with res_col2:
        st.markdown(f"#### Probability of Heart Disease: **{probability[1]*100:.1f}%**")
        st.progress(float(probability[1]))
        if prediction == 1:
            st.markdown("<p style='color: #e74c3c; font-weight: bold;'>Recommendation: Please consult a cardiologist.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #27ae60; font-weight: bold;'>Recommendation: Maintain a healthy lifestyle.</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #bdc3c7;'>Disclaimer: This application is for educational purposes only and should not replace professional medical advice.</p>", unsafe_allow_html=True)
