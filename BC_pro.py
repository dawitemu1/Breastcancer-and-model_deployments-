import joblib
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
# Load the Breast Cancer model
Breastcancer_model = joblib.load(open('C:/Users/Daveee/Desktop/Python apps/Iris_stremalit - Copy/Breastcnacer_rf_model.sav', 'rb'))
# Load the label encoder
with open("C:/Users/Daveee/Desktop/Python apps/Iris_stremalit - Copy/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Prediction System',
                           ['Breast_cancer predciction'],
                           icons=['person'],
                           default_index=0)
    
# Breast_cancer Prediction Page
if selected == 'Breast_cancer predciction':
    # page title
    st.title('Breast Cancer Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Radius_Mean')

    with col2:
        Glucose = st.text_input('Perimeter_Mean')

    with col3:
        BloodPressure = st.text_input('Area_Mean')

    with col1:
        SkinThickness = st.selectbox("Concave Points_Mean", ["0", "1"])

    with col2:
        Insulin = st.selectbox("Fractal_Dimension_Mean", ["0", "1"])

    with col3:
        BMI= st.slider('Radius_See', 0, 2)

    with col1:
        DiabetesPedigreeFunction = st.text_input('Texture_Se')

    with col2:
        Age = st.text_input('Area_Se')

    with col3:
        concavity_se = st.selectbox("Concavity_Se", ["0", "1"])

    with col1:
        concavepoints_se = st.selectbox("Concave Points_Se", ["0", "1"])

    with col2:
        texture_worst = st.text_input('Texture_Worst')

    with col3:
        perimeter_worst = st.text_input('Perimeter_Worst')
    with col1:
        area_worst = st.text_input('Area_Worst')

    with col2:
        compactness_worst = st.selectbox("Compactness_Worst", ["0", "1"])

    # code for Prediction
    diab_diagnosis = ''
    prediction_proba = ''

    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        diab_prediction = Breastcancer_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, concavity_se, concavepoints_se, texture_worst, perimeter_worst, area_worst, compactness_worst]])

        # Predict probabilities for each class
        prediction_proba = Breastcancer_model.predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, concavity_se, concavepoints_se, texture_worst, perimeter_worst, area_worst, compactness_worst]])
        
        # Display prediction result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is Malignant'
        else:
            diab_diagnosis = 'The person is Benign'
    
    st.success(diab_diagnosis)

    # Display probability for each class
    if isinstance(prediction_proba, np.ndarray) and prediction_proba.size > 0:
        st.write('Probability for Malignant:', prediction_proba[0][1])
        st.write('Probability for Benign:', prediction_proba[0][0])

        # Visualization of probability
        probabilities = prediction_proba[0]
        classes = ['Benign', 'Malignant']

        fig, ax = plt.subplots()
        ax.bar(classes, probabilities)
        ax.set_ylabel('Probability')
        ax.set_title('Probability Distribution')
        st.pyplot(fig)
st.markdown(
    '`Create by` [Dawit Shibabaw](https://www.linkedin.com/in/dawit-shibabaw-3a0a98190/) | \
         `Code:` [GitHub](https://github.com/dawitemu1)')