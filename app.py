import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler


# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="Under-Five Mortality Prediction",
    page_icon="/home/hdoop//U5MRIcon.png",
    layout="wide",
    #initial_sidebar_state="collapsed",  # Optional, collapses the sidebar by default
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">Under Five Mortality / Survival Prediction</h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining  header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #3498db;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'
# Load the XGBoost model
model_path = "C:/Users/Dawit Shibabaw/Desktop/Iris_stremalit - Copy/Breastcnacer_rf_model.sav"
loaded_model = joblib.load(open(model_path, "rb"))

# Load the label encoders
label_encoders_path = "C:/Users/Dawit Shibabaw/Desktop/Iris_stremalit - Copy/label_encoders.pkl"
label_encoders = pickle.load(open(label_encoders_path, "rb"))

# Load the MinMax scalers
scalers_path = "C:/Users/Dawit Shibabaw/Desktop/Iris_stremalit - Copy/minmax_scaler.pkl"
minmax_scalers = pickle.load(open(scalers_path, "rb"))

# Feature names and types
features = {
   'radius_mean': 'numerical', 
   'perimeter_mean': 'numerical', 
   'area_mean': 'numerical',
   'concave points_mean': 'categorical', 
   'fractal_dimension_mean': 'categorical', 
   'radius_se': 'categorical',
   'texture_se': 'categorical', 
   'area_se': 'numerical',
   'concavity_se': 'categorical', 
   'concave points_se': 'numerical',
   'texture_worst': 'numerical',
   'perimeter_worst': 'numerical',
   'area_worst': 'numerical', 
   'compactness_worst': 'categorical',  
}
# Sidebar title
st.sidebar.title("Input Parameters")
st.sidebar.markdown("""
[Example CSV input file](https://master/penguins_example.csv)
""")
# Create dictionary for grouping labels
group_labels = {
    'Geographic and Environmental': ['radius_mean', 'perimeter_mean', 'area_mean'],
    'Household Infrastructure': ['concave points_mean', 'fractal_dimension_mean', 'radius_se'],
    'Child-Related': ['texture_se', 'area_se', 'concavity_se', 'concave points_se'],
    'Parental and Reproductive': ['texture_worst', 'perimeter_worst', 'area_worst', 'compactness_worst']
}

# Option for CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# If CSV file is uploaded, read the file
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

# If CSV file is not uploaded, allow manual input
else:
    # Create empty dataframe to store input values
    input_df = pd.DataFrame(index=[0])

    # Loop through features and get user input
    # Loop through features and get user input
    for group, features_in_group in group_labels.items():
        st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
        st.sidebar.subheader(group)
        for feature in features_in_group:
            # Ensure each widget has a unique key
            widget_key = f"{group}_{feature}"

            # Display more descriptive labels
            if features[feature] == 'categorical':
                label = f"{feature.replace('_', ' ')}"
                input_df[feature] = st.sidebar.selectbox(label, label_encoders[feature].classes_, key=widget_key)
            else:
                label = f"{feature.replace('_', ' ')}"
                min_val, max_val = int(minmax_scalers[feature].data_min_[0]), int(minmax_scalers[feature].data_max_[0])
                default_val = (min_val + max_val) // 2

                # Check if the feature is of integer type
                is_integer = min_val == minmax_scalers[feature].data_min_[0] and max_val == \
                             minmax_scalers[feature].data_max_[0]

                # Use st.sidebar.slider with step 1 for integer features
                if is_integer:
                    input_df[feature] = st.sidebar.slider(label, min_value=min_val, max_value=max_val,
                                                          value=default_val, step=1, key=widget_key)
                else:
                    input_df[feature] = st.sidebar.slider(label, min_value=min_val, max_value=max_val,
                                                          value=default_val, key=widget_key)

# Display the input dataframe
st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature != 'child_alive':
            input_df[feature] = encoder.transform(input_df[feature])

    # Apply Min-max scaling to numerical features
    for feature, scaler in minmax_scalers.items():
        input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))

    # Make predictions
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)
    prediction = loaded_model.predict(input_df)

    # Display the prediction
    st.sidebar.write("Prediction:", prediction[0])

    # Apply model to make predictions
    prediction_proba = loaded_model.predict_proba(input_df)

    st.subheader('Prediction (Child will Survive?)')
    child_alive = np.array(['No', 'Yes'])
    st.write(child_alive[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)