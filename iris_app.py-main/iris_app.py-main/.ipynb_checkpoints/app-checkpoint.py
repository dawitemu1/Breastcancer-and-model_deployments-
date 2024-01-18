import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Classifying Iris Flowers using Streamlit')
st.markdown('Toy model to play to classify iris flowers into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

species = ''
if st.button("Predict type of Iris"):
    result = predict(
         np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
     
    if (result[0]== 0):
         species = 'Iris-setosa'
    elif (result[1] == 1):
          species = 'Iris-versicolor'
    else:
          species = 'Iris-virginica'
        
    st.success(species)
st.markdown(
    '`Create by` [Dawit Shibabaw](https://www.linkedin.com/in/dawit-shibabaw-3a0a98190/) | \
         `Code:` [GitHub](https://github.com/dawitemu1/iris_app.py/edit/main/app.py)')
