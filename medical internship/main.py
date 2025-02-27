import streamlit as st
# import numpy as np
import pandas as pd
# from sklearn import svm
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# from sklearn.prepocessing import StandardScaler
# from sklearn.model.selection import train_text_split 
selected_data=st.selectbox("Select a Disease to predict",["Diabetes Prediciton","Heart Disease Prediciton","Parkinsons Prediciton","Lung Cancer Prediciton","Thyroid Predicition"])
st.write(selected_data)
if selected_data=='Diabetes Prediciton':
    data=pd.read_csv('diabetes_data.csv')
    st.write(data)
elif selected_data=='Lung Cancer Prediciton':
    data=pd.read_csv('survey lung cancer.csv')
    st.write(data)
elif selected_data=='Heart Disease Prediciton':
    data=pd.read_csv('heart_disease_data.csv')
    st.write(data)
elif selected_data=='Parkinsons Prediciton':
    data=pd.read_csv('parkinson_data.csv')
    st.write(data)
elif selected_data=="Thyroid Predicition":
    data=pd.read_csv('hypothyroid.csv')
    st.write(data)