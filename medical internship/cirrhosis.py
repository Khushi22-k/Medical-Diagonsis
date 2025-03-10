import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import pickle

from sklearn.impute import KNNImputer

imputer = SimpleImputer(strategy='mean')
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set(rc={'figure.figsize': [20, 20]}, font_scale=1.4)

df=pd.read_csv('cirrhosis.csv')

df=df.replace({'N':0,'Y':1,'M':0,'F':1})

del df['Drug']
df.isnull().sum()
df=df.replace({1:1,2:1,3:1,4:1})

df['Stage'].value_counts()
cirrhosis=df.groupby('Stage')
st.write(df)
cols = df.columns[df.dtypes.eq('object')]


df.isnull().sum()
df['Sex'].fillna(df['Sex'].mean(), inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Bilirubin'].fillna(df['Bilirubin'].mean(),inplace=True)
df['Albumin'].fillna(df['Albumin'].mean(),inplace=True)
df['Copper'].fillna(df['Copper'].mean(),inplace=True)
df['Alk_Phos'].fillna(df['Alk_Phos'].mean(),inplace=True)
df['SGOT'].fillna(df['SGOT'].mean(),inplace=True)
df['Tryglicerides'].fillna(df['Tryglicerides'].mean(),inplace=True)
df['Platelets'].fillna(df['Platelets'].mean(),inplace=True)
df['Cholesterol'].fillna(df['Cholesterol'].mean(),inplace=True)

st.write(df)
df['Age'] = imputer.fit_transform(df[['Age']])
df['Sex'] = imputer.fit_transform(df[['Sex']])
df['Bilirubin'] = imputer.fit_transform(df[['Bilirubin']])
df['Albumin'] = imputer.fit_transform(df[['Albumin']])
df['Copper'] = imputer.fit_transform(df[['Copper']])
df['Alk_Phos'] = imputer.fit_transform(df[['Alk_Phos']])
df['SGOT'] = imputer.fit_transform(df[['SGOT']])
df['Tryglicerides'] = imputer.fit_transform(df[['Tryglicerides']])
df['Platelets'] = imputer.fit_transform(df[['Platelets']])
df['Cholesterol'] = imputer.fit_transform(df[['Cholesterol']])
df.isnull().sum()
del df['Stage']

df.isnull().sum()

st.write(df.columns)
df.to_csv('prepocessed_cirrhosis.csv')
x = df.drop('Status', axis=1)
y = df['Status']
st.write(y.isna().sum())  
st.write(df.head())
st.write(x.head())
st.write(y.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
st.write(x.shape, x_train.shape, x_test.shape)
x_test=x_test.drop(['ID','N_Days','Edema','Ascites','Hepatomegaly','Spiders','Edema','Prothrombin'],axis=1)
x_train = x_train.drop(['ID','N_Days','Edema','Ascites','Hepatomegaly','Spiders','Edema','Prothrombin'],axis=1)
st.write(df)
model = LogisticRegression()
imputer = SimpleImputer(strategy='mean')
st.write(x_train.columns)

model.fit(x_train, y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
st.write('Accuracy on Training data : ', training_data_accuracy)
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
st.write('Accuracy on Test data : ', test_data_accuracy)
input_data = (44,0,0,45,1,1.4,39,10,20,15)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Cirrhosis Disease')
else:
  print('The Person has Cirrhosis Disease')

filename = 'Cirrhosis.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open('Models/Cirrhosis.sav', 'rb'))
for column in x_train.columns:
  print(column)

st.write(x_train.info())