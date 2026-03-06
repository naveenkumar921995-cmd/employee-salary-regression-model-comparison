import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

dataset = pd.read_csv("data/emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

model = LinearRegression()
model.fit(X,y)

st.title("Employee Salary Prediction")

level = st.slider("Position Level",1.0,10.0,5.0)

prediction = model.predict([[level]])

st.write("Predicted Salary:", prediction[0])
