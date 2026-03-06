# ------------------------------------------------------------
# Employee Salary Regression Model Comparison Dashboard
# Author: Naveen Kumar
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------

st.set_page_config(page_title="Salary Prediction ML Dashboard", layout="wide")

st.title("Employee Salary Prediction Dashboard")
st.write("Comparison of Multiple Machine Learning Regression Models")

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

dataset = pd.read_csv("data/emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

st.subheader("Dataset Preview")
st.dataframe(dataset)

# ------------------------------------------------------------
# Train Models
# ------------------------------------------------------------

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

svr = SVR(kernel='rbf')
svr.fit(X, y)

knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(X, y)

dt = DecisionTreeRegressor(random_state=0)
dt.fit(X, y)

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X, y)

# ------------------------------------------------------------
# Salary Prediction Section
# ------------------------------------------------------------

st.header("Salary Prediction")

level = st.slider("Select Position Level", 1.0, 10.0, 5.0)

test_value = [[level]]

predictions = {
    "Linear Regression": lin_reg.predict(test_value)[0],
    "Polynomial Regression": poly_model.predict(poly.transform(test_value))[0],
    "SVR": svr.predict(test_value)[0],
    "KNN": knn.predict(test_value)[0],
    "Decision Tree": dt.predict(test_value)[0],
    "Random Forest": rf.predict(test_value)[0]
}

prediction_df = pd.DataFrame(
    predictions.items(),
    columns=["Model", "Predicted Salary"]
)

st.subheader("Prediction Comparison")
st.dataframe(prediction_df)

# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------

models = {
    "Linear Regression": lin_reg.predict(X),
    "Polynomial Regression": poly_model.predict(X_poly),
    "SVR": svr.predict(X),
    "KNN": knn.predict(X),
    "Decision Tree": dt.predict(X),
    "Random Forest": rf.predict(X)
}

evaluation = []

for name, pred in models.items():

    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    evaluation.append([name, r2, mae, rmse])

results = pd.DataFrame(
    evaluation,
    columns=["Model", "R2 Score", "MAE", "RMSE"]
)

st.subheader("Model Performance Metrics")
st.dataframe(results)

# ------------------------------------------------------------
# Model Ranking Dashboard
# ------------------------------------------------------------

st.header("Model Ranking Dashboard")

ranking = results.sort_values("R2 Score", ascending=False).reset_index(drop=True)
ranking.index = ranking.index + 1

st.dataframe(ranking)

# ------------------------------------------------------------
# Linear Regression Chart
# ------------------------------------------------------------

st.subheader("Linear Regression Visualization")

fig1, ax1 = plt.subplots()

ax1.scatter(X, y)
ax1.plot(X, lin_reg.predict(X))

ax1.set_title("Linear Regression")
ax1.set_xlabel("Position Level")
ax1.set_ylabel("Salary")

st.pyplot(fig1)

# ------------------------------------------------------------
# Polynomial Regression Chart
# ------------------------------------------------------------

st.subheader("Polynomial Regression Visualization")

X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape(-1, 1)

fig2, ax2 = plt.subplots()

ax2.scatter(X, y)
ax2.plot(X_grid, poly_model.predict(poly.transform(X_grid)))

ax2.set_title("Polynomial Regression")
ax2.set_xlabel("Position Level")
ax2.set_ylabel("Salary")

st.pyplot(fig2)

# ------------------------------------------------------------
# Advanced Comparison Charts
# ------------------------------------------------------------

st.header("Advanced Model Comparison Charts")

# R2 Score Chart
fig3, ax3 = plt.subplots()

ax3.bar(results["Model"], results["R2 Score"])
ax3.set_title("R² Score Comparison")
ax3.set_xlabel("Regression Models")
ax3.set_ylabel("R² Score")

plt.xticks(rotation=45)

st.pyplot(fig3)

# MAE Chart
fig4, ax4 = plt.subplots()

ax4.bar(results["Model"], results["MAE"])
ax4.set_title("Mean Absolute Error Comparison")
ax4.set_xlabel("Regression Models")
ax4.set_ylabel("MAE")

plt.xticks(rotation=45)

st.pyplot(fig4)

# RMSE Chart
fig5, ax5 = plt.subplots()

ax5.bar(results["Model"], results["RMSE"])
ax5.set_title("Root Mean Squared Error Comparison")
ax5.set_xlabel("Regression Models")
ax5.set_ylabel("RMSE")

plt.xticks(rotation=45)

st.pyplot(fig5)

# ------------------------------------------------------------
# Best Model Highlight
# ------------------------------------------------------------

best_model = ranking.iloc[0]

st.success(
    f"Best Model: {best_model['Model']} with R² Score = {best_model['R2 Score']:.3f}"
)
