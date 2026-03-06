# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Salary ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Salary Prediction & Model Comparison")

st.write(
"""
This dashboard compares **multiple regression algorithms**
to determine the best model for predicting employee salary.
"""
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("emp_sal.csv")

df = load_data()

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df)

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X,y)

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly,y)

svr_reg = SVR(kernel="rbf")
svr_reg.fit(X,y)

knn_reg = KNeighborsRegressor(n_neighbors=3,weights="distance")
knn_reg.fit(X,y)

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X,y)

rf_reg = RandomForestRegressor(n_estimators=100,random_state=0)
rf_reg.fit(X,y)

models = {
    "Linear Regression": lin_reg,
    "Polynomial Regression": poly_reg,
    "SVR": svr_reg,
    "KNN": knn_reg,
    "Decision Tree": dt_reg,
    "Random Forest": rf_reg
}

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------
st.sidebar.header("⚙️ Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

level = st.sidebar.slider(
    "Select Position Level",
    1.0,
    10.0,
    5.0
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if selected_model == "Polynomial Regression":
    prediction = poly_reg.predict(poly.transform([[level]]))[0]
else:
    prediction = models[selected_model].predict([[level]])[0]

st.subheader("💰 Salary Prediction")

st.success(
    f"Predicted Salary using **{selected_model}**: ₹ {prediction:,.2f}"
)

# ---------------------------------------------------
# MODEL LEADERBOARD
# ---------------------------------------------------
st.subheader("🏆 Model Accuracy Leaderboard")

results = []

for name, model in models.items():

    if name == "Polynomial Regression":
        preds = poly_reg.predict(poly.transform(X))
    else:
        preds = model.predict(X)

    score = r2_score(y, preds)

    results.append({
        "Model": name,
        "R2 Score": score
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

results_df.index = results_df.index + 1

st.dataframe(results_df)

# ---------------------------------------------------
# LEADERBOARD CHART
# ---------------------------------------------------
fig = px.bar(
    results_df,
    x="Model",
    y="R2 Score",
    color="Model",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------
# REGRESSION CURVE
# ---------------------------------------------------
st.subheader("📈 Regression Curve")

X_grid = np.arange(start=X.min(),stop=X.max(),step=0.1)
X_grid = X_grid.reshape(-1,1)

if selected_model == "Polynomial Regression":
    y_pred = poly_reg.predict(poly.transform(X_grid))
else:
    y_pred = models[selected_model].predict(X_grid)

plot_df = pd.DataFrame({
    "Position Level":X_grid.flatten(),
    "Predicted Salary":y_pred
})

fig2 = px.line(
    plot_df,
    x="Position Level",
    y="Predicted Salary",
    title=f"{selected_model} Regression Curve"
)

fig2.add_scatter(
    x=X.flatten(),
    y=y,
    mode="markers",
    name="Actual Data"
)

st.plotly_chart(fig2,use_container_width=True)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
st.subheader("📊 Feature Importance")

importance = pd.DataFrame({
    "Feature":["Position Level"],
    "Importance":[1.0]
})

fig3 = px.bar(
    importance,
    x="Feature",
    y="Importance",
    title="Feature Importance"
)

st.plotly_chart(fig3,use_container_width=True)

# ---------------------------------------------------
# ANIMATED VISUALIZATION
# ---------------------------------------------------
st.subheader("🎬 Animated Prediction Visualization")

anim_df = pd.DataFrame({
    "Position Level":X.flatten(),
    "Salary":y
})

fig4 = px.scatter(
    anim_df,
    x="Position Level",
    y="Salary",
    size="Salary",
    animation_frame="Position Level",
    title="Salary Growth by Position Level"
)

st.plotly_chart(fig4,use_container_width=True)

# ---------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------
st.subheader("📥 Download Model Report")

csv = results_df.to_csv(index=False)

st.download_button(
    "Download CSV Report",
    csv,
    "model_report.csv",
    "text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Python • Streamlit • Scikit-Learn • Plotly")
