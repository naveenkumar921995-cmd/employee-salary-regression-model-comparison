# ======================================
# IMPORT LIBRARIES
# ======================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AI Salary Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================================
# UI STYLE
# ======================================
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

[data-testid="metric-container"]{
background: rgba(255,255,255,0.08);
border-radius:12px;
padding:15px;
backdrop-filter: blur(10px);
}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    df = pd.read_csv("emp_sal.csv")
    return df

df = load_data()

# ======================================
# DATA CLEANING
# ======================================
df = df.dropna()

X = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values

# ======================================
# TRAIN MODELS
# ======================================
lin_model = LinearRegression()
lin_model.fit(X, y)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X, y)

# ======================================
# MODEL PERFORMANCE
# ======================================
lin_pred = lin_model.predict(X)
poly_pred = poly_model.predict(X_poly)
rf_pred = rf_model.predict(X)

lin_r2 = r2_score(y, lin_pred)
poly_r2 = r2_score(y, poly_pred)
rf_r2 = r2_score(y, rf_pred)

# ======================================
# SIDEBAR
# ======================================
st.sidebar.title("⚙️ Controls")

level = st.sidebar.slider(
    "Employee Level",
    float(df.iloc[:,0].min()),
    float(df.iloc[:,0].max()),
    float(df.iloc[:,0].mean())
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Auto Best Model","Linear Regression","Polynomial Regression","Random Forest"]
)

# ======================================
# AUTO MODEL SELECTION
# ======================================
scores = {
"Linear Regression":lin_r2,
"Polynomial Regression":poly_r2,
"Random Forest":rf_r2
}

if model_choice == "Auto Best Model":
    model_choice = max(scores, key=scores.get)

# ======================================
# PREDICTION
# ======================================
if model_choice == "Linear Regression":
    prediction = lin_model.predict([[level]])

elif model_choice == "Polynomial Regression":
    prediction = poly_model.predict(poly.transform([[level]]))

else:
    prediction = rf_model.predict([[level]])

# ======================================
# TITLE
# ======================================
st.title("🧠 AI Employee Salary Prediction Dashboard")

# ======================================
# KPI CARDS
# ======================================
c1,c2,c3,c4 = st.columns(4)

c1.metric("Linear R²", round(lin_r2,3))
c2.metric("Polynomial R²", round(poly_r2,3))
c3.metric("RandomForest R²", round(rf_r2,3))
c4.metric("Predicted Salary", f"${prediction[0]:,.0f}")

# ======================================
# MODEL LEADERBOARD
# ======================================
st.subheader("🏆 Model Accuracy Leaderboard")

leaderboard = pd.DataFrame({
"Model":scores.keys(),
"R2 Score":scores.values()
}).sort_values("R2 Score",ascending=False)

st.dataframe(leaderboard,use_container_width=True)

# ======================================
# HEATMAP
# ======================================
st.subheader("🔥 Model Performance Heatmap")

fig_heat = px.imshow(
leaderboard.set_index("Model"),
text_auto=True,
color_continuous_scale="viridis"
)

st.plotly_chart(fig_heat,use_container_width=True)

# ======================================
# 3D VISUALIZATION
# ======================================
st.subheader("📊 3D Salary Visualization")

fig3d = px.scatter_3d(
x=X.flatten(),
y=y,
z=lin_pred,
labels={
"x":"Level",
"y":"Actual Salary",
"z":"Predicted Salary"
}
)

st.plotly_chart(fig3d,use_container_width=True)

# ======================================
# TREND CHART
# ======================================
st.subheader("📈 Prediction Trend")

trend_df = pd.DataFrame({
"Level":X.flatten(),
"Actual Salary":y,
"Predicted Salary":poly_pred
})

fig_trend = px.line(
trend_df,
x="Level",
y=["Actual Salary","Predicted Salary"],
markers=True
)

st.plotly_chart(fig_trend,use_container_width=True)

# ======================================
# FEATURE IMPORTANCE
# ======================================
st.subheader("🎯 Feature Importance")

importance = pd.DataFrame({
"Feature":["Employee Level"],
"Importance":[rf_model.feature_importances_[0]]
})

fig_imp = px.bar(
importance,
x="Feature",
y="Importance"
)

st.plotly_chart(fig_imp,use_container_width=True)

# ======================================
# DOWNLOAD REPORT
# ======================================
st.subheader("📥 Download Prediction Report")

report = pd.DataFrame({
"Employee Level":[level],
"Model Used":[model_choice],
"Predicted Salary":[prediction[0]]
})

csv = report.to_csv(index=False)

st.download_button(
"Download CSV",
csv,
"salary_prediction_report.csv",
"csv"
)
