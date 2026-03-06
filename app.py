# ================================
# IMPORT LIBRARIES
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Employee Salary AI Dashboard",
    page_icon="💼",
    layout="wide"
)

# ================================
# PREMIUM UI STYLE
# ================================
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#141E30,#243B55);
color:white;
}

[data-testid="metric-container"]{
background:rgba(255,255,255,0.1);
border-radius:12px;
padding:15px;
backdrop-filter: blur(10px);
}

[data-testid="stSidebar"]{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(10px);
}

</style>
""", unsafe_allow_html=True)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("emp_sal.csv")
    return df

df = load_data()

X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

# ================================
# TRAIN MODELS
# ================================
lin_model = LinearRegression()
lin_model.fit(X, y)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X, y)

# ================================
# MODEL PERFORMANCE
# ================================
lin_pred = lin_model.predict(X)
poly_pred = poly_model.predict(X_poly)
rf_pred = rf_model.predict(X)

lin_r2 = r2_score(y, lin_pred)
poly_r2 = r2_score(y, poly_pred)
rf_r2 = r2_score(y, rf_pred)

# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.title("⚙️ Controls")

position_level = st.sidebar.slider(
    "Employee Level",
    float(df.iloc[:,0].min()),
    float(df.iloc[:,0].max()),
    5.0
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression","Polynomial Regression","Random Forest","Auto Best Model"]
)

# ================================
# AUTO MODEL SELECTION
# ================================
if model_choice == "Auto Best Model":
    
    scores = {
        "Linear Regression": lin_r2,
        "Polynomial Regression": poly_r2,
        "Random Forest": rf_r2
    }
    
    best_model = max(scores, key=scores.get)
    
    model_choice = best_model

# ================================
# PREDICTION
# ================================
if model_choice == "Linear Regression":
    prediction = lin_model.predict([[position_level]])

elif model_choice == "Polynomial Regression":
    prediction = poly_model.predict(poly.transform([[position_level]]))

else:
    prediction = rf_model.predict([[position_level]])

# ================================
# DASHBOARD TITLE
# ================================
st.title("🧠 AI Employee Salary Prediction Dashboard")

# ================================
# KPI METRIC CARDS
# ================================
col1,col2,col3,col4 = st.columns(4)

col1.metric("Linear R²", round(lin_r2,3))
col2.metric("Polynomial R²", round(poly_r2,3))
col3.metric("RandomForest R²", round(rf_r2,3))
col4.metric("Predicted Salary", f"${prediction[0]:,.0f}")

# ================================
# MODEL RANKING TABLE
# ================================
st.subheader("🏆 Model Accuracy Leaderboard")

leaderboard = pd.DataFrame({
"Model":["Linear Regression","Polynomial Regression","Random Forest"],
"R2 Score":[lin_r2,poly_r2,rf_r2]
}).sort_values(by="R2 Score",ascending=False)

st.dataframe(leaderboard,use_container_width=True)

# ================================
# MODEL PERFORMANCE HEATMAP
# ================================
st.subheader("🔥 Model Performance Heatmap")

fig_heat = px.imshow(
    leaderboard.set_index("Model"),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="viridis"
)

st.plotly_chart(fig_heat,use_container_width=True)

# ================================
# 3D PLOTLY VISUALIZATION
# ================================
st.subheader("📊 3D Salary Prediction Visualization")

fig3d = px.scatter_3d(
    x=df.iloc[:,0],
    y=y,
    z=lin_pred,
    labels={'x':'Position Level','y':'Actual Salary','z':'Predicted Salary'},
)

st.plotly_chart(fig3d,use_container_width=True)

# ================================
# PREDICTION TREND ANIMATION
# ================================
st.subheader("📈 Prediction Trend Animation")

trend_df = pd.DataFrame({
"Position":X.flatten(),
"Actual Salary":y,
"Predicted Salary":poly_pred
})

fig_trend = px.line(
    trend_df,
    x="Position",
    y=["Actual Salary","Predicted Salary"],
    markers=True
)

st.plotly_chart(fig_trend,use_container_width=True)

# ================================
# FEATURE IMPORTANCE
# ================================
st.subheader("🎯 Feature Importance")

importance = pd.DataFrame({
"Feature":["Position Level"],
"Importance":[rf_model.feature_importances_[0]]
})

fig_imp = px.bar(
importance,
x="Feature",
y="Importance"
)

st.plotly_chart(fig_imp,use_container_width=True)

# ================================
# DOWNLOAD REPORT
# ================================
st.subheader("📥 Download Prediction Report")

report = pd.DataFrame({
"Selected Position":[position_level],
"Model Used":[model_choice],
"Predicted Salary":[prediction[0]]
})

csv = report.to_csv(index=False).encode('utf-8')

st.download_button(
label="Download Report",
data=csv,
file_name="salary_prediction_report.csv",
mime="text/csv"
)
