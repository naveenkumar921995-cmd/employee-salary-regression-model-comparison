# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Salary ML Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# GLASSMORPHISM UI
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#1f1c2c,#928dab);
}

.metric-card {
background: rgba(255,255,255,0.15);
border-radius: 15px;
padding: 20px;
backdrop-filter: blur(10px);
box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📊 Employee Salary Prediction & Model Comparison Dashboard")

st.write(
"""
Compare multiple **Machine Learning Regression Models**
and automatically detect the **best performing algorithm**.
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
# MODEL EVALUATION
# ---------------------------------------------------
results = []

for name,model in models.items():

    if name=="Polynomial Regression":
        preds = poly_reg.predict(poly.transform(X))
    else:
        preds = model.predict(X)

    r2 = r2_score(y,preds)
    mae = mean_absolute_error(y,preds)

    results.append({
        "Model":name,
        "R2 Score":r2,
        "MAE":mae
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------
# AUTO SELECT BEST MODEL
# ---------------------------------------------------
best_model_name = results_df.sort_values(
    by="R2 Score",
    ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------
st.sidebar.header("⚙️ Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    index=list(models.keys()).index(best_model_name)
)

level = st.sidebar.slider(
    "Position Level",
    1.0,
    10.0,
    5.0
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if selected_model=="Polynomial Regression":
    prediction = poly_reg.predict(poly.transform([[level]]))[0]
else:
    prediction = models[selected_model].predict([[level]])[0]

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------
col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Best Model",best_model_name)

with col2:
    st.metric(
        "Best R² Score",
        round(results_df["R2 Score"].max(),3)
    )

with col3:
    st.metric(
        "Predicted Salary",
        f"₹ {prediction:,.0f}"
    )

# ---------------------------------------------------
# MODEL LEADERBOARD
# ---------------------------------------------------
st.subheader("🏆 Model Leaderboard")

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

results_df.index = results_df.index+1

st.dataframe(results_df)

# ---------------------------------------------------
# MODEL PERFORMANCE CHART
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
# PERFORMANCE HEATMAP
# ---------------------------------------------------
st.subheader("🔥 Model Performance Heatmap")

heatmap_data = results_df.set_index("Model")

fig_heat = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    title="Model Performance Heatmap"
)

st.plotly_chart(fig_heat,use_container_width=True)

# ---------------------------------------------------
# REGRESSION CURVE
# ---------------------------------------------------
st.subheader("📈 Regression Curve")

X_grid = np.arange(start=X.min(),stop=X.max(),step=0.1)
X_grid = X_grid.reshape(-1,1)

if selected_model=="Polynomial Regression":
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
# ANIMATED VISUALIZATION
# ---------------------------------------------------
st.subheader("🎬 Salary Growth Animation")

anim_df = pd.DataFrame({
    "Position Level":X.flatten(),
    "Salary":y
})

fig_anim = px.scatter(
    anim_df,
    x="Position Level",
    y="Salary",
    size="Salary",
    animation_frame="Position Level"
)

st.plotly_chart(fig_anim,use_container_width=True)

# ---------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------
st.subheader("📥 Download Model Report")

csv = results_df.to_csv(index=False)

st.download_button(
    "Download CSV",
    csv,
    "model_performance_report.csv",
    "text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Python • Streamlit • Scikit-learn • Plotly")
