import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# 1. Page Configuration (Must be first)
# ✅ FIX 3: Switch to centered layout to stop wide-mode resize loops on HF
st.set_page_config(
    page_title="Solar Forecast AI",
    page_icon="☀️",
    layout="centered"
)

# 2. Custom Styling
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    /* ✅ FIX 4: Stop iframe height oscillation on Hugging Face */
    iframe {
        height: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar: Input Parameters
st.sidebar.header("🎛️ Input Conditions")

def user_input_features():
    st.sidebar.subheader("Weather")
    temp = st.sidebar.slider("Temperature (°C)", -10.0, 60.0, 25.5)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 45.0)
    ghi = st.sidebar.slider("GHI (W/m²)", 0.0, 1200.0, 600.0)
    
    st.sidebar.subheader("Time")
    hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
    
    st.sidebar.subheader("Lag Features")
    st.sidebar.caption("Power output from previous hours")
    power_t_1 = st.sidebar.number_input("Power (t-1)", min_value=0.0, value=150.0)
    power_t_2 = st.sidebar.number_input("Power (t-2)", min_value=0.0, value=140.0)
    
    # Feature Engineering (Backend expects sin/cos)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    
    return {
        "temperature": temp,
        "humidity": humidity,
        "ghi": ghi,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "power_t_1": power_t_1,
        "power_t_2": power_t_2
    }, hour

input_data, selected_hour = user_input_features()

# ✅ FIX 1: Cache Data Loading
# This prevents Streamlit from re-reading the CSV on every tiny UI interaction
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        # Fallback logic for different execution contexts
        alt_path = "../" + path
        if os.path.exists(alt_path):
            return pd.read_csv(alt_path)
        else:
            raise FileNotFoundError(f"Could not find {path} or {alt_path}")
    return pd.read_csv(path)

# 4. Main Dashboard Layout
st.title("☀️ Solar Power Forecasting System")
st.markdown("Predict future solar energy output using a machine learning model served via FastAPI.")
st.divider()

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("⚡ Live Prediction")
    st.caption("Send parameters to the ML API")
    
    if st.button("Generate Forecast", type="primary"):
        try:
            # Call FastAPI Endpoint
            api_url = "http://127.0.0.1:8000/predict"
            response = requests.post(api_url, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                pred_power = result["predicted_power"]
                version = result.get("model_version", "N/A")
                
                # Display Result (KPI Card Style)
                st.success("Prediction Successful")
                st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="margin:0; color:#555;">Predicted Output</h4>
                        <h1 style="margin:0; font-size: 3rem; color:#FF4B4B;">{pred_power:.2f} W</h1>
                        <p style="margin:0; color:#888;">Model Version: {version}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Server Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Refused!")
            st.info("Make sure your FastAPI backend is running on port 8000.")

with col2:
    st.subheader("📊 Historical Analysis")
    st.caption("Context from training data (Actual Power vs GHI)")
    
    try:
        data_path = "data/final_solar_dataset.csv"
        # Use the cached loader
        df = load_data(data_path)
        
        # ✅ FIX 2: Lock Chart Size & Styling
        # Smaller fixed size (8x4) prevents responsive resize loops in iframes
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Scatter plot with slightly lighter points
        scatter = ax.scatter(
            df['ghi'], 
            df['power_output'], 
            alpha=0.6, 
            c=df['temperature'], 
            cmap='coolwarm', 
            s=12
        )
        
        # Marker for current input
        ax.scatter([input_data['ghi']], [0], color='red', s=80, marker='x', label='Current Input GHI')
        
        # Styling
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)')
        ax.set_xlabel("Global Horizontal Irradiance (GHI)")
        ax.set_ylabel("Power Output (Watts)")
        ax.set_title("Historical Power vs GHI (Colored by Temp)")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # ✅ FIX 5: Freeze chart in container (Prevents HF iframe resizing jitter)
        plot_container = st.container()
        with plot_container:
            st.pyplot(fig, clear_figure=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(df.sample(100).sort_values("timestamp"), height=200)
            
    except Exception as e:
        st.warning(f"⚠️ Could not load historical data for visualization.\nError: {e}")

# Footer
st.markdown("---")
st.markdown("Rocket Research & Development | Solar Forecast Demo v1.0")