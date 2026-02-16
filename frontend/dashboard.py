import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import numpy as np

# 1. Page Configuration (Must be first)
# ✅ FIX 3: Switch to centered layout to stop wide-mode resize loops on HF/Render
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
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
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
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        # Fallback logic for different execution contexts
        alt_path = os.path.join("..", path)
        if os.path.exists(alt_path):
            return pd.read_csv(alt_path)
        else:
            return None
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
            # ✅ FIX 6: Dynamic API URL for Cloud Deployment
            default_url = "http://127.0.0.1:8000"
            api_base_url = os.getenv("API_URL", default_url).rstrip("/")
            api_url = f"{api_base_url}/predict"
            
            with st.spinner('Calculating...'):
                response = requests.post(api_url, json=input_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                # Support both naming conventions for robustness
                pred_power = result.get("predicted_power", result.get("prediction", 0))
                version = result.get("model_version", "v2")
                
                # Display Result (KPI Card Style)
                st.success("Success!")
                st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="margin:0; color:#555; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">Predicted Output</h4>
                        <h1 style="margin:10px 0; font-size: 3rem; color:#FF4B4B;">{pred_power:.2f} W</h1>
                        <p style="margin:0; color:#888; font-size: 0.7rem;">Model Integrity: Verified ({version})</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Server Error {response.status_code}")
                st.info(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Refused!")
            st.info(f"Target: {os.getenv('API_URL', 'http://127.0.0.1:8000')}")
            st.info("Check if the Backend API is active and the URL is correct.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with col2:
    st.subheader("📊 Historical Analysis")
    st.caption("Actual Power vs GHI context")
    
    try:
        data_path = "data/final_solar_dataset.csv"
        df = load_data(data_path)
        
        if df is not None:
            # ✅ FIX 2: Lock Chart Size & Styling
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            # Scatter plot
            scatter = ax.scatter(
                df['ghi'], 
                df['power_output'], 
                alpha=0.4, 
                c=df['temperature'], 
                cmap='coolwarm', 
                s=10
            )
            
            # Marker for current input
            ax.scatter([input_data['ghi']], [0], color='red', s=100, marker='X', label='Current Input GHI', zorder=5)
            
            # Styling
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Temperature (°C)')
            ax.set_xlabel("Global Horizontal Irradiance (GHI)")
            ax.set_ylabel("Power Output (Watts)")
            ax.set_title("Historical Training Context", fontsize=10)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.2)
            
            # ✅ FIX 5: Freeze chart in container
            st.pyplot(fig, clear_figure=True)
            
            with st.expander("Explore Training Samples"):
                st.dataframe(df.sample(min(100, len(df))).sort_index(), height=150)
        else:
            st.warning("⚠️ Historical dataset not found. Skipping visualization.")
            
    except Exception as e:
        st.warning(f"⚠️ Visualization error: {e}")

# Footer
st.markdown("---")
st.caption("© 2024 Rocket Research & Development | Solar Forecast Engine v1.1")
