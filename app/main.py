import os
import logging
import joblib
import numpy as np
import csv  # ✅ Added for logging
from datetime import datetime  # ✅ Added for timestamps
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary to store model artifacts
ml_models = {}

# Version Control
MODEL_VERSION = "v2.0"

# 2. Lifespan Manager (Modern startup/shutdown logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logger.info(f"🔄 Loading ML models (Version: {MODEL_VERSION})...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Updated filenames to include versioning
        model_path = os.path.join(base_dir, "model_v2.joblib")
        scaler_path = os.path.join(base_dir, "scaler_v2.joblib")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            ml_models["model"] = joblib.load(model_path)
            ml_models["scaler"] = joblib.load(scaler_path)
            logger.info("✅ Model and Scaler loaded successfully.")
        else:
            logger.error("⚠️ Critical: Model or Scaler file not found in app/ directory.")
            logger.info(f"Expected: {model_path}")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")

    yield  # Application runs here

    # --- Shutdown Logic ---
    ml_models.clear()
    logger.info("🛑 Models unloaded.")

# 3. Initialize FastAPI with lifespan
app = FastAPI(
    title="Solar Power Prediction API", 
    version="2.1",
    description="Production-ready API for Solar Irradiance Forecasting",
    lifespan=lifespan
)

# 4. Define Input Schema with Examples & Validation
class SolarInput(BaseModel):
    temperature: float = Field(..., ge=-10, le=60, description="Ambient temperature in Celsius", json_schema_extra={"example": 25.5})
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %", json_schema_extra={"example": 45.0})
    ghi: float = Field(..., ge=0, description="Global Horizontal Irradiance", json_schema_extra={"example": 600.5})
    hour_sin: float = Field(..., description="Cyclical hour feature (Sine)", json_schema_extra={"example": -0.5})
    hour_cos: float = Field(..., description="Cyclical hour feature (Cosine)", json_schema_extra={"example": -0.866})
    power_t_1: float = Field(..., ge=0, description="Power output 1 hour ago", json_schema_extra={"example": 150.0})
    power_t_2: float = Field(..., ge=0, description="Power output 2 hours ago", json_schema_extra={"example": 140.0})

# 5. Endpoints
@app.get("/")
def health_check():
    """Simple health check to verify service status."""
    response = {
        "status": "ok", 
        "message": "Solar Forecasting API is ready",
        "documentation_url": "/docs"
    }
    if not ml_models:
        response["status"] = "warning"
        response["message"] = "Service running but models not loaded"
    return response

@app.get("/health")
def health():
    """Ops-friendly health check endpoint."""
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict_solar_power(data: SolarInput):
    """
    Predicts solar power output based on weather and lag features.
    Logs inputs and predictions to app/prediction_logs.csv.
    """
    if "model" not in ml_models or "scaler" not in ml_models:
        raise HTTPException(status_code=503, detail="ML Model is not loaded available")

    try:
        # Prepare features (Order must match training!)
        features = np.array([[
            data.temperature,
            data.humidity,
            data.ghi,
            data.hour_sin,
            data.hour_cos,
            data.power_t_1,
            data.power_t_2
        ]])

        # Scale features
        scaler = ml_models["scaler"]
        scaled_features = scaler.transform(features)

        # Predict
        model = ml_models["model"]
        prediction = model.predict(scaled_features)
        pred_value = float(prediction[0])

        # --- 📝 LOGGING LOGIC ---
        try:
            # We use absolute path to ensure logs are saved in the app folder regardless of where script is run
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(base_dir, "prediction_logs.csv")
            
            file_exists = os.path.isfile(log_file)
            
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                # Write Header if file is new
                if not file_exists:
                    writer.writerow(["timestamp", "temperature", "humidity", "ghi", "power_t_1", "power_t_2", "predicted_power"])
                
                # Write Data Row
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    data.temperature,
                    data.humidity,
                    data.ghi,
                    data.power_t_1,
                    data.power_t_2,
                    pred_value
                ])
        except Exception as log_error:
            # We catch logging errors so they don't crash the user's prediction request
            logger.error(f"⚠️ Failed to write log: {log_error}")

        return {
            "predicted_power": pred_value,
            "unit": "Watts",
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

if __name__ == "__main__":
    import uvicorn
    # This allows running the script directly with Python
    print("\n🌞 Solar API is starting...")
    print("👉 Open this link for the Dashboard: http://127.0.0.1:8000/docs")
    print() # Print a clean newline
    uvicorn.run(app, host="127.0.0.1", port=8000)