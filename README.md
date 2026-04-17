☀️ Solar Power Forecasting System

A high-performance, microservice-style machine learning application built to predict solar power output based on real-time meteorological data. This project leverages an optimized Random Forest model to provide accurate forecasts, serving them via a robust FastAPI backend and a dynamic Streamlit dashboard.

📖 Table of Contents

Overview

System Architecture

Machine Learning Pipeline

Getting Started

API Documentation

Project Structure

License

🌟 Overview

Reliable solar power forecasting is critical for grid stability and efficient energy management. This system integrates high-frequency generation data from Kaggle with localized meteorological data from NASA POWER (Bhopal region) to deliver:

Real-time Predictions: Inference based on temperature, humidity, and GHI.

Drift Monitoring: Automatic logging of predictions for future model retraining.

Interactive Visuals: Comparative analysis of historical trends vs. current live inputs.

🏗 System Architecture

The project follows a decoupled microservices pattern, ensuring scalability and ease of maintenance.

graph LR
    A[Meteorological Data] --> B[FastAPI Backend]
    C[Model Artifacts] --> B
    B --> D[Prediction Logs]
    B <--> E[Streamlit Frontend]
    E --> F[User Dashboard]


Key Components:

Backend (Inference Engine): A FastAPI service that manages model lifecycles and validates physical constraints (e.g., ensuring temperature is within realistic bounds).

Frontend (UX/UI): A Streamlit dashboard that provides an intuitive interface for users to input data and visualize results.

Data Pipeline: An ETL process that merges high-frequency power data with hourly NASA weather data.

🧠 Machine Learning Pipeline

Model Details

Algorithm: Random Forest Regressor

Feature Set: ['temperature', 'humidity', 'ghi', 'hour_sin', 'hour_cos', 'power_t_1', 'power_t_2']

Normalization: StandardScaler applied to all numerical features.

Evaluation: Validated using TimeSeriesSplit to ensure temporal integrity, achieving high accuracy across RMSE and MAE metrics.

Feature Engineering

Temporal Encodings: Sin/Cos transformations for hours to capture diurnal cycles.

Lag Features: Inclusion of $Power_{t-1}$ and $Power_{t-2}$ to capture short-term momentum.

🚀 Getting Started

Prerequisites

Python 3.9+

Docker (Optional)

Installation & Setup

Clone the Repository

git clone [https://github.com/VarshVishwakarma/solar-forecast.git](https://github.com/VarshVishwakarma/solar-forecast.git)
cd solar-forecast


Environment Setup

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


Run the Services

Start Backend: uvicorn app.main:app --host 0.0.0.0 --port 8000

Start Frontend: streamlit run frontend/dashboard.py

📡 API Documentation

The backend provides a self-documenting Swagger UI accessible at /docs.

Endpoint

Method

Description

/health

GET

System status and model versioning.

/predict

POST

Returns power forecast (Watts) based on weather JSON.

📂 Project Structure

├── app/
│   ├── main.py              # FastAPI service & lifespan management
│   ├── model_v2.joblib      # Trained Random Forest model
│   ├── scaler_v2.joblib     # Feature scaling artifact
│   └── prediction_logs.csv  # Historical logs for drift analysis
├── frontend/
│   └── dashboard.py         # Streamlit UI & Data visualization
├── data/
│   └── final_solar_dataset.csv # Processed dataset for dashboard metrics
├── Dockerfile               # Containerization logic
└── requirements.txt         # Project dependencies


🛠 Tech Stack

Core: Python 3.9, FastAPI, Streamlit

ML/Science: Scikit-Learn, Pandas, NumPy, Joblib

Visualization: Matplotlib, Seaborn

DevOps: Docker, Uvicorn, Pydantic

🧑‍💻 Author

Varsh Vishwakarma
AI • ML • DL • Data Science • Cloud • Full-Stack ML Developer

GitHub: @VarshVishwakarma

Hugging Face: Space

Developed for the intersection of Machine Learning and Renewable Energy.
