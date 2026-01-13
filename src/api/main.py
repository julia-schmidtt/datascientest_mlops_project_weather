"""
FastAPI for rain prediction in Australia.
Loads automatically production model from MLFlow for predictions.

Endpoints:
- POST /train: Train new model
- POST /predict: Make predictions with production model (110 input features needed, use e.g. tests/test_api_prediction.py to extract single sample from test set)
- POST /predict/simple: Simplified prediction with production model (not 110 input features needed, missing features will be filled)
- GET  /health: Health check
- GET  /model/info: Get current production model info
- POST /model/refresh: Reload production model

Usage:
    python src/api/main.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import sys
import json
import pickle
from datetime import datetime

# Load environment variables
load_dotenv()

# Set MLflow credentials for DagsHub
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')

# Import params from params.yaml
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS


app = FastAPI(
    title="Rain Prediction API",
    description="Predict rain in Australia using XGBoost with MLflow integration"
)


# MLflow configuration
MLFLOW_TRACKING_URI = PARAMS['mlflow']['tracking_uri']
MODEL_NAME = "RainTomorrow_XGBoost"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Global model variable (loaded on startup and can be refreshed)
model = None
model_version = None
model_info = {}
scaler = None
defaults = None
validation_data = None


# Load defaults and scaler
def load_defaults():
    """Load default values and scaler"""
    global defaults, scaler, validation_data
    
    try:
        # Load defaults
        with open('src/api/defaults.json', 'r') as f:
            defaults = json.load(f)
        print("Defaults loaded.")
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded.")
        
        # Load validation data
        with open('src/api/validation_data.json', 'r') as f:
            validation_data = json.load(f)
        print("Validation data loaded.")
        
        return True
    except Exception as e:
        print(f"Warning loading defaults/scaler: {e}")
        return False


# Load production model from MLFlow
def load_production_model():
    """Load the current production model from MLflow"""
    global model, model_version, model_info
    
    try:
        client = MlflowClient()
        
        # Get production model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not prod_versions:
            print("No production model found in MLflow.")
            return False
        
        # Get latest production version
        prod_version = max(prod_versions, key=lambda x: int(x.version))
        model_version = prod_version.version
        
        # Load model
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.xgboost.load_model(model_uri)
        
        # Get model info
        run = client.get_run(prod_version.run_id)
        model_info = {
            "model_name": MODEL_NAME,
            "version": model_version,
            "stage": "Production",
            "run_id": prod_version.run_id,
            "registered_at": prod_version.creation_timestamp,
            "metrics": {
                "f1_score": run.data.metrics.get("f1_score"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision"),
                "recall": run.data.metrics.get("recall"),
                "roc_auc": run.data.metrics.get("roc_auc")
            },
            "params": {
                "split_id": run.data.params.get("split_id"),
                "years": run.data.params.get("years"),
                "train_samples": run.data.params.get("train_samples_original")
            }
        }
        
        print(f"Loaded production model: Version {model_version}")
        print(f"F1 Score: {model_info['metrics']['f1_score']:.4f}")
        print(f"Accuracy: {model_info['metrics']['accuracy']:.4f}")
        print(f"Precision: {model_info['metrics']['precision']:.4f}")
        print(f"Recall: {model_info['metrics']['recall']:.4f}")
        print(f"ROC-AUC: {model_info['metrics']['roc_auc']:.4f}")
        print(f"Trained on: {model_info['params']['years']}")
        
        return True
        
    except Exception as e:
        print(f"Error loading production model: {e}")
        return False


# Load model when API starts
@app.on_event("startup")
async def startup_event():
    """Load production model on API startup"""
    print("STARTING RAIN PREDICTION API")

    load_defaults()    
    success = load_production_model()
    
    if not success:
        print("API started without model. Use /model/refresh to load.")
    

# Input schema for predictions
class SimplePredictionInput(BaseModel):
    """Simplified input for real-world usage - only 5 required fields"""
    # Required basics
    location: str
    date: str  # "YYYY-MM-DD"
    min_temp: float
    max_temp: float
    rain_today: int # 0: No, 1:Yes
    
    # Optional detailed weather
    wind_gust_speed: Optional[float] = None
    wind_gust_dir: Optional[str] = None
    wind_speed_9am: Optional[float] = None
    wind_dir_9am: Optional[str] = None
    wind_speed_3pm: Optional[float] = None
    wind_dir_3pm: Optional[str] = None
    humidity_9am: Optional[float] = None
    humidity_3pm: Optional[float] = None
    pressure_9am: Optional[float] = None
    pressure_3pm: Optional[float] = None
    temp_9am: Optional[float] = None
    temp_3pm: Optional[float] = None
    rainfall: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "location": "Sydney",
                "date": "2025-01-15",
                "min_temp": 18.0,
                "max_temp": 28.0,
                "rain_today": 0,
                "wind_gust_speed": 35.0,
                "wind_gust_dir": "W",
                "humidity_9am": 65.0,
                "humidity_3pm": 45.0,
                "rainfall": 0.0
            }
        }



# Endpoint 1: Train model
@app.post("/train")
async def train(split_id: int):
    """
    Train a new model on specified split.
    
    This triggers the training script. If the new model is better than the current production model,
    it will automatically be promoted to production and reload in API.
    """
    if not (1 <= split_id <= 9):
        raise HTTPException(
            status_code=400,
            detail="split_id must be between 1 and 9"
        )
    
    try:
        print(f"Triggering training for split {split_id}.")
        
        # Run training script
        result = subprocess.run(
            ["python", "src/models/train_model.py", "--split_id", str(split_id)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")
        
        # Reload model (in case it was promoted)
        load_production_model()
        
        return {
            "status": "success",
            "message": f"Training completed for split {split_id}",
            "output": result.stdout[-500:],  # Last 500 chars
            "current_production": model_info
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Training timeout (>5 minutes)"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Endpoint 2: Make prediction with 110 features as input data
@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """
    Make a rain prediction using the production model (FULL 110 features required)
    
    Input: Dictionary with all required 110 features
    Output: Prediction (0=No Rain, 1=Rain) with probability
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Use POST /model/refresh first."
        )
    
    try:
        # Convert to DataFrame (single row)
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return {
            "status": "success",
            "prediction": int(prediction),
            "label": "Rain" if prediction == 1 else "No Rain",
            "probability_rain": float(probability),
            "probability_no_rain": float(1 - probability),
            "model_version": model_version,
            "model_info": {
                "f1_score": model_info['metrics']['f1_score'],
                "trained_on": model_info['params']['years']
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )


# Endpoint 3: Make prediction with simple input (5 features required)
@app.post("/predict/simple")
async def predict_simple(data: SimplePredictionInput):
    """
    Simplified prediction with minimal required features (5 features: location, date, min_temp, max_temp, rain_today).
    Missing features are automatically filled with training set defaults and properly scaled.
    
    Required: location, date, min_temp, max_temp, rain_today
    Optional: All other weather features (wind, humidity, pressure, etc.)
    """

    if model is None or scaler is None or defaults is None:
        raise HTTPException(
            status_code=503,
            detail="Model, scaler, or defaults not loaded. Use POST /model/refresh."
        )

    try:
        # Validate location
        if data.location not in validation_data['locations']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid location '{data.location}'. Must be one of: {', '.join(validation_data['locations'][:10])}."
            )
        
        # Extract date features
        date_obj = datetime.strptime(data.date, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.month

        # Map season
        season_map = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        }
        season = season_map[month]

        # Build feature dictionary with defaults
        features_raw = {
            'MinTemp': data.min_temp,
            'MaxTemp': data.max_temp,
            'Rainfall': data.rainfall,
            'WindGustSpeed': data.wind_gust_speed if data.wind_gust_speed is not None else defaults['train_medians']['WindGustSpeed'],
            'WindSpeed9am': data.wind_speed_9am if data.wind_speed_9am is not None else defaults['train_medians']['WindSpeed9am'],
            'WindSpeed3pm': data.wind_speed_3pm if data.wind_speed_3pm is not None else defaults['train_medians']['WindSpeed3pm'],
            'Humidity9am': data.humidity_9am if data.humidity_9am is not None else defaults['train_medians']['Humidity9am'],
            'Humidity3pm': data.humidity_3pm if data.humidity_3pm is not None else defaults['train_medians']['Humidity3pm'],
            'Pressure9am': data.pressure_9am if data.pressure_9am is not None else defaults['train_medians']['Pressure9am'],
            'Pressure3pm': data.pressure_3pm if data.pressure_3pm is not None else defaults['train_medians']['Pressure3pm'],
            'Temp9am': data.temp_9am if data.temp_9am is not None else defaults['train_medians']['Temp9am'],
            'Temp3pm': data.temp_3pm if data.temp_3pm is not None else defaults['train_medians']['Temp3pm'],
            'Rainfall': data.rainfall if data.rainfall is not None else defaults['train_medians']['Rainfall'],
            'Year': year,
            'Location': data.location,
            'WindGustDir': data.wind_gust_dir if data.wind_gust_dir else defaults['train_modes']['WindGustDir'],
            'WindDir9am': data.wind_dir_9am if data.wind_dir_9am else defaults['train_modes']['WindDir9am'],
            'WindDir3pm': data.wind_dir_3pm if data.wind_dir_3pm else defaults['train_modes']['WindDir3pm'],
            'Season': season
        }

        # Create DataFrame
        df = pd.DataFrame([features_raw])
        
        # One-hot encode categoricals (same as preprocessing)
        categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Season']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Get all expected columns from training (110 features)
        expected_cols = pd.read_csv('data/processed/X_train.csv', nrows=0).columns.tolist()
        
        # Add missing columns with 0
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Delete column if not expected
        for col in df.columns:
            if col not in expected_cols:
                df = df.drop(col, axis=1)

        # Reorder columns to match training
        df = df[expected_cols]
        
        # Scale numerical features
        numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
                         'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                         'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
        
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "status": "success",
            "prediction": int(prediction),
            "label": "Rain Tomorrow" if prediction == 1 else "No Rain Tomorrow",
            "probability_rain": float(probability),
            "probability_no_rain": float(1 - probability),
            "inputs_used": {
                "location": data.location,
                "date": data.date,
                "season": season,
                "year": year,
                "rain_today": data.rain_today
            },
            "model_version": model_version,
            "model_f1_score": round(model_info['metrics']['f1_score'], 4)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# Endpoint Health Check
@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version if model else None,
        "scaler_loaded": scaler is not None,
        "defaults_loaded": defaults is not None
    }



# Endpoint API information
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Rain Prediction API",
        "description": "Predict rain in Australia using production MLflow model",
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model/info",
            "model_refresh": "POST /model/refresh",
            "predict_full": "POST /predict",
            "predict_simple": "POST /predict/simple",
            "train": "POST /train"
        },
        "mlflow_ui": MLFLOW_TRACKING_URI,
        "features": {
            "simple_prediction": "Only 5 required fields (location, date, min_temp, max_temp, rain_today)",
            "auto_scaling": "Automatic feature scaling for input data.",
            "default_filling": "Missing features filled with training defaults."
        }
    }


# Endpoint model information
@app.get("/model/info")
async def get_model_info():
    """Get current production model information"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Use POST /model/refresh to load production model."
        )
    
    return {
        "status": "success",
        "model": model_info
    }


# Endpoint refresh model
@app.post("/model/refresh")
async def refresh_model():
    """Refresh/reload the production model from MLflow"""
    print("Refreshing production model from MLflow.")
    
    load_defaults() 
    success = load_production_model()
    
    if success:
        return {
            "status": "success",
            "message": "Production model reloaded",
            "model": model_info
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to load production model from MLflow"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
