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
from src.models.predict_model import RainPredictor


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

    print("--------------------")
    print("\033[1mLoading defaults, scaler and validation data:\033[0m")
    global defaults, scaler, validation_data
    
    try:
        # Load defaults
        with open('src/api/defaults.json', 'r') as f:
            defaults = json.load(f)
        print("\n- Defaults loaded to fill missing fields in input data.")
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("\n- Scaler loaded to scale input data before model prediction.")
        
        # Load validation data
        with open('src/api/validation_data.json', 'r') as f:
            validation_data = json.load(f)
        print("\n- Validation data loaded to check input data for location and season.\n")
        print("--------------------")
        
        return True
    except Exception as e:
        print(f"\n\033[1mWarning loading defaults/scaler: {e}\033[0m")
        return False


# Load production model from MLFlow
def load_production_model():
    """Load the current production model from MLflow"""
    global model, model_version, model_info
    
    try:
        client = MlflowClient()
        
        print("--------------------")
        # Get production model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not prod_versions:
            print("\n\033[1mNo production model found in MLflow.\033[0m")
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

        print("\n\033[1mLoading production model:\033[0m")
        print(f"\n- Loaded production model: Version {model_version}")
        print(f"\n- F1 Score: {model_info['metrics']['f1_score']:.4f}")
        print(f"\n- Accuracy: {model_info['metrics']['accuracy']:.4f}")
        print(f"\n- Precision: {model_info['metrics']['precision']:.4f}")
        print(f"\n- Recall: {model_info['metrics']['recall']:.4f}")
        print(f"\n- ROC-AUC: {model_info['metrics']['roc_auc']:.4f}")
        print(f"\n- Trained on: {model_info['params']['years']} data")
        print(f"\n- Split ID: {model_info['params']['split_id']}")
        print(f"\n- Number of training samples: {model_info['params']['train_samples']}")
        print("--------------------")

        return True
        
    except Exception as e:
        print(f"\n\033[1mError loading production model: {e}\033[0m")
        return False


# Load model when API starts
@app.on_event("startup")
async def startup_event():
    """Load production model on API startup"""
    print("\n\033[1m==============================\033[0m")
    print("\n\033[1mSTARTING RAIN PREDICTION API\033[0m")
    print("\n\033[1m==============================\033[0m")


    load_defaults()    

    success = load_production_model()
    
    if not success:
        print("\n\033[1mAPI started without model. Use /model/refresh to load.\033[0m")
    

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

    @property
    def model_validate(self):
        """Pydantic V2 validator"""
        return self
    
    def model_post_init(self, __context):
        """Custom validation after model initialization"""
        # Valid wind directions
        valid_wind_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        
        # 1. Validate rain_today (only 0 or 1)
        if self.rain_today not in [0, 1]:
            raise ValueError(f"\n\033[1mrain_today must be 0 (No) or 1 (Yes), got {self.rain_today}\033[0m")
        
        # 2. Validate rainfall (not negative)
        if self.rainfall is not None and self.rainfall < 0:
            raise ValueError(f"\n\033[1mrainfall cannot be negative, got {self.rainfall}\033[0m")
        
        # 3. Validate min_temp < max_temp
        if self.min_temp >= self.max_temp:
            raise ValueError( f"\n\033[1mmin_temp ({self.min_temp}) must be less than max_temp ({self.max_temp})\033[0m")
        
        # 4. Validate wind directions
        if self.wind_gust_dir and self.wind_gust_dir not in valid_wind_dirs:
            raise ValueError(f"\n\033[1mwind_gust_dir must be one of {valid_wind_dirs}, got '{self.wind_gust_dir}'\033[0m")

        if self.wind_dir_9am and self.wind_dir_9am not in valid_wind_dirs:
            raise ValueError(f"\n\033[1mwind_dir_9am must be one of {valid_wind_dirs}, got '{self.wind_dir_9am}'\033[0m")

        if self.wind_dir_3pm and self.wind_dir_3pm not in valid_wind_dirs:
            raise ValueError(f"\n\033[1mwind_dir_3pm must be one of {valid_wind_dirs}, got '{self.wind_dir_3pm}'\033[0m")
        
        # 5. Validate humidity (0-100%)
        if self.humidity_9am is not None and not (0 <= self.humidity_9am <= 100):
            raise ValueError(f"\n\033[1mhumidity_9am must be between 0-100, got {self.humidity_9am}\033[0m")

        if self.humidity_3pm is not None and not (0 <= self.humidity_3pm <= 100):
            raise ValueError(f"\n\033[1mhumidity_3pm must be between 0-100, got {self.humidity_3pm}\033[0m")
        
        # 6. Validate wind speed (not negative)
        if self.wind_gust_speed is not None and self.wind_gust_speed < 0:
            raise ValueError(f"\n\033[1mwind_gust_speed cannot be negative, got {self.wind_gust_speed}\033[0m")

        if self.wind_speed_9am is not None and self.wind_speed_9am < 0:
            raise ValueError(f"\n\033[1mwind_speed_9am cannot be negative, got {self.wind_speed_9am}\033[0m")

        if self.wind_speed_3pm is not None and self.wind_speed_3pm < 0:
            raise ValueError(f"\n\033[1mwind_speed_3pm cannot be negative, got {self.wind_speed_3pm}\033[0m")


# Endpoint 1: Train model
@app.post("/train")
async def train(split_id: int):
    """
    Train a new model on specified split.
    
    This triggers the training script. If the new model is better than the current production model,
    it will automatically be promoted to production and reload in API.
    """
    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mTrain Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    if not (1 <= split_id <= 9):
        raise HTTPException(
            status_code=400,
            detail="split_id must be between 1 and 9")
    
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
    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mFull Prediction Endpoint (110 features input):\033[0m")
    print("\n\033[1m--------------------\033[0m")

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
    Simplified prediction using predict_model module from src/models/predict_model.py    

    Required: location, date, min_temp, max_temp, rain_today
    Optional: All other weather features (wind, humidity, pressure, etc.)
    """

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mSimple Prediction Endpoint (5 input features needed):\033[0m")
    print("\n\033[1m--------------------\033[0m")

    if model is None or scaler is None or defaults is None:
        raise HTTPException(
            status_code=503,
            detail="Model, scaler, or defaults not loaded. Use POST /model/refresh."
        )

    try:
        # Use RainPredictor from predict_model.py
        predictor = RainPredictor()

        # Prepare optional features
        optional_features = {}
        if data.wind_gust_speed is not None:
            optional_features['wind_gust_speed'] = data.wind_gust_speed
        if data.wind_gust_dir:
            optional_features['wind_gust_dir'] = data.wind_gust_dir
        if data.wind_speed_9am is not None:
            optional_features['wind_speed_9am'] = data.wind_speed_9am
        if data.wind_dir_9am:
            optional_features['wind_dir_9am'] = data.wind_dir_9am
        if data.wind_speed_3pm is not None:
            optional_features['wind_speed_3pm'] = data.wind_speed_3pm
        if data.wind_dir_3pm:
            optional_features['wind_dir_3pm'] = data.wind_dir_3pm
        if data.humidity_9am is not None:
            optional_features['humidity_9am'] = data.humidity_9am
        if data.humidity_3pm is not None:
            optional_features['humidity_3pm'] = data.humidity_3pm
        if data.pressure_9am is not None:
            optional_features['pressure_9am'] = data.pressure_9am
        if data.pressure_3pm is not None:
            optional_features['pressure_3pm'] = data.pressure_3pm
        if data.temp_9am is not None:
            optional_features['temp_9am'] = data.temp_9am
        if data.temp_3pm is not None:
            optional_features['temp_3pm'] = data.temp_3pm
        if data.rainfall is not None:
            optional_features['rainfall'] = data.rainfall

        # Make prediction
        result = predictor.predict_simple(
            location=data.location,
            date=data.date,
            min_temp=data.min_temp,
            max_temp=data.max_temp,
            rain_today=data.rain_today,
            **optional_features
        )

        result["status"] = "success"
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
       

# Endpoint Health Check
@app.get("/health")
async def health():
    """Health check"""

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mHealth Check Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

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

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mAPI Info Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

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

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mModel Info Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

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

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mModel Reload Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

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
