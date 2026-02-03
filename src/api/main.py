"""
FastAPI for rain prediction in Australia.
Loads automatically production model from MLFlow for predictions.

Endpoints:
- POST /train: Train new model
- POST /predict: Make predictions with production model (110 input features needed, use e.g. tests/test_api_prediction.py to extract single sample from test set)
- POST /predict/simple: Simplified prediction with production model (not 110 input features needed, missing features will be filled)
- POST /pipeline/next-split: Automated pipeline (create split, track, train, promote)
- POST /pipeline/next-split-drift-detection: Automated pipeline (create split, check data drift, track, train if data drift present, promote)
- GET  /health: Health check
- GET  /model/info: Get current production model info
- POST /model/refresh: Reload production model

Usage:
    python src/api/main.py
"""

from fastapi import FastAPI, HTTPException, Request, Response
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
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge
from src.utils.track_with_dvc import track_split

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
AUTOMATION_EXPERIMENT_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M')}_Automated_Pipeline_WeatherPrediction_Australia"
MODEL_NAME = "RainTomorrow_XGBoost"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Prometheus metrics definition
registry = CollectorRegistry()

api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'Duration of API requests in seconds',
    ['endpoint', 'method'],
    buckets=[0.1, 0.2, 0.5, 1, 2, 5, 10],
    registry=registry
)

model_accuracy = Gauge("model_accuracy", "Accuracy of the current production model", registry=registry)
model_f1_score = Gauge("model_f1_score", "F1 Score of the current production model", registry=registry)
model_precision = Gauge("model_precision", "Precision of the current production model", registry=registry)
model_recall = Gauge("model_recall", "Recall of the current production model", registry=registry)
model_roc_auc = Gauge("model_roc_auc", "ROC-AUC of the current production model", registry=registry)
model_split_id = Gauge("model_split_id", "Training data split ID of the current production model", registry=registry)


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
        with open('/app/src/api/defaults.json', 'r') as f:
            defaults = json.load(f)
        print("\n- Defaults loaded to fill missing fields in input data.")
        
        # Load scaler
        with open('/app/models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("\n- Scaler loaded to scale input data before model prediction.")
        
        # Load validation data
        with open('/app/src/api/validation_data.json', 'r') as f:
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

        # Update Prometheus gauges
        model_accuracy.set(model_info['metrics']['accuracy'])
        model_f1_score.set(model_info['metrics']['f1_score'])
        model_precision.set(model_info['metrics']['precision'])
        model_recall.set(model_info['metrics']['recall'])
        model_roc_auc.set(model_info['metrics']['roc_auc'])
        model_split_id.set(int(model_info['params']['split_id']))

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
            raise ValueError(f"rain_today must be 0 (No) or 1 (Yes), got {self.rain_today}")
        
        # 2. Validate rainfall (not negative)
        if self.rainfall is not None and self.rainfall < 0:
            raise ValueError(f"rainfall cannot be negative, got {self.rainfall}")
        
        # 3. Validate min_temp < max_temp
        if self.min_temp >= self.max_temp:
            raise ValueError( f"min_temp ({self.min_temp}) must be less than max_temp ({self.max_temp})")
        
        # 4. Validate wind directions
        if self.wind_gust_dir and self.wind_gust_dir not in valid_wind_dirs:
            raise ValueError(f"wind_gust_dir must be one of {valid_wind_dirs}, got '{self.wind_gust_dir}'")

        if self.wind_dir_9am and self.wind_dir_9am not in valid_wind_dirs:
            raise ValueError(f"wind_dir_9am must be one of {valid_wind_dirs}, got '{self.wind_dir_9am}'")

        if self.wind_dir_3pm and self.wind_dir_3pm not in valid_wind_dirs:
            raise ValueError(f"wind_dir_3pm must be one of {valid_wind_dirs}, got '{self.wind_dir_3pm}'")
        
        # 5. Validate humidity (0-100%)
        if self.humidity_9am is not None and not (0 <= self.humidity_9am <= 100):
            raise ValueError(f"humidity_9am must be between 0-100, got {self.humidity_9am}")

        if self.humidity_3pm is not None and not (0 <= self.humidity_3pm <= 100):
            raise ValueError(f"humidity_3pm must be between 0-100, got {self.humidity_3pm}")
        
        # 6. Validate wind speed (not negative)
        if self.wind_gust_speed is not None and self.wind_gust_speed < 0:
            raise ValueError(f"wind_gust_speed cannot be negative, got {self.wind_gust_speed}")

        if self.wind_speed_9am is not None and self.wind_speed_9am < 0:
            raise ValueError(f"wind_speed_9am cannot be negative, got {self.wind_speed_9am}")

        if self.wind_speed_3pm is not None and self.wind_speed_3pm < 0:
            raise ValueError(f"wind_speed_3pm cannot be negative, got {self.wind_speed_3pm}")


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
    start_time = time.time()
    status_code = 200
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
            timeout=300  
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
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/train", method="POST").observe(duration)
        api_requests_total.labels(endpoint="/train", method="POST", status_code=status_code).inc()


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
    start_time = time.time()
    status_code = 200
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
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/predict", method="POST").observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()



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
    start_time = time.time()
    status_code = 200
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
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/predict/simple", method="POST").observe(duration)
        api_requests_total.labels(endpoint="/predict/simple", method="POST", status_code=status_code).inc()


# Endpoint 4: Automated Pipeline - Process Next Split
@app.post("/pipeline/next-split")
async def process_next_split():
    """
    Complete automated pipeline: Create next split, track with DVC, train, compare, promote.
    
    Workflow:
    1. Create next temporal split (incrementally adds one year)
    2. Track with DVC (git + dvc push)
    3. Train model on new split
    4. Track with MLflow
    5. Compare with production model
    6. Auto-promote if better performance (F1 score)
    7. Reload production model in API
    """

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mAutomated Pipeline:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    start_time = time.time()
    status_code = 200

    try:

        # STEP 1: Create next training data split
        print("\nSTEP 1: Creating next training data split.")
        create_result = subprocess.run(
            ["python", "src/data/automation_create_split.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/app" 
        )

        if create_result.returncode != 0:
            # Check if all splits complete
            if "All splits complete" in create_result.stdout:
                return {
                    "status": "info",
                    "message": "All temporal splits already created.",
                    "suggestion": "No more splits to process. Data accumulation simulation complete."
                }

            raise Exception(f"Training data split creation failed: {create_result.stderr}")

        # Extract split_id from output
        output_lines = create_result.stdout.strip().split('\n')
        split_line = [l for l in output_lines if 'Split' in l and 'created successfully' in l.lower()]
        if not split_line:
            raise Exception("Could not determine split_id from output")

        # Parse split_id
        try:
            split_id = int(split_line[0].split('Split')[1].split()[0])
        except (IndexError, ValueError):
            raise Exception(f"Could not parse split_id from: {split_line[0]}")
        
        print(f"Split {split_id} created")
        print(create_result.stdout)

        # STEP 2: Skip DVC tracking
        print(f"\nSTEP 2: DVC tracking skipped (use /pipeline/next-split-dvc for DVC)")
        dvc_status = "skipped"
        git_commit = None
              
        # STEP 3: Train model
        print(f"\nSTEP 3: Training model on training data split {split_id}.")
        automation_experiment = AUTOMATION_EXPERIMENT_NAME

        train_result = subprocess.run(
            ["python", "src/models/train_model_pipeline.py", "--split_id", str(split_id), "--experiment_name", automation_experiment],
            capture_output=True,
            text=True,
            timeout=600,
            cwd="/app"
        )

        if train_result.returncode != 0:
            raise Exception(f"Training failed: {train_result.stderr}")
        
        print("Training completed")

        # STEP 4: Reload production model
        print("\nSTEP 4: Reloading production model.")
        load_production_model()
        print("Production model reloaded")

        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("------------------")

        return {
            "status": "success",
            "message": f"Pipeline completed: Split {split_id} created, tracked, and trained",
            "split_processed": split_id,
            "pipeline_steps": {
                "split_creation": "completed",
                "dvc_tracking": dvc_status,
                "git_commit": git_commit,
                "model_training": "completed",
                "model_reload": "completed"
            },
            "current_production_model": model_info,
            "outputs": {
                "split_creation": create_result.stdout,
                "dvc_tracking": "skipped",
                "training": train_result.stdout[-500:]
            }
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/pipeline/next-split", method="POST").observe(duration)
        api_requests_total.labels(endpoint="/pipeline/next-split", method="POST", status_code=status_code).inc()


# Endpoint 5: Automated Pipeline Process Next Split with Drift Monitoring
@app.post("/pipeline/next-split-drift-detection")
async def process_next_split_with_drift_monitoring():
    """
    Automated pipeline with drift-triggered training.
    
    Workflow:
    1. Create next temporal split
    3. Check for data drift vs production model's training data
    4. If drift > threshold: Train new model (compare performance of new model with performance of production model)
    5. If no drift: Skip training 
    """
    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mAutomated Pipeline with Drift Detection:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    start_time = time.time()
    status_code = 200

    try:

        # STEP 1: Create next training data split
        print("\nSTEP 1: Creating next training data split.")
        result = subprocess.run(
            ["python", "src/data/automation_create_split.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/app" 
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Split creation failed: {result.stderr}"
            )
        
        split_output = result.stdout
        print(split_output)

        # Extract split_id from output
        import re
        match = re.search(r'Split (\d+) created', split_output)
        if not match:
            raise HTTPException(
                status_code=500,
                detail="Could not determine split_id from output"
            )
        
        split_id = int(match.group(1))
        print(f"Split {split_id} created")
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Split creation timed out"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating split: {str(e)}"
        )


    # STEP 2: Skip DVC tracking
    print(f"\nSTEP 2: DVC tracking skipped (use /pipeline/next-split-drift-detection-dvc for DVC)")
    dvc_status = "skipped"
    git_commit = None


    # STEP 3: DRIFT CHECK
    print("STEP 3: Data Drift Monitoring")
    
    # Load monitoring config
    import yaml
    with open('/app/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        drift_config = params.get('monitoring', {})
        drift_threshold = drift_config.get('drift_threshold', 0.10)
        drift_enabled = drift_config.get('drift_check_enabled', True)

    
    drift_result = None
    skip_training = False
    
    if split_id == 1:
        print("\nFirst split - no drift check needed (no reference)")
        print("Proceeding to training.")
    elif not drift_enabled:
        print("\nDrift check disabled in config")
        print("Proceeding to training.")
    else:
        try:
            from src.monitoring.data_drift import DataDriftMonitor
            
            monitor = DataDriftMonitor()
            drift_result = monitor.check_drift_for_split(split_id, save_report=True)
            
            drift_percentage = drift_result['share_of_drifted_columns']
            
            print(f"\nDrift Analysis:")
            print(f"  Reference: Production Model (Split {drift_result['reference_split']})")
            print(f"  Current: Split {split_id}")
            print(f"  Drifted Columns: {drift_result['number_of_drifted_columns']}/110 ({drift_percentage*100:.1f}%)")
            print(f"  Threshold: {drift_threshold*100:.1f}%")
            print(f"  Dataset Drift: {drift_result['dataset_drift']}")
            
            if drift_percentage < drift_threshold:
                # NO SIGNIFICANT DRIFT - SKIP TRAINING
                print(f"\nDECISION: Drift below threshold")
                print(f" SKIPPING TRAINING (save resources)")
                skip_training = True
            else:
                # DRIFT DETECTED - CONTINUE TO TRAINING
                print(f"\nDECISION: Drift above threshold")
                print(f" PROCEEDING TO TRAINING")
                print(f"\n  Alert Level: {drift_result['alert']['severity']}")
                print(f"  Message: {drift_result['alert']['message']}")
                print(f"  Recommendation: {drift_result['alert']['recommendation']}")
                
        except Exception as e:
            print(f"\nDrift check failed: {e}")
            print("  Proceeding to training anyway (failsafe)...")
    
    # Return if skipping training
    if skip_training:
        return {
            "status": "skipped",
            "reason": "no_significant_drift",
            "message": f"Drift below threshold ({drift_threshold*100:.0f}%), training skipped to save resources",
            "split_processed": split_id,
            "drift_check": {
                "enabled": True,
                "drift_detected": drift_result['dataset_drift'],
                "drift_percentage": f"{drift_result['share_of_drifted_columns']*100:.1f}%",
                "threshold": f"{drift_threshold*100:.1f}%",
                "drifted_columns": drift_result['number_of_drifted_columns'],
                "reference_split": drift_result['reference_split'],
                "alert": drift_result['alert']
            },
            "pipeline_steps": {
                "split_creation": "completed",
                "dvc_tracking": dvc_status,
                "git_commit": git_commit,
                "drift_check": "passed_no_significant_drift",
                "model_training": "skipped",
                "model_reload": "skipped"
            }
        }
    
    # STEP 4: Training
    print(f"STEP 4: Training model on split {split_id}")
    
    try:
        automation_experiment = AUTOMATION_EXPERIMENT_NAME
        result = subprocess.run(
            ["python", "src/models/train_model_pipeline.py", "--split_id", str(split_id), "--experiment_name", automation_experiment],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/app"
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {result.stderr}"
            )
        
        training_output = result.stdout
        print(training_output)
        print("Training completed")
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Training timed out"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during training: {str(e)}"
        )
    
    # STEP 5: Model Reload
    print("STEP 5: Reloading production model")
    
    load_defaults()
    success = load_production_model()
    
    if success:
        print("Production model loaded")
    else:
        print("Warning: Could not reload production model")
    
    print("\n" + "="*60)
    print("DRIFT-BASED PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    # Return detailed response
    response = {
        "status": "success",
        "message": f"Drift-detection pipeline completed: Split {split_id} created, drift detected, model trained",
        "split_processed": split_id,
        "drift_check": {
            "enabled": True,
            "drift_detected": drift_result['dataset_drift'] if drift_result else None,
            "drift_percentage": f"{drift_result['share_of_drifted_columns']*100:.1f}%" if drift_result else "N/A",
            "threshold": f"{drift_threshold*100:.1f}%",
            "drifted_columns": drift_result['number_of_drifted_columns'] if drift_result else None,
            "reference_split": drift_result['reference_split'] if drift_result else None,
            "alert": drift_result['alert'] if drift_result else None
        },
        "pipeline_steps": {
            "split_creation": "completed",
            "dvc_tracking": dvc_status,
            "git_commit": git_commit,
            "drift_check": "drift_detected_training_triggered" if drift_result else "skipped_first_split",
            "model_training": "completed",
            "model_reload": "completed"
        },
        "current_production_model": model_info,
        "outputs": {
            "split_creation": split_output,
            "dvc_tracking": "skipped",
            "training": training_output
        }
    }
    
    
    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(endpoint="/pipeline/next-split-drift-detection", method="POST").observe(duration)
    api_requests_total.labels(endpoint="/pipeline/next-split-drift-detection", method="POST", status_code=status_code).inc()


    return response



# Endpoint 6: Automated Pipeline with DVC - Process Next Split
@app.post("/pipeline/next-split-dvc")
async def process_next_split_with_dvc():
    """
    Complete automated pipeline with DVC tracking.

    Workflow:
    1. Create next temporal split
    2. Track with DVC + Git
    3. Train model on new split
    4. Compare with production model
    5. Auto-promote if better
    6. Reload production model
    """

    print("\n\033[1m--------------------\033[0m")
    print("\n\033[1mAutomated Pipeline with DVC:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    start_time = time.time()
    status_code = 200

    try:
        # STEP 1: Create split
        print("\nSTEP 1: Creating next training data split.")
        create_result = subprocess.run(
            ["python", "src/data/automation_create_split.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/app"
        )

        if create_result.returncode != 0:
            if "All splits complete" in create_result.stdout:
                return {
                    "status": "info",
                    "message": "All temporal splits already created.",
                    "suggestion": "No more splits to process."
                }
            raise Exception(f"Split creation failed: {create_result.stderr}")

        # Extract split_id
        output_lines = create_result.stdout.strip().split('\n')
        split_line = [l for l in output_lines if 'Split' in l and 'created successfully' in l.lower()]
        if not split_line:
            raise Exception("Could not determine split_id")

        try:
            split_id = int(split_line[0].split('Split')[1].split()[0])
        except (IndexError, ValueError):
            raise Exception(f"Could not parse split_id from: {split_line[0]}")

        print(f"Split {split_id} created")
        print(create_result.stdout)

        # STEP 2: Track with DVC
        print(f"\nSTEP 2: Tracking Split {split_id} with DVC and Git.")
        dvc_result = track_split(split_id)

        if dvc_result["status"] != "success":
            print(f"[WARNING] DVC tracking failed: {dvc_result['message']}")
            dvc_status = "failed"
            git_commit = None
        else:
            dvc_status = "completed"
            git_commit = dvc_result["commit_hash"]
            print(f"[SUCCESS] DVC tracking completed. Commit: {git_commit[:8]}")

        # STEP 3: Train model
        print(f"\nSTEP 3: Training model on split {split_id}.")
        automation_experiment = AUTOMATION_EXPERIMENT_NAME

        train_result = subprocess.run(
            ["python", "src/models/train_model_pipeline.py", "--split_id", str(split_id), "--experiment_name", automation_experiment],
            capture_output=True,
            text=True,
            timeout=600,
            cwd="/app"
        )

        if train_result.returncode != 0:
            raise Exception(f"Training failed: {train_result.stderr}")

        print("Training completed")

        # STEP 4: Reload model
        print("\nSTEP 4: Reloading production model.")
        load_production_model()
        print("Production model reloaded")

        print("PIPELINE WITH DVC COMPLETED")
        print("------------------")

        return {
            "status": "success",
            "message": f"Pipeline with DVC completed: Split {split_id} created, tracked, trained",
            "split_processed": split_id,
            "pipeline_steps": {
                "split_creation": "completed",
                "dvc_tracking": dvc_status,
                "git_commit": git_commit,
                "model_training": "completed",
                "model_reload": "completed"
            },
            "current_production_model": model_info,
            "outputs": {
                "split_creation": create_result.stdout,
                "dvc_tracking": dvc_result.get("message", ""),
                "training": train_result.stdout[-500:]
            }
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/pipeline/next-split-dvc", method="POST").observe(duration)
        api_requests_total.labels(endpoint="/pipeline/next-split-dvc", method="POST", status_code=status_code).inc()


# Endpoint 7: Automated Pipeline with DVC and Drift Detection
@app.post("/pipeline/next-split-drift-detection-dvc")
async def process_next_split_with_drift_and_dvc():
    """
    Automated pipeline with DVC tracking and drift-triggered training.

    Workflow:
    1. Create next temporal split
    2. Track with DVC + Git
    3. Check for data drift
    4. If drift > threshold: Train model
    5. If no drift: Skip training
    """
    print("\n\033[1m--------------------\033[0m")
    print("\n\033[1mAutomated Pipeline with DVC + Drift:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    start_time = time.time()
    status_code = 200

    try:
        # STEP 1: Create split
        print("\nSTEP 1: Creating next training data split.")
        result = subprocess.run(
            ["python", "src/data/automation_create_split.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/app"
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Split creation failed: {result.stderr}")

        split_output = result.stdout
        print(split_output)

        # Extract split_id
        import re
        match = re.search(r'Split (\d+) created', split_output)
        if not match:
            raise HTTPException(status_code=500, detail="Could not determine split_id")

        split_id = int(match.group(1))
        print(f"Split {split_id} created")

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Split creation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating split: {str(e)}")

    # STEP 2: Track with DVC
    print(f"\nSTEP 2: Tracking Split {split_id} with DVC and Git.")
    dvc_result = track_split(split_id)

    if dvc_result["status"] != "success":
        print(f"[WARNING] DVC tracking failed: {dvc_result['message']}")
        dvc_status = "failed"
        git_commit = None
    else:
        dvc_status = "completed"
        git_commit = dvc_result["commit_hash"]
        print(f"[SUCCESS] DVC tracking completed. Commit: {git_commit[:8]}")

    # STEP 3: Drift check
    print("STEP 3: Data Drift Monitoring")

    drift_config = PARAMS.get('monitoring', {})
    drift_threshold = drift_config.get('drift_threshold', 0.10)
    drift_enabled = drift_config.get('drift_check_enabled', True)


    drift_result = None
    skip_training = False

    if split_id == 1:
        print("\nFirst split - no drift check needed")
        print("Proceeding to training.")
    elif not drift_enabled:
        print("\nDrift check disabled")
        print("Proceeding to training.")
    else:
        try:
            from src.monitoring.data_drift import DataDriftMonitor

            monitor = DataDriftMonitor()
            drift_result = monitor.check_drift_for_split(split_id, save_report=True)
            drift_percentage = drift_result['share_of_drifted_columns']

            print(f"\nDrift Analysis:")
            print(f"  Drifted Columns: {drift_result['number_of_drifted_columns']}/110 ({drift_percentage*100:.1f}%)")
            print(f"  Threshold: {drift_threshold*100:.1f}%")

            if drift_percentage < drift_threshold:
                print(f"\nDECISION: Drift below threshold - SKIPPING TRAINING")
                skip_training = True
            else:
                print(f"\nDECISION: Drift above threshold - PROCEEDING TO TRAINING")

        except Exception as e:
            print(f"\nDrift check failed: {e}")
            print("Proceeding to training anyway...")

    # Return if skipping
    if skip_training:
        return {
            "status": "skipped",
            "reason": "no_significant_drift",
            "message": f"Drift below threshold, training skipped",
            "split_processed": split_id,
            "drift_check": {
                "enabled": True,
                "drift_detected": drift_result['dataset_drift'],
                "drift_percentage": f"{drift_result['share_of_drifted_columns']*100:.1f}%",
                "threshold": f"{drift_threshold*100:.1f}%"
            },
            "pipeline_steps": {
                "split_creation": "completed",
                "dvc_tracking": dvc_status,
                "git_commit": git_commit,
                "drift_check": "passed_no_significant_drift",
                "model_training": "skipped",
                "model_reload": "skipped"
            }
        }

    # STEP 4: Training
    print(f"STEP 4: Training model on split {split_id}")

    try:
        automation_experiment = AUTOMATION_EXPERIMENT_NAME
        result = subprocess.run(
            ["python", "src/models/train_model_pipeline.py", "--split_id", str(split_id), "--experiment_name", automation_experiment],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/app"
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")

        training_output = result.stdout
        print(training_output)
        print("Training completed")

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Training timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

    # STEP 5: Reload model
    print("STEP 5: Reloading production model")
    load_defaults()
    success = load_production_model()

    if success:
        print("Production model loaded")
    else:
        print("Warning: Could not reload production model")

    print("\n" + "="*60)
    print("PIPELINE WITH DVC + DRIFT COMPLETED")
    print("="*60)

    response = {
        "status": "success",
        "message": f"Pipeline with DVC + Drift completed: Split {split_id}",
        "split_processed": split_id,
        "drift_check": {
            "enabled": True,
            "drift_detected": drift_result['dataset_drift'] if drift_result else None,
            "drift_percentage": f"{drift_result['share_of_drifted_columns']*100:.1f}%" if drift_result else "N/A",
            "threshold": f"{drift_threshold*100:.1f}%"
        },
        "pipeline_steps": {
            "split_creation": "completed",
            "dvc_tracking": dvc_status,
            "git_commit": git_commit,
            "drift_check": "drift_detected_training_triggered" if drift_result else "skipped_first_split",
            "model_training": "completed",
            "model_reload": "completed"
        },
        "current_production_model": model_info,
        "outputs": {
            "split_creation": split_output,
            "dvc_tracking": dvc_result.get("message", ""),
            "training": training_output
        }
    }

    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(endpoint="/pipeline/next-split-drift-detection-dvc", method="POST").observe(duration)
    api_requests_total.labels(endpoint="/pipeline/next-split-drift-detection-dvc", method="POST", status_code=status_code).inc()

    return response


# Endpoint Health Check
@app.get("/health")
async def health():
    """Health check"""

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mHealth Check Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    start_time = time.time()
    status_code = 200
    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(endpoint="/health", method="GET").observe(duration)
    api_requests_total.labels(endpoint="/health", method="GET", status_code=status_code).inc()

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version if model else None,
        "scaler_loaded": scaler is not None,
        "defaults_loaded": defaults is not None
    }



# Root  Endpoint API information
@app.get("/")
async def root():
    """API information"""

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mAPI Info Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")
    status_code = 200
    api_requests_total.labels(endpoint="/", method="GET", status_code=status_code).inc()

    return {
        "name": "Rain Prediction API",
        "description": "Predict rain in Australia using production MLflow model",
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model/info",
            "model_refresh": "POST /model/refresh",
            "predict_full": "POST /predict",
            "predict_simple": "POST /predict/simple",
            "train": "POST /train",
            "pipeline": "POST /pipeline/next-split",
            "pipeline_drift_detection": "POST /pipeline/next-split-drift-detection",
            "pipeline_dvc": "POST /pipeline/next-split-dvc", 
            "pipeline_drift_detection_dvc": "POST /pipeline/next-split-drift-detection-dvc",  
            "metrics": "GET /metrics"
        },
        "mlflow_ui": MLFLOW_TRACKING_URI,
        "features": {
            "simple_prediction": "Only 5 required fields (location, date, min_temp, max_temp, rain_today)",
            "auto_scaling": "Automatic feature scaling for input data.",
            "default_filling": "Missing features filled with training defaults when using simple prediction endpoint.",
            "conditional_training": "When using pipeline endpoint with drift detection."
        }
    }


# Endpoint model information
@app.get("/model/info")
async def get_model_info():
    """Get current production model information"""

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mModel Info Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")
    status_code = 200
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Use POST /model/refresh to load production model."
        )

    api_requests_total.labels(endpoint="/model/info", method="GET", status_code=status_code).inc()

    return {
        "status": "success",
        "model": model_info
    }


# Endpoint refresh model
@app.post("/model/refresh")
async def refresh_model():
    """Refresh/reload the production model from MLflow"""
    start_time = time.time()
    status_code = 200
    global model, model_version, model_info

    print("\n\033[1m--------------------\033[0m")     
    print("\n\033[1mModel Reload Endpoint:\033[0m")
    print("\n\033[1m--------------------\033[0m")

    print("Refreshing production model from MLflow.")
    
    load_defaults() 
    success = load_production_model()

    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(endpoint="/model/refresh", method="POST").observe(duration)
    api_requests_total.labels(endpoint="/model/refresh", method="POST", status_code=status_code).inc()

    if success:
        return {
            "status": "success",
            "message": "Production model reloaded",
            "model": model_info
        }
    else:
        # Clear cached model on failure
        model = None
        model_version = None
        model_info = {}

        raise HTTPException(
            status_code=500,
            detail="Failed to load production model from MLflow. All models may be archived or no model has been trained yet."
        )

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
