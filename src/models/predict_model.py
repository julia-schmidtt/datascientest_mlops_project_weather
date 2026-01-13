"""
Prediction module for rain prediction in Australia.
Loads production model from MLflow and makes predictions.
Loads scaler and default validation data for feature processing and validation of prediction input.

Usage:
    from src.models.predict_model import RainPredictor
    
    predictor = RainPredictor()
    prediction = predictor.predict_simple(
        location="Sydney",
        date="2025-01-15",
        min_temp=18.0,
        max_temp=28.0,
        rainfall=0.0,
        rain_today=0
    )
"""

import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS


class RainPredictor:
    def __init__(self):
        """Initialize predictor with model, scaler, and defaults"""
        self.model = None
        self.model_version = None
        self.model_info = {}
        self.scaler = None
        self.defaults = None
        self.validation_data = None

        # MLflow configuration
        self.MLFLOW_TRACKING_URI = PARAMS['mlflow']['tracking_uri']
        self.MODEL_NAME = "RainTomorrow_XGBoost"
        
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        
        # Load everything
        self._load_defaults()
        self._load_production_model()
    
    

    def _load_defaults(self):
        """Load preprocessing defaults and scaler"""
        try:
            # Load defaults
            with open('src/api/defaults.json', 'r') as f:
                self.defaults = json.load(f)
            
            # Load scaler
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load validation data
            with open('src/api/validation_data.json', 'r') as f:
                self.validation_data = json.load(f)
            
            print("Defaults, scaler, and validation data loaded")
            return True
            
        except Exception as e:
            print(f"Error loading defaults/scaler: {e}")
            raise    

    
    def _load_production_model(self):
        """Load current production model from MLflow"""
        try:
            client = MlflowClient()
            
            # Get production model
            versions = client.search_model_versions(f"name='{self.MODEL_NAME}'")
            prod_versions = [v for v in versions if v.current_stage == "Production"]
            
            if not prod_versions:
                raise Exception("No production model found in MLflow")
            
            # Get latest production version
            prod_version = max(prod_versions, key=lambda x: int(x.version))
            self.model_version = prod_version.version
            
            # Load model
            model_uri = f"models:/{self.MODEL_NAME}/Production"
            self.model = mlflow.xgboost.load_model(model_uri)
            
            # Get model info
            run = client.get_run(prod_version.run_id)
            self.model_info = {
                "model_name": self.MODEL_NAME,
                "version": self.model_version,
                "stage": "Production",
                "run_id": prod_version.run_id,
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
            
            print(f"Loaded production model: Version {self.model_version}")
            print(f"F1 Score: {self.model_info['metrics']['f1_score']:.4f}")
            print(f"Trained on: {self.model_info['params']['years']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading production model: {e}")
            raise

    def _preprocess_simple_input(
        self,
        location: str,
        date: str,
        min_temp: float,
        max_temp: float,
        rain_today: int,
        **optional_features
    ) -> pd.DataFrame:
        """
        Preprocess simplified input into model-ready features.
        
        Args:
            location: Weather station location
            date: Date in YYYY-MM-DD format
            min_temp: Minimum temperature
            max_temp: Maximum temperature
            rain_today: 0=No, 1=Yes
            **optional_features: Optional weather features
        
        Returns:
            Preprocessed DataFrame with 110 features
        """

        # Validate location
        if location not in self.validation_data['locations']:
            raise ValueError(
                f"Invalid location '{location}'. "
                f"Must be one of: {', '.join(self.validation_data['locations'][:10])}..."
            )
        
        # Extract date features
        date_obj = datetime.strptime(date, "%Y-%m-%d")
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
            'MinTemp': min_temp,
            'MaxTemp': max_temp,
            'RainToday': rain_today,
            'WindGustSpeed': optional_features.get('wind_gust_speed') or self.defaults['train_medians']['WindGustSpeed'],
            'WindSpeed9am': optional_features.get('wind_speed_9am') or self.defaults['train_medians']['WindSpeed9am'],
            'WindSpeed3pm': optional_features.get('wind_speed_3pm') or self.defaults['train_medians']['WindSpeed3pm'],
            'Humidity9am': optional_features.get('humidity_9am') or self.defaults['train_medians']['Humidity9am'],
            'Humidity3pm': optional_features.get('humidity_3pm') or self.defaults['train_medians']['Humidity3pm'],
            'Pressure9am': optional_features.get('pressure_9am') or self.defaults['train_medians']['Pressure9am'],
            'Pressure3pm': optional_features.get('pressure_3pm') or self.defaults['train_medians']['Pressure3pm'],
            'Temp9am': optional_features.get('temp_9am') or self.defaults['train_medians']['Temp9am'],
            'Temp3pm': optional_features.get('temp_3pm') or self.defaults['train_medians']['Temp3pm'],
            'Rainfall': optional_features.get('rainfall') or self.defaults['train_medians']['Rainfall'],
            'Year': year,
            'Location': location,
            'WindGustDir': optional_features.get('wind_gust_dir') or self.defaults['train_modes']['WindGustDir'],
            'WindDir9am': optional_features.get('wind_dir_9am') or self.defaults['train_modes']['WindDir9am'],
            'WindDir3pm': optional_features.get('wind_dir_3pm') or self.defaults['train_modes']['WindDir3pm'],
            'Season': season
        }

        # Create DataFrame
        df = pd.DataFrame([features_raw])
        
        # One-hot encode categoricals
        categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Season']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Get expected columns from training
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
        numerical_cols = [
            'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'
        ]
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df


    def predict_simple(
        self,
        location: str,
        date: str,
        min_temp: float,
        max_temp: float,
        rain_today: int,
        **optional_features
    ) -> Dict[str, Any]:
        """
        Make simplified prediction with minimal required features.
        
        Args:
            location: Weather station location
            date: Date in YYYY-MM-DD format
            min_temp: Minimum temperature (°C)
            max_temp: Maximum temperature (°C)
            rain_today: 0=No, 1=Yes
            **optional_features: Optional weather features
        
        Returns:
            Dictionary with prediction results
        """

        # Preprocess input
        df = self._preprocess_simple_input(
            location=location,
            date=date,
            min_temp=min_temp,
            max_temp=max_temp,
            rain_today=rain_today,
            **optional_features
        )
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        # Extract season for metadata
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        season_map = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        }
        season = season_map[date_obj.month]
        
        return {
            "prediction": int(prediction),
            "label": "Rain Tomorrow" if prediction == 1 else "No Rain Tomorrow",
            "probability_rain": float(probability),
            "probability_no_rain": float(1 - probability),
            "confidence": "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low",
            "inputs_used": {
                "location": location,
                "date": date,
                "season": season,
                "year": date_obj.year,
                "rain_today": rain_today
            },
            "model_version": self.model_version,
            "model_f1_score": round(self.model_info['metrics']['f1_score'], 4)
        }
    

    def predict_full(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with full 110 features.
        
        Args:
            features: Dictionary with all 110 features
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise Exception("Model not loaded. Initialize predictor first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            "prediction": int(prediction),
            "label": "Rain" if prediction == 1 else "No Rain",
            "probability_rain": float(probability),
            "probability_no_rain": float(1 - probability),
            "model_version": self.model_version,
            "model_f1_score": round(self.model_info['metrics']['f1_score'], 4)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current production model"""
        return self.model_info

    def reload_model(self):
        """Reload production model from MLflow"""
        self._load_production_model()
        print("Model reloaded")

# Function for standalone usage
def predict_rain(
    location: str,
    date: str,
    min_temp: float,
    max_temp: float,
    rain_today: int = 0,
    **optional_features
) -> Dict[str, Any]:
    """
    Standalone prediction function.
    
    Usage:
        from src.models.predict_model import predict_rain
        
        result = predict_rain(
            location="Sydney",
            date="2025-01-15",
            min_temp=18.0,
            max_temp=28.0,
            rain_today=0
        )
    """

    predictor = RainPredictor()
    return predictor.predict_simple(
        location=location,
        date=date,
        min_temp=min_temp,
        max_temp=max_temp,
        rain_today=rain_today,
        **optional_features
    )


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')


    print("=== Rain Predictor Test ===\n")

    # Test 1: Simplified prediction
    print("Test 1: Sydney Summer")
    result = predict_rain(
        location="Sydney",
        date="2025-01-15",
        min_temp=18.0,
        max_temp=28.0,
        rain_today=0
    )
    print(f"Prediction: {result['label']}")
    print(f"Probability: {result['probability_rain']:.1%}")

    # Test 2: With optional features
    print("Test 2: Melbourne Winter")
    result = predict_rain(
        location="Melbourne",
        date="2025-06-20",
        min_temp=8.0,
        max_temp=15.0,
        rain_today=1,
        humidity_9am=85.0,
        humidity_3pm=70.0
    )
    print(f"Prediction: {result['label']}")
    print(f"Probability: {result['probability_rain']:.1%}")
