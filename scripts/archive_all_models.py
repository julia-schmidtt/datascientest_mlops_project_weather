#!/usr/bin/env python3
"""
Archive all models in MLflow Model Registry.
This prepares for a clean start of automated pipeline.
"""

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import sys

# Load credentials
load_dotenv()
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')

# Connect to MLflow
mlflow.set_tracking_uri("https://dagshub.com/julia-schmidtt/datascientest_mlops_project_weather.mlflow")
client = MlflowClient()

MODEL_NAME = "RainTomorrow_XGBoost"

print("\n" + "="*60)
print("ARCHIVE ALL MODELS")
print("="*60)

try:
    # Get all versions
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print(f"\nFound {len(versions)} model versions")
    
    if len(versions) == 0:
        print("No models found to archive.")
        sys.exit(0)
    
    # Archive all
    print("\nArchiving all versions...")
    print("-" * 60)
    
    for v in versions:
        current_stage = v.current_stage if v.current_stage else "None"
        print(f"Version {v.version:2s}: {current_stage:12s} → Archived")
        
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived"
        )
    
    print("-" * 60)
    print(f"\nSuccessfully archived {len(versions)} model versions!")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\nError: {e}\n")
    sys.exit(1)
