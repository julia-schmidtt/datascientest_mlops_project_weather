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
from datetime import datetime


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
    print("\nProcessing versions...")
    print("-" * 60)
    
    archived_count = 0
    already_archived_count = 0
    
    for v in versions:
        current_stage = v.current_stage if v.current_stage else "None"

        # Check if already archived
        if current_stage == "Archived":
            print(f"Version {v.version:2s}: Already archived - checking tags")
            already_archived_count += 1

            # Get existing tags
            existing_tags = v.tags if v.tags else {}
    
            # Only set archived tag if it doesn't exist
            if "archived" not in existing_tags:
               client.set_model_version_tag(
                   name=MODEL_NAME,
                   version=v.version,
                   key="archived",
                   value="true"
               )
               print('Archived tag added.')


            # Only set archived_at if it doesn't exist yet
            if "archived_at" not in existing_tags:
                client.set_model_version_tag(
                    name=MODEL_NAME,
                    version=v.version,
                    key="archived_at",
                    value=datetime.now().isoformat()
                )
                print('Archived at tag added.')
            
            continue
        
        # Archive  models that aren't archived yet
        print(f"Version {v.version:2s}: {current_stage:12s} → Archived (tags added in MLflow)")
        
        
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived"
        )

        # Add archived tag
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=v.version,
            key="archived",
            value="true"
        )
        
        # Add timestamp tag
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=v.version,
            key="archived_at",
            value=datetime.now().isoformat()

        )

        archived_count += 1
    
    print("-" * 60)
    print(f"  Total versions: {len(versions)}")
    print(f"  Newly archived: {archived_count}")
    print(f"  Already archived: {already_archived_count}")

    total_processed = archived_count + already_archived_count
    if total_processed > 0:
        print(f"\nSuccessfully processed {total_processed} model version(s)!")
    else:
        print(f"\nNo models needed processing")

    print("="*60 + "\n")
    


except Exception as e:
    print(f"\nError: {e}\n")
    sys.exit(1)
