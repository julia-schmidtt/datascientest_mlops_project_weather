#!/bin/bash
set -e

# Load environment variables
#if [ -f .env ]; then
#    export $(cat .env | grep -v '^#' | xargs)
#fi

# Clean up automated splits data
#rm -rf data/automated_splits/split_*
#rm -f data/automated_splits/metadata.yaml

#echo "Archive all models on MLflow..."
#python scripts/archive_all_models.py

echo "Generate all preprocessing outputs..."
python src/data/preprocess.py


# Start the application
echo "Starting API..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
