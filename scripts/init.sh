#!/bin/bash
set -e

rm -rf data/automated_splits/split_*
rm -f data/automated_splits/metadata.yaml

echo "Archive all models on MLflow..."
python scripts/archive_all_models.py

echo "Generate all preprocessing outputs..."
python src/data/preprocess.py

echo "Generate manually training splits..."
python src/data/training_data_splits_by_year.py

# Start the application
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
