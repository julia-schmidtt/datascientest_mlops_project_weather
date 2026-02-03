#!/bin/bash
set -e


# Clean up automated splits data
rm -rf data/automated_splits/split_*
rm -f data/automated_splits/metadata.yaml
mkdir -p data/automated_splits
echo "[INFO] Cleaning old splits..."


# Clean up DVC tracked splits
echo "[INFO] Cleaning old splits in DVC repo..."
if [ -d "/app/dvc-tracking/data/automated_splits" ]; then
    cd /app/dvc-tracking
    rm -rf data/automated_splits/split_*
    rm -f data/automated_splits/*.dvc
    rm -f data/automated_splits/.gitignore
    
    # Git commit the deletion (if there's anything to commit)
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add data/automated_splits/
        git commit -m "Clean up old splits" || echo "Nothing to commit"
        git push || echo "Push failed or nothing to push"
    fi
    
    cd /app
fi


#echo "Archive all models on MLflow..."
python scripts/archive_all_models.py

echo "[INFO] Generate preprocessing outputs..."
python src/data/preprocess.py


# Start API
echo "[INFO] Start API..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
