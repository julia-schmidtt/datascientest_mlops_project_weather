#!/bin/bash
# Automated Pipeline with Drift Detection: Process Next Training Data Split
# Usage: ./scripts/process_next_split_with_drift.sh

echo "=========================================================="
echo "Automated ML Pipeline with Drift Detection"
echo "Weather Prediction Australia"
echo "=========================================================="

# Change to project directory
cd "$(dirname "$0")/.."

# Activate virtual environment
#source venv/bin/activate

# Call API endpoint with drift detection
echo ""
echo "[INFO] Calling /pipeline/next-split-drift-detection-dvc endpoint"
echo "[INFO] Drift threshold: 2% (configured in params.yaml)"
echo ""

response=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/pipeline/next-split-drift-detection-dvc)

# Extract status code
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

# Check response
if [ "$http_code" = "200" ]; then
    echo "[SUCCESS] Pipeline completed successfully"
    echo ""
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    exit_code=0
else
    echo "[ERROR] Pipeline failed with HTTP $http_code"
    echo "$body"
    exit_code=1
fi

echo ""
echo "=========================================================="
echo "ML Pipeline with Drift Detection completed"
echo "=========================================================="

exit $exit_code
