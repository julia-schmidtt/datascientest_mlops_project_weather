#!/bin/bash
# Automated Pipeline: Process Next Training Data Split
# Usage: ./scripts/process_next_split.sh

echo "=========================================================="
echo "Automated ML Pipeline - Weather Prediction Australia"
echo "=========================================================="

# Change to project directory
cd "$(dirname "$0")/.."

# Activate virtual environment
#source venv/bin/activate

# Call API endpoint
echo ""
echo "[INFO] Calling /pipeline/next-split endpoint."
response=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/pipeline/next-split)

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
echo "ML Pipeline completed"
echo "=========================================================="

exit $exit_code
