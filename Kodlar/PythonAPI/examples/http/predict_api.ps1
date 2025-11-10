# Set MODEL_URI before running server, e.g.:
# $env:MODEL_URI = "models:/credit-risk-model/Production"
# uvicorn src.api.predict_api:app --host 0.0.0.0 --port 8001 --reload

# Example predict request (dict-of-lists compatible with pandas)
$payload = @{ data = @{ feature1 = @(1.1, 2.2); feature2 = @(3.3, 4.4) } } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://localhost:8001/predict -ContentType application/json -Body $payload
