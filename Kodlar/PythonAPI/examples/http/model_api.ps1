# Start server:
# uvicorn src.api.model_api:app --host 0.0.0.0 --port 8002 --reload

# Train + log only
$body1 = @{ 
	experiment_name = "ModelSaveDemo"; 
	register = $false 
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8002/models/train-and-log -ContentType application/json -Body $body1

# Train + log + register and transition to Staging
$body2 = @{ 
	experiment_name = "ModelSaveDemo"; 
	model_name = "credit-risk-model"; 
	register = $true; 
	stage = "Staging" 
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8002/models/train-and-log -ContentType application/json -Body $body2
