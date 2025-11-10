# Create run
$body = @{ experiment_name = "DemoExp"; run_name = "api-run-1" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/runs/create -ContentType application/json -Body $body

# Replace with actual run_id from the previous response
$runId = "<RUN_ID>"

# Log param
$paramBody = @{ run_id = $runId; key = "lr"; value = "0.01" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/runs/log-param -ContentType application/json -Body $paramBody

# Log metric
$metricBody = @{ run_id = $runId; key = "loss"; value = 0.42; step = 1 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/runs/log-metric -ContentType application/json -Body $metricBody
