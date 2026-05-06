# start_all_bots.ps1
# This script launches multiple trading bots simultaneously in their own separate terminal windows.

Write-Host "Starting XAUUSD Bot (min_prob=0.55)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", " `$env:SYMBOL='XAUUSD.sim'; `$env:SOTA_MODEL_PATH='train_pipeline/models_sota/patchtst_primary.pt'; `$env:SOTA_CONFIG_PATH='train_pipeline/models_sota/sota_config.json'; python live_sota_trading.py --min-prob 0.55"

Start-Sleep -Seconds 2

Write-Host "Starting GBPUSD Bot (min_prob=0.79)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", " `$env:SYMBOL='GBPUSD'; `$env:SOTA_MODEL_PATH='train_pipeline/reports/gbpusd/patchtst_primary.pt'; `$env:SOTA_CONFIG_PATH='train_pipeline/reports/gbpusd/sota_config.json'; python live_sota_trading.py --min-prob 0.77"

Write-Host "Both bots are now running in separate windows!"
