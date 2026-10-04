# Daily: collect posts (Reddit, news), tag tickers, refresh recent prices.
# Exit codes from collect_daily: 0 ok, 2 partial (data kept), 1 failed.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe -m jobs.collect_daily
exit $LASTEXITCODE
