# Weekly (weekend, after Friday's US close): refresh the universe, rank the
# week, then update the lead-lag/backtest report. The ranking is what matters,
# so a universe refresh failure doesn't stop it (the previous universe is kept).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

& $py -m jobs.build_universe
if ($LASTEXITCODE -ne 0) { Write-Warning "build_universe failed (exit $LASTEXITCODE); keeping the existing universe" }

& $py -m jobs.rank_weekly
$rank = $LASTEXITCODE
if ($rank -ne 0) { Write-Error "rank_weekly failed (exit $rank)" }

& $py -m jobs.run_analysis --refresh-prices
if ($LASTEXITCODE -ne 0) { Write-Warning "run_analysis failed (exit $LASTEXITCODE)" }

exit $rank
