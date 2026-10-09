# Weekly (weekend, after Friday's US close): refresh the universe, rank the
# week, then update the lead-lag/backtest report. The ranking is what matters,
# so a universe refresh failure doesn't stop it (the previous universe is kept).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

# Make sure the week's last news is in: collect again unless a collection has
# finished since the week's cutoff (Saturday 00:00 UTC). Wait for any
# collection already running, since ranking needs the database too.
$utc = [DateTime]::UtcNow
$cutoff = $utc.Date.AddDays(-(([int]$utc.DayOfWeek + 1) % 7))
& .\scripts\run_daily.ps1 -MinHours ($utc - $cutoff).TotalHours
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*jobs.collect_daily*" }
if ($busy) { Wait-Process -Id $busy.ProcessId -ErrorAction SilentlyContinue }

& $py -m jobs.build_universe
if ($LASTEXITCODE -ne 0) { Write-Warning "build_universe failed (exit $LASTEXITCODE); keeping the existing universe" }

& $py -m jobs.rank_weekly
$rank = $LASTEXITCODE
if ($rank -ne 0) { Write-Error "rank_weekly failed (exit $rank)" }

& $py -m jobs.run_analysis --refresh-prices
if ($LASTEXITCODE -ne 0) { Write-Warning "run_analysis failed (exit $LASTEXITCODE)" }

exit $rank
