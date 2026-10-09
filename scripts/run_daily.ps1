# Daily: collect news, tag tickers, refresh SEC filings and recent prices.
# Exit codes from collect_daily: 0 ok, 2 partial (data kept), 1 failed.
#
# The scheduled task fires whenever the laptop is plugged in (and at a backup
# time), so this may be started several times a day. It runs at most once per
# $MinHours: a successful run leaves a marker, and later starts exit quietly.
# A start while another collection is still running also exits quietly
# instead of failing on the database lock.
param([double]$MinHours = 16, [switch]$Force)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$marker = "data\last_daily_ok.txt"

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*jobs.collect_daily*" }
if ($running) { Write-Host "A collection is already running; skipping."; exit 0 }

if (-not $Force -and (Test-Path $marker)) {
    $last = [datetime]::Parse((Get-Content $marker -Raw).Trim(), [Globalization.CultureInfo]::InvariantCulture)
    if (((Get-Date) - $last).TotalHours -lt $MinHours) {
        Write-Host "Last successful collection was at $last; skipping."; exit 0
    }
}

$started = Get-Date
& .\.venv\Scripts\python.exe -m jobs.collect_daily
$code = $LASTEXITCODE
if ($code -in 0, 2) {
    Set-Content $marker $started.ToString("s", [Globalization.CultureInfo]::InvariantCulture)
}
exit $code
