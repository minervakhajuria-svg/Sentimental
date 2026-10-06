# Registers two Windows scheduled tasks for the current user:
#   Sentimental-Daily   every day at 21:00 local time (the laptop sleeps overnight)
#   Sentimental-Weekly  Saturdays at 10:00 local time
# They run only while you're logged in (no stored password) and catch up if the
# PC was off at the scheduled time, on battery too. Remove with:  .\scripts\register_tasks.ps1 -Remove
param([switch]$Remove)
$root = Split-Path $PSScriptRoot -Parent
$names = "Sentimental-Daily", "Sentimental-Weekly"

if ($Remove) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Host "Removed: $($names -join ', ')"
    exit 0
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 30)

function New-JobAction($script) {
    New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\$script`"" -WorkingDirectory $root
}

Register-ScheduledTask -TaskName "Sentimental-Daily" -Action (New-JobAction "run_daily.ps1") `
    -Trigger (New-ScheduledTaskTrigger -Daily -At 9:00pm) -Settings $settings `
    -Description "Sentimental: daily collection" -Force | Out-Null
Register-ScheduledTask -TaskName "Sentimental-Weekly" -Action (New-JobAction "run_weekly.ps1") `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00am) -Settings $settings `
    -Description "Sentimental: weekly universe, ranking and analysis" -Force | Out-Null

Get-ScheduledTask -TaskName $names | Select-Object TaskName, State
