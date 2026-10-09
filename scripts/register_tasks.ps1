# Registers two Windows scheduled tasks for the current user:
#   Sentimental-Daily   whenever the laptop is plugged in, plus 13:00 as a backup.
#                       run_daily.ps1 collects at most once every 16 hours, so
#                       extra starts just exit.
#   Sentimental-Weekly  Saturdays at 10:00 local time
# They run only while you're logged in (no stored password), on battery too,
# and catch up if the PC was off at the scheduled time.
# Remove with:  .\scripts\register_tasks.ps1 -Remove
param([switch]$Remove)
$root = Split-Path $PSScriptRoot -Parent
$names = "Sentimental-Daily", "Sentimental-Weekly"

if ($Remove) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Host "Removed: $($names -join ', ')"
    exit 0
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 30)

function New-JobAction($script) {
    New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\$script`"" -WorkingDirectory $root
}

# "Power source changed to AC" (Kernel-Power event 105 with AcOnline = true),
# delayed a few minutes so the network is back after waking from sleep.
$eventClass = Get-CimClass -ClassName MSFT_TaskEventTrigger -Namespace Root/Microsoft/Windows/TaskScheduler
$pluggedIn = New-CimInstance -CimClass $eventClass -ClientOnly
$pluggedIn.Enabled = $true
$pluggedIn.Delay = "PT3M"
$pluggedIn.Subscription = @"
<QueryList><Query Id="0" Path="System"><Select Path="System">
*[System[Provider[@Name='Microsoft-Windows-Kernel-Power'] and EventID=105]]
and *[EventData[Data[@Name='AcOnline']='true']]
</Select></Query></QueryList>
"@

Register-ScheduledTask -TaskName "Sentimental-Daily" -Action (New-JobAction "run_daily.ps1") `
    -Trigger @($pluggedIn, (New-ScheduledTaskTrigger -Daily -At 1:00pm)) -Settings $settings `
    -Description "Sentimental: daily collection (on plug-in, at most every 16 hours)" -Force | Out-Null
Register-ScheduledTask -TaskName "Sentimental-Weekly" -Action (New-JobAction "run_weekly.ps1") `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00am) -Settings $settings `
    -Description "Sentimental: weekly universe, ranking and analysis" -Force | Out-Null

Get-ScheduledTask -TaskName $names | Select-Object TaskName, State
