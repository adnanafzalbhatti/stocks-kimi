$ErrorActionPreference = "Stop"
$TaskName = "StockBotWatchdog"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WatchdogScript = Join-Path $Root "watchdog.ps1"

if (-not (Test-Path $WatchdogScript)) {
    Write-Error "watchdog.ps1 not found at $WatchdogScript"
    exit 1
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File $WatchdogScript" -WorkingDirectory $Root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -DontStopOnIdleEnd -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Stock Trading AI Agent Bot" -Force

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "  Task Name: $TaskName"
Write-Host "  Trigger:   At logon for $env:USERNAME"
Write-Host "  Script:    $WatchdogScript"
Write-Host ""
Write-Host "Run now:     Start-ScheduledTask -TaskName $TaskName"
Write-Host "Check:       Get-ScheduledTask -TaskName $TaskName"
Write-Host "Remove:      Unregister-ScheduledTask -TaskName $TaskName"
