# =============================================================================
# STOCK BOT WATCHDOG — Auto-restart with crash recovery
# =============================================================================
# This script runs the trading bot in a loop. If the bot crashes or exits,
# it waits 30 seconds and restarts. Designed to be run by Windows Task Scheduler.
#
# Usage (manual):   powershell -ExecutionPolicy Bypass -File watchdog.ps1
# Usage (Task Sched): Registered automatically by register_task.ps1
# =============================================================================

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# --- Find Python ---
$Candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "C:\Python313\python.exe",
    "C:\Python312\python.exe"
)
$Py = $null
foreach ($c in $Candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) { $Py = $c; break }
}
if (-not $Py) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Py = $cmd.Source }
}
if (-not $Py) { Write-Error "Python not found"; exit 1 }

$LogFile = Join-Path $Root "watchdog.log"
$MaxRestarts = 50  # Max restarts per day before giving up
$RestartDelay = 30 # Seconds to wait before restart

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts - $msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

# --- Kill any existing bot processes ---
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'stock_agent_v3.*live') } |
    ForEach-Object {
        Write-Log "Stopping existing bot PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 2

# --- Watchdog loop ---
$restartCount = 0
$dayStart = (Get-Date).Date

Write-Log "=== WATCHDOG STARTED ==="
Write-Log "Python: $Py"
Write-Log "Working dir: $Root"

while ($restartCount -lt $MaxRestarts) {
    # Reset counter at midnight
    if ((Get-Date).Date -ne $dayStart) {
        $restartCount = 0
        $dayStart = (Get-Date).Date
        Write-Log "New day — restart counter reset"
    }

    $restartCount++
    Write-Log "Starting bot (attempt #$restartCount today)..."

    # Run the bot synchronously — this blocks until the bot exits
    $proc = Start-Process -FilePath $Py `
        -ArgumentList @("-u", "stock_agent_v3.py", "live") `
        -WorkingDirectory $Root `
        -NoNewWindow `
        -PassThru `
        -Wait

    $exitCode = $proc.ExitCode
    Write-Log "Bot exited with code $exitCode"

    # If it's outside market hours (before 9:30 or after 16:30 ET), don't restart
    $now = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), 'Eastern Standard Time')
    $hour = $now.Hour
    $minute = $now.Minute
    $dow = $now.DayOfWeek

    if ($dow -eq 'Saturday' -or $dow -eq 'Sunday') {
        Write-Log "Weekend — sleeping until Monday 9:00 AM ET"
        # Calculate sleep until next Monday 9:00 AM
        $daysUntilMonday = if ($dow -eq 'Saturday') { 2 } else { 1 }
        $nextMonday = $now.Date.AddDays($daysUntilMonday).AddHours(9)
        $sleepSeconds = [math]::Max(60, ($nextMonday - $now).TotalSeconds)
        Start-Sleep -Seconds $sleepSeconds
        continue
    }

    if ($hour -ge 17) {
        Write-Log "After market hours — sleeping until 9:00 AM ET tomorrow"
        $tomorrow9am = $now.Date.AddDays(1).AddHours(9)
        $sleepSeconds = [math]::Max(60, ($tomorrow9am - $now).TotalSeconds)
        Start-Sleep -Seconds $sleepSeconds
        continue
    }

    # During market hours — restart after delay
    Write-Log "Restarting in $RestartDelay seconds..."
    Start-Sleep -Seconds $RestartDelay
}

Write-Log "=== WATCHDOG: Max restarts ($MaxRestarts) reached today. Stopping. ==="
