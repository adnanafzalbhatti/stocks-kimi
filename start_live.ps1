# Start / restart Telegram live agent (full universe by default)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

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

Write-Host "Working dir: $Root"
Write-Host "Python: $Py"

# Stop any existing stock_agent_v3 live processes
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'stock_agent_v3') } |
    ForEach-Object {
        Write-Host "Stopping PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Start-Sleep -Seconds 1

# Compile check
& $Py -m py_compile "$Root\stock_agent_v3.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compile failed"
    exit 1
}

$stdout = Join-Path $Root "live_stdout.log"
$stderr = Join-Path $Root "live_stderr.log"
if (Test-Path $stdout) { Remove-Item $stdout -Force -ErrorAction SilentlyContinue }
if (Test-Path $stderr) { Remove-Item $stderr -Force -ErrorAction SilentlyContinue }

$p = Start-Process -FilePath $Py `
    -ArgumentList @("-u", "stock_agent_v3.py", "live") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started live PID=$($p.Id)"
Start-Sleep -Seconds 10
Write-Host "--- STDOUT ---"
if (Test-Path $stdout) { Get-Content $stdout -ErrorAction SilentlyContinue }
Write-Host "--- STDERR (tail) ---"
if (Test-Path $stderr) { Get-Content $stderr -Tail 25 -ErrorAction SilentlyContinue }
Write-Host "Done. Keep PC awake. Check Telegram for boot + heartbeat."
