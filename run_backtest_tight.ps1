$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$Py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
if (-not (Test-Path -LiteralPath $Py)) { $Py = "python" }
Write-Host "Running FULL universe TIGHT backtest (conf>=85)..."
& $Py -u stock_agent_v3.py backtest --all-tickers --start 2024-01-01 --end 2026-07-01 *>&1 |
    Tee-Object -FilePath backtest_v33_tight_run.log
Write-Host "Done. See backtest_v33_tight_run.log and backtest_v33_full_tight_results.csv"
