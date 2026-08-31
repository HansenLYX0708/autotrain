$ErrorActionPreference = "Stop"
$PidFile = Join-Path $PSScriptRoot "data\autotrain.pid"
if (-not (Test-Path $PidFile)) {
    Write-Host "AutoTrain is not running."
    exit 0
}
$ServerPid = [int](Get-Content $PidFile -Raw)
$Process = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
if ($Process) {
    Stop-Process -Id $ServerPid
    [void]$Process.WaitForExit(10000)
}
Remove-Item $PidFile -Force
Write-Host "AutoTrain stopped."
