param(
    [string]$InstallRoot = "C:\AutoTrain",
    [switch]$NoStart
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
$PackageRoot = $PSScriptRoot
$Payload = Join-Path $PackageRoot "payload"
$SourceApp = Join-Path $Payload "app"
$TargetApp = Join-Path $InstallRoot "app"
if (-not (Test-Path (Join-Path $SourceApp "server.js") -PathType Leaf)) { throw "Upgrade payload app/server.js is missing." }
if (-not (Test-Path $TargetApp -PathType Container)) { throw "AutoTrain app directory was not found: $TargetApp" }
$StopScript = Join-Path $InstallRoot "Stop-AutoTrain.cmd"
if (Test-Path $StopScript -PathType Leaf) { & $StopScript }
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupRoot = Join-Path $InstallRoot "upgrade-backups\$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
Copy-Item $TargetApp (Join-Path $BackupRoot "app") -Recurse -Force
Copy-Item (Join-Path $SourceApp "*") $TargetApp -Recurse -Force
foreach ($File in @("Start.ps1", "Stop.ps1", "Test-Installation.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "configure_db.py")) {
    $Source = Join-Path $Payload $File
    if (Test-Path $Source -PathType Leaf) { Copy-Item $Source (Join-Path $InstallRoot $File) -Force }
}
if (-not $NoStart) { & (Join-Path $InstallRoot "Start-AutoTrain.cmd") }
Write-Host "AutoTrain upgrade applied. Backup: $BackupRoot" -ForegroundColor Green
