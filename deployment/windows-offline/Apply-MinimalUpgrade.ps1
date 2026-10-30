param(
    [string]$InstallRoot = "C:\AutoTrain",
    [string]$UserConfigsPath = "",
    [switch]$NoStart
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
$PackageRoot = $PSScriptRoot
$Payload = Join-Path $PackageRoot "payload"
$SourceApp = Join-Path $Payload "app"
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$TargetApp = Join-Path $InstallRoot "app"
if (-not (Test-Path (Join-Path $SourceApp "server.js") -PathType Leaf)) { throw "Upgrade payload app/server.js is missing." }
if (-not (Test-Path $TargetApp -PathType Container)) { throw "AutoTrain app directory was not found: $TargetApp" }
# Verify the extracted package before stopping the installed app.
$ChecksumFile = Join-Path $PackageRoot "SHA256SUMS.txt"
if (-not (Test-Path $ChecksumFile -PathType Leaf)) { throw "Package checksums are missing." }
foreach ($Line in Get-Content -LiteralPath $ChecksumFile -Encoding UTF8) {
    if ($Line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw "Invalid checksum entry." }
    $Expected = $Matches[1]
    $File = [IO.Path]::GetFullPath((Join-Path $PackageRoot $Matches[2]))
    if (-not $File.StartsWith($PackageRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Invalid package path." }
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing package file: $File" }
    if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -ne $Expected) { throw "Checksum mismatch: $File" }
}
if (-not $UserConfigsPath) {
    $Database = Join-Path $InstallRoot "data\autotrain.db"
    if (Test-Path $Database -PathType Leaf) {
        $Node = Join-Path $InstallRoot "runtime\node\node.exe"
        if (-not (Test-Path $Node -PathType Leaf)) { throw "Installed Node.js was not found: $Node" }
        $UserConfigsPath = (& $Node (Join-Path $PackageRoot "Read-ConfigPath.mjs") $Database $SourceApp | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Cannot read the installed config path. Supply -UserConfigsPath explicitly." }
    }
    if (-not $UserConfigsPath) { $UserConfigsPath = Join-Path $InstallRoot "data\configs" }
}
if (-not [IO.Path]::IsPathRooted($UserConfigsPath)) { throw "UserConfigsPath must be absolute." }
$Defaults = Join-Path $Payload "default-configs"
if (-not (Test-Path (Join-Path $Defaults "inventory.json"))) { throw "Default config inventory is missing." }
$StopScript = Join-Path $InstallRoot "Stop-AutoTrain.cmd"
if (Test-Path $StopScript -PathType Leaf) {
    & $StopScript
    if ($LASTEXITCODE -ne 0) { throw "Failed to stop the installed app." }
}
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupRoot = Join-Path $InstallRoot "upgrade-backups\$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
Copy-Item $TargetApp (Join-Path $BackupRoot "app") -Recurse -Force
Copy-Item (Join-Path $Defaults "inventory.json") (Join-Path $BackupRoot "default-config-inventory.json")
Copy-Item (Join-Path $SourceApp "*") $TargetApp -Recurse -Force
foreach ($File in @("Start.ps1", "Stop.ps1", "Test-Installation.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "configure_db.py")) {
    $Source = Join-Path $Payload $File
    if (Test-Path $Source -PathType Leaf) {
        $Target = Join-Path $InstallRoot $File
        if (Test-Path $Target -PathType Leaf) { Copy-Item $Target (Join-Path $BackupRoot $File) }
        Copy-Item $Source $Target -Force
    }
}
$Added = @()
$Preserved = 0
foreach ($File in Get-ChildItem $Defaults -Recurse -File | Where-Object { $_.Extension -in ".yaml", ".yml" }) {
    $Relative = $File.FullName.Substring($Defaults.Length + 1)
    $Target = Join-Path $UserConfigsPath $Relative
    if (Test-Path -LiteralPath $Target) { $Preserved++; continue }
    New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $Target
    $Added += $Target
}
[IO.File]::WriteAllLines((Join-Path $BackupRoot "added-default-configs.txt"), [string[]]$Added, (New-Object Text.UTF8Encoding($false)))
Write-Host "Default configs: added $($Added.Count), preserved $Preserved. Directory: $UserConfigsPath"
if (-not $NoStart) {
    & (Join-Path $InstallRoot "Start-AutoTrain.cmd")
    if ($LASTEXITCODE -ne 0) { throw "App restart failed. Check installation logs. Backup: $BackupRoot" }
}
Write-Host "AutoTrain upgrade applied. Backup: $BackupRoot" -ForegroundColor Green
