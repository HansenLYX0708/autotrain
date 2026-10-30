param(
    [string]$OutputDirectory = ".\dist",
    [string]$Timestamp = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$DefaultConfigsPath = "",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
$ScriptRoot = $PSScriptRoot
$AppSource = [IO.Path]::GetFullPath((Join-Path $ScriptRoot "..\.."))
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$PackageName = "AutoTrain-Minimal-Upgrade-$Timestamp"
$PackageRoot = Join-Path $OutputDirectory $PackageName
$Payload = Join-Path $PackageRoot "payload"
if (Test-Path $PackageRoot) { throw "Output exists: $PackageRoot" }
New-Item -ItemType Directory -Force -Path (Join-Path $Payload "app") | Out-Null
if (-not $SkipBuild) {
    Push-Location $AppSource
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Production build failed with exit code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }
}
Push-Location $AppSource
try {
    # Also run this with -SkipBuild: `next build` alone omits static/public assets.
    & node.exe copy.js
    if ($LASTEXITCODE -ne 0) { throw "Copying web assets failed." }
} finally { Pop-Location }
$Standalone = Join-Path $AppSource ".next\standalone"
if (-not (Test-Path (Join-Path $Standalone "server.js") -PathType Leaf)) { throw "Next.js standalone build was not found." }
# Dynamic filesystem tracing can pull temp datasets, databases and .env into
# standalone. Ship the runtime explicitly, rather than everything traced.
foreach ($Entry in @(".next", "node_modules", "public", "package.json", "server.js")) {
    $Source = Join-Path $Standalone $Entry
    if (-not (Test-Path -LiteralPath $Source)) { throw "Missing runtime entry: $Source" }
    Copy-Item -LiteralPath $Source -Destination (Join-Path $Payload "app\$Entry") -Recurse -Force
}
$DefaultOutput = Join-Path $Payload "default-configs"
Push-Location $AppSource
try {
    $ExportArgs = @("tsx", "scripts/export-default-configs.ts", "--output", $DefaultOutput)
    if ($DefaultConfigsPath) { $ExportArgs += @("--source", ([IO.Path]::GetFullPath($DefaultConfigsPath))) }
    & npx.cmd @ExportArgs
    if ($LASTEXITCODE -ne 0) { throw "Exporting default configurations failed." }
} finally { Pop-Location }
$Inventory = Get-Content (Join-Path $DefaultOutput "inventory.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$Unexpected = @(Get-ChildItem (Join-Path $Payload "app") -Recurse -Force -File | Where-Object {
    $_.Name -match '^\.env($|\.)' -or $_.Name -match '\.db($|-(wal|shm|journal)$)'
})
if ($Unexpected.Count) { throw "Runtime contains environment/database files: $($Unexpected.FullName -join ', ')" }
foreach ($File in @("Start.ps1", "Stop.ps1", "Test-Installation.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "configure_db.py")) {
    Copy-Item (Join-Path $ScriptRoot "payload\$File") (Join-Path $Payload $File)
}
Copy-Item (Join-Path $ScriptRoot "Apply-MinimalUpgrade.ps1") $PackageRoot
Copy-Item (Join-Path $ScriptRoot "Read-ConfigPath.mjs") $PackageRoot
Copy-Item (Join-Path $ScriptRoot "UPGRADE.md") (Join-Path $PackageRoot "README.md")
foreach ($File in @("Schedule-AutoTrainUpdate.cmd", "Schedule-AutoTrainUpdate.ps1", "Invoke-AutoTrainUpdate.ps1", "Update-Common.ps1", "Start-UpdateServer.ps1", "SCHEDULED-UPDATE.md", "SCHEDULED-UPDATE.zh-CN.md")) {
    Copy-Item (Join-Path $ScriptRoot $File) $PackageRoot
}
$Manifest = [ordered]@{
    packageName = $PackageName
    packageType = "application-and-default-configs-upgrade"
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    preserves = @("data", "frameworks", "runtime", "logs")
    buildId = (Get-Content (Join-Path $Standalone ".next\BUILD_ID") -Raw).Trim()
    databaseMigrationRequired = $false
    defaultConfigCount = $Inventory.Count
    defaultConfigPolicy = "copy-missing-preserve-existing"
    frameworks = @($Inventory.framework | Select-Object -Unique)
}
[IO.File]::WriteAllText((Join-Path $PackageRoot "manifest.json"), ($Manifest | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
$ChecksumLines = Get-ChildItem $PackageRoot -File -Recurse | Where-Object { $_.Name -ne "SHA256SUMS.txt" } | Sort-Object FullName | ForEach-Object {
    $RelativePath = $_.FullName.Substring($PackageRoot.Length + 1).Replace("\", "/")
    "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower(), $RelativePath
}
[IO.File]::WriteAllLines((Join-Path $PackageRoot "SHA256SUMS.txt"), $ChecksumLines, (New-Object Text.UTF8Encoding($false)))
$Archive = Join-Path $OutputDirectory "$PackageName.zip"
& tar.exe -a -c -f $Archive -C $OutputDirectory $PackageName
if ($LASTEXITCODE -ne 0) { throw "Archive creation failed with exit code $LASTEXITCODE." }
$ArchiveChecksum = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower()
[IO.File]::WriteAllText("$Archive.sha256", "$ArchiveChecksum  $PackageName.zip`n", (New-Object Text.UTF8Encoding($false)))
Write-Host "Minimal upgrade archive created: $Archive" -ForegroundColor Green
