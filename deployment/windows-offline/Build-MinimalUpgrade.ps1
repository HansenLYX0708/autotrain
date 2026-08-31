param(
    [string]$OutputDirectory = ".\dist",
    [string]$Timestamp = (Get-Date -Format "yyyyMMdd-HHmmss"),
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
$Standalone = Join-Path $AppSource ".next\standalone"
if (-not (Test-Path (Join-Path $Standalone "server.js") -PathType Leaf)) { throw "Next.js standalone build was not found." }
Copy-Item (Join-Path $Standalone "*") (Join-Path $Payload "app") -Recurse -Force
foreach ($File in @("Start.ps1", "Stop.ps1", "Test-Installation.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "configure_db.py")) {
    Copy-Item (Join-Path $ScriptRoot "payload\$File") (Join-Path $Payload $File)
}
Copy-Item (Join-Path $ScriptRoot "Apply-MinimalUpgrade.ps1") $PackageRoot
$Manifest = [ordered]@{
    packageName = $PackageName
    packageType = "application-only-upgrade"
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    preserves = @("data", "frameworks", "runtime", "logs")
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
