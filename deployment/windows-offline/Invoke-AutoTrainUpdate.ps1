param([Parameter(Mandatory = $true)][string]$PlanPath)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Update-Common.ps1")
$PlanPath = [IO.Path]::GetFullPath($PlanPath)
$PlanDirectory = Split-Path $PlanPath
$Plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Root = Get-UpdateRoot $Plan.installRoot
$Lock = $null
$Transcript = $false
$Stopped = $false
$BackupReady = $false
$ServiceNames = @()
$Run = Join-Path $PlanDirectory ("run-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 6))
$StatusPath = Join-Path $PlanDirectory "status.json"
$Status = [ordered]@{ status = "running"; startedAt = (Get-Date).ToString("o"); finishedAt = $null; error = $null; recovery = $null; runDirectory = $Run }
$LauncherNames = @("Start.ps1", "Stop.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "Test-Installation.ps1", "configure_db.py")
$Port = 3000
$ExitCode = 1
try {
    $Lock = [IO.File]::Open((Join-Path $Root "data\scheduled-update.lock"), 'OpenOrCreate', 'ReadWrite', 'None')
    New-Item -ItemType Directory -Path $Run | Out-Null
    Start-Transcript -LiteralPath (Join-Path $PlanDirectory "update.log") -Append | Out-Null
    $Transcript = $true
    Write-UpdateJson $StatusPath $Status
    Write-Host "Starting update: $Root"
    $Package = Expand-VerifiedUpdate $Plan.packagePath (Join-Path $Run "extracted") $Plan.packageSha256
    $Settings = Join-Path $Root "install-settings.json"
    if (Test-Path -LiteralPath $Settings) { $Port = [int](Get-Content -LiteralPath $Settings -Raw -Encoding UTF8 | ConvertFrom-Json).port }
    if ($Port -lt 1 -or $Port -gt 65535) { throw "Invalid installation HTTP port." }
    $Services = @(Get-AutoTrainServices $Root)
    if ($Services.Count -gt 1) { throw "More than one Windows service matches this installation; resolve the ambiguity before updating." }
    $ServiceNames = @($Services | ForEach-Object { $_.Name })
    Write-Host "Matching services: $($ServiceNames -join ', '); application processes: $(@(Get-AutoTrainProcesses $Root).Count)"
    $Stopped = $true
    Stop-AutoTrainInstance $Root $ServiceNames
    $Backup = Join-Path $Run "backup"
    New-Item -ItemType Directory -Path $Backup | Out-Null
    Copy-Item -LiteralPath (Join-Path $Root "app") -Destination (Join-Path $Backup "app") -Recurse -Force
    foreach ($Name in $LauncherNames) {
        if (Test-Path -LiteralPath (Join-Path $Root $Name)) { Copy-Item -LiteralPath (Join-Path $Root $Name) -Destination $Backup }
    }
    $BackupReady = $true
    & "$PSHOME\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File (Join-Path $Package "Apply-MinimalUpgrade.ps1") -InstallRoot $Root -NoStart
    if ($LASTEXITCODE -ne 0) { throw "Package installer failed with exit code $LASTEXITCODE." }
    Start-AutoTrainInstance $Root $ServiceNames $Port
    Wait-AutoTrainHealthy $Root $Port
    $Status.status = "succeeded"
    $ExitCode = 0
    Write-Host "Update succeeded and the application is healthy."
} catch {
    $Status.status = "failed"
    $Status.error = $_.Exception.Message
    Write-Host "Update failed: $($Status.error)" -ForegroundColor Red
    if ($Stopped) {
        try {
            Stop-AutoTrainInstance $Root $ServiceNames
            if ($BackupReady) {
                Move-Item -LiteralPath (Join-Path $Root "app") -Destination (Join-Path $Run "failed-app")
                Copy-Item -LiteralPath (Join-Path $Backup "app") -Destination (Join-Path $Root "app") -Recurse -Force
                foreach ($Name in $LauncherNames) {
                    if (Test-Path -LiteralPath (Join-Path $Backup $Name)) { Copy-Item -LiteralPath (Join-Path $Backup $Name) -Destination (Join-Path $Root $Name) -Force }
                }
            }
            Start-AutoTrainInstance $Root $ServiceNames $Port
            Wait-AutoTrainHealthy $Root $Port
            $Status.recovery = "Previous application restored and healthy."
        } catch { $Status.recovery = "Recovery failed: $($_.Exception.Message)" }
        Write-Host $Status.recovery
    }
} finally {
    if ($Lock) {
        $Status.finishedAt = (Get-Date).ToString("o")
        Write-UpdateJson $StatusPath $Status
        $Lock.Dispose()
    }
    if ($Transcript) { Stop-Transcript | Out-Null }
}
exit $ExitCode
