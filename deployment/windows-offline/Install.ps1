param(
    [string]$InstallRoot = "C:\AutoTrain",
    [int]$Port = 3000,
    [switch]$RequireGpu,
    [switch]$NoStart
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
$BundleRoot = $PSScriptRoot
$Payload = Join-Path $BundleRoot "payload"
$ManifestPath = Join-Path $BundleRoot "manifest.json"
if (-not [Environment]::Is64BitOperatingSystem) { throw "Only 64-bit Windows is supported." }
if (-not (Test-Path $ManifestPath)) { throw "manifest.json is missing. Run Build-OfflineBundle.ps1 on the online build machine first." }
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$Directories = @(
    $InstallRoot,
    (Join-Path $InstallRoot "app"),
    (Join-Path $InstallRoot "frameworks"),
    (Join-Path $InstallRoot "runtime"),
    (Join-Path $InstallRoot "runtime\envs"),
    (Join-Path $InstallRoot "data"),
    (Join-Path $InstallRoot "data\configs"),
    (Join-Path $InstallRoot "data\users"),
    (Join-Path $InstallRoot "data\cache"),
    (Join-Path $InstallRoot "logs")
)
foreach ($Directory in $Directories) { New-Item -ItemType Directory -Force -Path $Directory | Out-Null }
function Copy-Directory([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) { throw "Missing bundle payload: $Source" }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}
Copy-Directory (Join-Path $Payload "app") (Join-Path $InstallRoot "app")
Copy-Directory (Join-Path $Payload "frameworks") (Join-Path $InstallRoot "frameworks")
Copy-Directory (Join-Path $Payload "runtime\node") (Join-Path $InstallRoot "runtime\node")
Copy-Directory (Join-Path $Payload "requirements") (Join-Path $InstallRoot "runtime\requirements")
if (Test-Path (Join-Path $Payload "cache")) { Copy-Directory (Join-Path $Payload "cache") (Join-Path $InstallRoot "data\cache") }
foreach ($File in @("Start.ps1", "Stop.ps1", "Test-Installation.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "configure_db.py")) {
    Copy-Item (Join-Path $Payload $File) (Join-Path $InstallRoot $File) -Force
}
$Database = Join-Path $InstallRoot "data\autotrain.db"
if (-not (Test-Path $Database)) { Copy-Item (Join-Path $Payload "database\empty.db") $Database }
$VcRedist = Join-Path $Payload "installers\VC_redist.x64.exe"
$VcProcess = Start-Process -FilePath $VcRedist -ArgumentList @("/install", "/quiet", "/norestart") -Wait -PassThru
if ($VcProcess.ExitCode -notin @(0, 1638, 3010)) { throw "Visual C++ Redistributable installation failed with exit code $($VcProcess.ExitCode)." }
if ($VcProcess.ExitCode -eq 3010) { Write-Host "Windows must be restarted before training." -ForegroundColor Yellow }
$PythonRoot = Join-Path $InstallRoot "runtime\python"
$BasePython = Join-Path $PythonRoot "python.exe"
if (-not (Test-Path $BasePython)) {
    $PortablePython = Join-Path $Payload "runtime\python-base"
    if (Test-Path (Join-Path $PortablePython "python.exe")) {
        Copy-Directory $PortablePython $PythonRoot
    } else {
        $PythonInstaller = Join-Path $Payload "installers\python-installer.exe"
        $PythonArguments = "/quiet InstallAllUsers=0 `"TargetDir=$PythonRoot`" Include_pip=1 Include_launcher=0 AssociateFiles=0 Shortcuts=0 PrependPath=0 Include_test=0"
        $PythonProcess = Start-Process -FilePath $PythonInstaller -ArgumentList $PythonArguments -Wait -PassThru
        if ($PythonProcess.ExitCode -ne 0 -or -not (Test-Path $BasePython)) { throw "Python installation failed with exit code $($PythonProcess.ExitCode)." }
    }
}
$Wheelhouse = Join-Path $Payload "wheels"
function Invoke-Python([string]$Python, [string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Python $($Arguments -join ' ')" }
}
$Environments = @("paddle", "torch", "anomaly")
foreach ($Name in $Environments) {
    $EnvironmentPython = Join-Path $InstallRoot "runtime\envs\$Name\Scripts\python.exe"
    if (-not (Test-Path $EnvironmentPython)) { Invoke-Python $BasePython @("-m", "venv", (Split-Path (Split-Path $EnvironmentPython))) }
    Invoke-Python $EnvironmentPython @("-m", "pip", "install", "--no-index", "--find-links", $Wheelhouse, "pip==25.1.1", "setuptools==80.9.0", "wheel==0.45.1")
}
$PaddlePython = Join-Path $InstallRoot "runtime\envs\paddle\Scripts\python.exe"
$TorchPython = Join-Path $InstallRoot "runtime\envs\torch\Scripts\python.exe"
$AnomalyPython = Join-Path $InstallRoot "runtime\envs\anomaly\Scripts\python.exe"
Invoke-Python $PaddlePython (@("-m", "pip", "install", "--no-index", "--find-links", $Wheelhouse, [string]$Manifest.paddlePackage, "-r", (Join-Path $InstallRoot "runtime\requirements\requirements-paddle.txt")))
Invoke-Python $TorchPython (@("-m", "pip", "install", "--no-index", "--find-links", $Wheelhouse) + @($Manifest.torchPackages) + @("-r", (Join-Path $InstallRoot "runtime\requirements\requirements-torch.txt")))
Invoke-Python $AnomalyPython (@("-m", "pip", "install", "--no-index", "--find-links", $Wheelhouse) + @($Manifest.torchPackages) + @("-r", (Join-Path $InstallRoot "runtime\requirements\requirements-anomaly.txt")))
$BundledAnomalyCache = Join-Path $Payload "cache\anomalib"
if (Test-Path $BundledAnomalyCache) {
    $AnomalyCacheRoot = (& $AnomalyPython -c "import platformdirs; print(platformdirs.user_cache_path('anomalib', ensure_exists=True))" | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $AnomalyCacheRoot) { throw "Could not resolve the anomalib cache directory." }
    Copy-Directory $BundledAnomalyCache $AnomalyCacheRoot
}
$SitePackages = Join-Path $InstallRoot "runtime\envs\paddle\Lib\site-packages"
foreach ($Framework in @("PaddleDetection", "PaddleClas", "PaddleSeg")) {
    $Repository = Join-Path $InstallRoot "frameworks\$Framework"
    [IO.File]::WriteAllText((Join-Path $SitePackages "$Framework.pth"), $Repository, (New-Object Text.UTF8Encoding($false)))
}
Invoke-Python $BasePython @((Join-Path $InstallRoot "configure_db.py"), "--db", $Database, "--root", $InstallRoot)
$Settings = @{ port = $Port; installedAt = (Get-Date).ToString("o"); bundle = $Manifest.bundleName } | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $InstallRoot "install-settings.json"), $Settings, (New-Object Text.UTF8Encoding($false)))
& (Join-Path $InstallRoot "Test-Installation.ps1") -RequireGpu:$RequireGpu
if ($LASTEXITCODE -ne 0) { throw "Installation verification failed with exit code $LASTEXITCODE." }
if (-not $NoStart) { & (Join-Path $InstallRoot "Start.ps1") -Port $Port }
Write-Host "AutoTrain was installed to $InstallRoot" -ForegroundColor Green
Write-Host "First registered account becomes the administrator."
