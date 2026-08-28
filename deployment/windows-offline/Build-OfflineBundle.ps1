param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [string]$OutputDirectory = ".\dist",
    [switch]$SkipNpmInstall,
    [switch]$CreateArchive,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
$ScriptRoot = $PSScriptRoot
$ConfigPath = [IO.Path]::GetFullPath($ConfigPath)
$ConfigRoot = Split-Path $ConfigPath
$Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
function Resolve-ConfiguredPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $ConfigRoot $Value))
}
function Assert-File([string]$Path, [string]$Name) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "$Name was not found: $Path" }
}
function Assert-Directory([string]$Path, [string]$Name) {
    if (-not (Test-Path $Path -PathType Container)) { throw "$Name was not found: $Path" }
}
function Invoke-External([string]$Executable, [object[]]$Arguments, [string]$WorkingDirectory = "") {
    if ($WorkingDirectory) { Push-Location $WorkingDirectory }
    try {
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
    } finally {
        if ($WorkingDirectory) { Pop-Location }
    }
}
function Copy-Tree([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy.exe $Source $Destination /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD .git output log logs __pycache__ | Out-Null
    if ($LASTEXITCODE -gt 7) { throw "Copy failed ($LASTEXITCODE): $Source" }
}
$AppSource = Resolve-ConfiguredPath ([string]$Config.appSource)
$NodeArchive = Resolve-ConfiguredPath ([string]$Config.nodeArchive)
$PythonInstaller = Resolve-ConfiguredPath ([string]$Config.pythonInstaller)
$VcRedistInstaller = Resolve-ConfiguredPath ([string]$Config.vcRedistInstaller)
$BuilderPython = Resolve-ConfiguredPath ([string]$Config.builderPython)
Assert-Directory $AppSource "Application source"
Assert-File $NodeArchive "Node.js Windows x64 archive"
Assert-File $PythonInstaller "Python Windows x64 installer"
Assert-File $VcRedistInstaller "Visual C++ x64 Redistributable installer"
Assert-File $BuilderPython "Builder Python"
$FrameworkSources = @{}
foreach ($Name in @("PaddleDetection", "PaddleClas", "PaddleSeg")) {
    $Path = Resolve-ConfiguredPath ([string]$Config.frameworkSources.$Name)
    Assert-Directory $Path "$Name source"
    Assert-File (Join-Path $Path "tools\train.py") "$Name tools/train.py"
    $ModuleDirectory = @{ PaddleDetection = "ppdet"; PaddleClas = "ppcls"; PaddleSeg = "paddleseg" }[$Name]
    Assert-Directory (Join-Path $Path $ModuleDirectory) "$Name Python module"
    $FrameworkSources[$Name] = $Path
}
Assert-File (Join-Path $AppSource "torchtrain\tools\train.py") "torchtrain tools/train.py"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$BundleRoot = Join-Path $OutputDirectory ([string]$Config.bundleName)
if (Test-Path $BundleRoot) {
    if (-not $Force) { throw "Output exists: $BundleRoot. Use -Force to replace it." }
    Remove-Item $BundleRoot -Recurse -Force
}
$Payload = Join-Path $BundleRoot "payload"
foreach ($Directory in @("app", "frameworks", "runtime\node", "runtime\python-base", "installers", "wheels", "requirements", "database")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Payload $Directory) | Out-Null
}
if (-not $SkipNpmInstall) { Invoke-External "npm.cmd" @("ci") $AppSource }
Invoke-External "npm.cmd" @("run", "db:generate") $AppSource
Invoke-External "npm.cmd" @("run", "build") $AppSource
Assert-File (Join-Path $AppSource ".next\standalone\server.js") "Next.js standalone server"
Copy-Tree (Join-Path $AppSource ".next\standalone") (Join-Path $Payload "app")
foreach ($Name in $FrameworkSources.Keys) { Copy-Tree $FrameworkSources[$Name] (Join-Path $Payload "frameworks\$Name") }
Copy-Tree (Join-Path $AppSource "torchtrain") (Join-Path $Payload "frameworks\torchtrain")
$NodeTemp = Join-Path $env:TEMP ("autotrain-node-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $NodeTemp | Out-Null
try {
    Expand-Archive -Path $NodeArchive -DestinationPath $NodeTemp
    $NodeExe = Get-ChildItem $NodeTemp -Filter node.exe -Recurse | Select-Object -First 1
    if (-not $NodeExe) { throw "node.exe was not found in $NodeArchive" }
    $BundleNodeVersion = & $NodeExe.FullName --version
    $BuilderNodeVersion = & node.exe --version
    if (($BundleNodeVersion -split '\.')[0] -ne ($BuilderNodeVersion -split '\.')[0]) { throw "Node.js major versions differ: builder $BuilderNodeVersion, bundle $BundleNodeVersion" }
    Copy-Tree $NodeExe.Directory.FullName (Join-Path $Payload "runtime\node")
} finally {
    Remove-Item $NodeTemp -Recurse -Force -ErrorAction SilentlyContinue
}
Copy-Tree (Split-Path $BuilderPython) (Join-Path $Payload "runtime\python-base")
Copy-Item $PythonInstaller (Join-Path $Payload "installers\python-installer.exe")
Copy-Item $VcRedistInstaller (Join-Path $Payload "installers\VC_redist.x64.exe")
Copy-Item (Join-Path $ScriptRoot "requirements-paddle.txt") (Join-Path $Payload "requirements")
Copy-Item (Join-Path $ScriptRoot "requirements-torch.txt") (Join-Path $Payload "requirements")
Copy-Item (Join-Path $ScriptRoot "requirements-anomaly.txt") (Join-Path $Payload "requirements")
foreach ($File in @("Start.ps1", "Stop.ps1", "Test-Installation.ps1", "Start-AutoTrain.cmd", "Stop-AutoTrain.cmd", "configure_db.py")) {
    Copy-Item (Join-Path $ScriptRoot "payload\$File") (Join-Path $Payload $File)
}
Copy-Item (Join-Path $ScriptRoot "Install.ps1") $BundleRoot
Copy-Item (Join-Path $ScriptRoot "Install-AutoTrain.cmd") $BundleRoot
$Wheelhouse = Join-Path $Payload "wheels"
$PythonIndex = [string]$Config.pythonIndexUrl
Invoke-External $BuilderPython @("-m", "pip", "wheel", "--wheel-dir", $Wheelhouse, "--index-url", $PythonIndex, "pip==25.1.1", "setuptools==80.9.0", "wheel==0.45.1")
Invoke-External $BuilderPython @("-m", "pip", "wheel", "--wheel-dir", $Wheelhouse, "--index-url", $PythonIndex, "-r", (Join-Path $ScriptRoot "requirements-paddle.txt"))
$PaddleArguments = @("-m", "pip", "wheel", "--wheel-dir", $Wheelhouse) + @($Config.paddlePipArgs) + @([string]$Config.paddlePackage)
Invoke-External $BuilderPython $PaddleArguments
$TorchArguments = @("-m", "pip", "wheel", "--wheel-dir", $Wheelhouse, "--index-url", [string]$Config.torchIndexUrl, "--extra-index-url", $PythonIndex) + @($Config.torchPackages) + @("-r", (Join-Path $ScriptRoot "requirements-torch.txt"))
Invoke-External $BuilderPython $TorchArguments
$AnomalyArguments = @("-m", "pip", "wheel", "--wheel-dir", $Wheelhouse, "--index-url", [string]$Config.torchIndexUrl, "--extra-index-url", $PythonIndex) + @($Config.torchPackages) + @("-r", (Join-Path $ScriptRoot "requirements-anomaly.txt"))
Invoke-External $BuilderPython $AnomalyArguments
$EmptyDatabase = Join-Path $Payload "database\empty.db"
$PreviousDatabaseUrl = $env:DATABASE_URL
try {
    $env:DATABASE_URL = "file:$($EmptyDatabase.Replace('\', '/'))"
    Invoke-External "npx.cmd" @("prisma", "db", "push", "--skip-generate") $AppSource
} finally {
    $env:DATABASE_URL = $PreviousDatabaseUrl
}
if ($Config.PSObject.Properties.Name -contains "cacheSources") {
    foreach ($Name in @("paddle", "torch", "anomalib")) {
        $Value = [string]$Config.cacheSources.$Name
        if ($Value) {
            $CacheSource = Resolve-ConfiguredPath $Value
            Assert-Directory $CacheSource "$Name cache"
            Copy-Tree $CacheSource (Join-Path $Payload "cache\$Name")
        }
    }
}
$Manifest = [ordered]@{
    bundleName = [string]$Config.bundleName
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    paddlePackage = [string]$Config.paddlePackage
    torchPackages = @($Config.torchPackages)
    pythonInstaller = [IO.Path]::GetFileName($PythonInstaller)
    nodeArchive = [IO.Path]::GetFileName($NodeArchive)
}
[IO.File]::WriteAllText((Join-Path $BundleRoot "manifest.json"), ($Manifest | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
Copy-Item (Join-Path $ScriptRoot "README.md") (Join-Path $BundleRoot "DEPLOYMENT.md")
if ($CreateArchive) {
    $Archive = Join-Path $OutputDirectory (([string]$Config.bundleName) + ".zip")
    if (Test-Path $Archive) { Remove-Item $Archive -Force }
    Invoke-External "tar.exe" @("-a", "-c", "-f", $Archive, "-C", $OutputDirectory, [string]$Config.bundleName)
    Write-Host "Offline archive created: $Archive" -ForegroundColor Green
}
Write-Host "Offline bundle created: $BundleRoot" -ForegroundColor Green
