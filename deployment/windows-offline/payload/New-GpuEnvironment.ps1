param(
    [int]$GpuId = 1,
    [ValidateSet("paddle", "torch", "anomaly")]
    [string]$SourceEnv = "paddle",
    [string]$TargetEnv = "",
    [Parameter(Mandatory = $true)]
    [string]$WheelDir,
    [string]$InstallRoot = "",
    [switch]$Register,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
if (-not $InstallRoot) { $InstallRoot = if (Test-Path (Join-Path $PSScriptRoot "runtime\envs")) { $PSScriptRoot } else { "C:\AutoTrain" } }
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$WheelDir = [IO.Path]::GetFullPath($WheelDir)
if (-not $TargetEnv) { $TargetEnv = "${SourceEnv}_gpu$GpuId" }
$BasePython = Join-Path $InstallRoot "runtime\python\python.exe"
$SourceDir = Join-Path $InstallRoot "runtime\envs\$SourceEnv"
$TargetDir = Join-Path $InstallRoot "runtime\envs\$TargetEnv"
$SourcePython = Join-Path $SourceDir "Scripts\python.exe"
$TargetPython = Join-Path $TargetDir "Scripts\python.exe"
foreach ($Path in @($BasePython, $SourcePython)) { if (-not (Test-Path $Path -PathType Leaf)) { throw "Not found: $Path" } }
if (-not (Test-Path $WheelDir -PathType Container)) { throw "Wheel directory not found: $WheelDir" }
if ($TargetDir -eq $SourceDir) { throw "TargetEnv must differ from SourceEnv." }
function Invoke-Python([string]$Python, [string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Python $($Arguments -join ' ')" }
}
if (Test-Path $TargetDir) {
    if (-not $Force) { throw "Target environment already exists: $TargetDir. Use -Force to recreate it." }
    Remove-Item $TargetDir -Recurse -Force
}

Write-Host "Creating $TargetDir" -ForegroundColor Cyan
Invoke-Python $BasePython @("-m", "venv", $TargetDir)
Invoke-Python $TargetPython @("-m", "pip", "install", "--no-index", "--find-links", $WheelDir, "pip==25.1.1", "setuptools==80.9.0", "wheel==0.45.1")

$Frozen = & $SourcePython -m pip freeze --exclude-editable
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed in $SourceDir" }
$Frozen = @($Frozen | Where-Object { $_ -and $_ -notmatch '^\s*#' })
foreach ($Line in @($Frozen | Where-Object { $_ -match ' @ ' })) { Write-Warning "Direct reference, must still exist on disk: $Line" }
$Requirements = Join-Path $InstallRoot "runtime\envs\$TargetEnv-requirements.txt"
[IO.File]::WriteAllLines($Requirements, [string[]]$Frozen, (New-Object Text.UTF8Encoding($false)))
Write-Host "Installing $($Frozen.Count) packages from $WheelDir (list: $Requirements)" -ForegroundColor Cyan
Invoke-Python $TargetPython @("-m", "pip", "install", "--no-index", "--find-links", $WheelDir, "-r", $Requirements)

$SourceSite = Join-Path $SourceDir "Lib\site-packages"
$TargetSite = Join-Path $TargetDir "Lib\site-packages"
Get-ChildItem $SourceSite -Filter *.pth -File | Where-Object { -not (Test-Path (Join-Path $TargetSite $_.Name)) } | ForEach-Object {
    Copy-Item $_.FullName $TargetSite
    Write-Host "Copied $($_.Name)"
}

$Checks = @{
    paddle  = @("import paddle,ppdet,ppcls,paddleseg; print('paddle', paddle.__version__, 'cuda', paddle.device.is_compiled_with_cuda(), 'gpus', paddle.device.cuda.device_count())", "frameworks\PaddleDetection")
    torch   = @("import torch,torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())", "frameworks\torchtrain")
    anomaly = @("import importlib.metadata,torch,anomalib; print('torch', torch.__version__, 'anomalib', importlib.metadata.version('anomalib'), 'cuda', torch.cuda.is_available())", "frameworks\torchtrain")
}
$Check = $Checks[$SourceEnv]
$PreviousDevices = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES = [string]$GpuId
Push-Location (Join-Path $InstallRoot $Check[1])
try {
    Write-Host "Verifying on GPU $GpuId" -ForegroundColor Cyan
    Invoke-Python $TargetPython @("-c", $Check[0])
} finally {
    Pop-Location
    $env:CUDA_VISIBLE_DEVICES = $PreviousDevices
}

$Frameworks = @{ paddle = @("PaddleDetection", "PaddleClas", "PaddleSeg"); torch = @("TorchDet", "TorchSeg"); anomaly = @("TorchAnomaly") }[$SourceEnv]
$Keys = @($Frameworks | ForEach-Object { "${_}:$GpuId" })
if ($Register) {
    $Database = Join-Path $InstallRoot "data\autotrain.db"
    if (-not (Test-Path $Database -PathType Leaf)) { throw "Database not found: $Database" }
    $Backup = "$Database.bak-" + (Get-Date).ToString("yyyyMMdd-HHmmss")
    Copy-Item $Database $Backup
    Write-Host "Database backed up to $Backup"
    $Script = Join-Path $env:TEMP ("autotrain-register-" + [Guid]::NewGuid().ToString("N") + ".py")
    $Code = @'
import json, sqlite3, sys, time
db, python, keys = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
conn = sqlite3.connect(db)
row = conn.execute('SELECT "id", "frameworkPythonMappings" FROM "SystemConfig" ORDER BY "createdAt" LIMIT 1').fetchone()
if not row:
    sys.exit("SystemConfig row not found in " + db)
mapping = json.loads(row[1] or "{}")
for key in keys:
    mapping[key] = python
conn.execute('UPDATE "SystemConfig" SET "frameworkPythonMappings"=?, "updatedAt"=? WHERE "id"=?', (json.dumps(mapping), int(time.time() * 1000), row[0]))
conn.commit()
conn.close()
print(json.dumps(mapping, indent=2))
'@
    [IO.File]::WriteAllText($Script, $Code, (New-Object Text.UTF8Encoding($false)))
    try { Invoke-Python $BasePython @($Script, $Database, $TargetPython, ($Keys -join ",")) } finally { Remove-Item $Script -Force -ErrorAction SilentlyContinue }
    Write-Host "Registered $($Keys -join ', ') -> $TargetPython" -ForegroundColor Green
} else {
    Write-Host "Not registered. Add these entries to the framework Python mappings in Settings (or rerun with -Register):" -ForegroundColor Yellow
    foreach ($Key in $Keys) { Write-Host ("  `"{0}`": `"{1}`"," -f $Key, $TargetPython.Replace('\', '\\')) }
}
Write-Host "Environment ready: $TargetDir" -ForegroundColor Green
