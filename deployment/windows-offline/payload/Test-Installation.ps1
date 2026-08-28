param(
    [switch]$RequireGpu
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Failures = New-Object System.Collections.Generic.List[string]
function Test-Command([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory = $Root) {
    Write-Host "Checking $Name..."
    Push-Location $WorkingDirectory
    try {
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) { $Failures.Add($Name) }
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        $Failures.Add($Name)
    } finally {
        Pop-Location
    }
}
$Node = Join-Path $Root "runtime\node\node.exe"
$PaddlePython = Join-Path $Root "runtime\envs\paddle\Scripts\python.exe"
$TorchPython = Join-Path $Root "runtime\envs\torch\Scripts\python.exe"
$AnomalyPython = Join-Path $Root "runtime\envs\anomaly\Scripts\python.exe"
Test-Command "Node.js" $Node @("--version")
Test-Command "Paddle frameworks" $PaddlePython @("-c", "import paddle,ppdet,ppcls,paddleseg; print('paddle', paddle.__version__, 'cuda', paddle.device.is_compiled_with_cuda())") (Join-Path $Root "frameworks\PaddleDetection")
Test-Command "PyTorch frameworks" $TorchPython @("-c", "import torch,torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())") (Join-Path $Root "frameworks\torchtrain")
Test-Command "TorchAnomaly" $AnomalyPython @("-c", "import importlib.metadata,torch,torchvision,anomalib; print('torch', torch.__version__, 'anomalib', importlib.metadata.version('anomalib'), 'cuda', torch.cuda.is_available())") (Join-Path $Root "frameworks\torchtrain")
Test-Command "Database" $PaddlePython @("-c", "import sqlite3; c=sqlite3.connect(r'$((Join-Path $Root 'data\autotrain.db'))'); print(c.execute('select count(*) from SystemConfig').fetchone()[0]); c.close()")
$Required = @(
    "app\server.js",
    "frameworks\PaddleDetection\tools\train.py",
    "frameworks\PaddleClas\tools\train.py",
    "frameworks\PaddleSeg\tools\train.py",
    "frameworks\torchtrain\tools\train.py"
)
foreach ($RelativePath in $Required) {
    if (-not (Test-Path (Join-Path $Root $RelativePath))) { $Failures.Add($RelativePath) }
}
if ($RequireGpu) {
    Test-Command "NVIDIA driver" "nvidia-smi.exe" @()
    Test-Command "Paddle CUDA" $PaddlePython @("-c", "import paddle,sys; sys.exit(0 if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() else 1)")
    Test-Command "PyTorch CUDA" $TorchPython @("-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)")
}
if ($Failures.Count -gt 0) {
    Write-Host "Installation check failed: $($Failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "Installation check passed." -ForegroundColor Green
