param(
    [int]$Port = 0,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$SettingsPath = Join-Path $Root "install-settings.json"
if ($Port -le 0 -and (Test-Path $SettingsPath)) { $Port = [int]((Get-Content $SettingsPath -Raw | ConvertFrom-Json).port) }
if ($Port -le 0) { $Port = 3000 }
$PidFile = Join-Path $Root "data\autotrain.pid"
if (Test-Path $PidFile) {
    $ExistingPid = [int](Get-Content $PidFile -Raw)
    if (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue) {
        Write-Host "AutoTrain is already running. PID: $ExistingPid"
        if (-not $NoBrowser) { Start-Process "http://localhost:$Port" }
        exit 0
    }
    Remove-Item $PidFile -Force
}
$env:NODE_ENV = "production"
$env:HOSTNAME = "0.0.0.0"
$env:PORT = "$Port"
$env:DATABASE_URL = "file:$((Join-Path $Root 'data\autotrain.db').Replace('\', '/'))"
$env:DATABASE_PATH = Join-Path $Root "data\users"
$env:USER_CONFIGS_PATH = Join-Path $Root "data\configs"
$env:AUTH_COOKIE_SECURE = "false"
$env:TORCH_HOME = Join-Path $Root "data\cache\torch"
$env:PADDLE_HOME = Join-Path $Root "data\cache\paddle"
$env:TIMM_USE_OLD_CACHE = "1"
$env:XDG_CACHE_HOME = Join-Path $Root "data\cache"
$env:HF_HOME = Join-Path $Root "data\cache\huggingface"
$Node = Join-Path $Root "runtime\node\node.exe"
$Server = Join-Path $Root "app\server.js"
$Stdout = Join-Path $Root "logs\server.log"
$Stderr = Join-Path $Root "logs\server-error.log"
$Process = Start-Process -FilePath $Node -ArgumentList @($Server) -WorkingDirectory (Join-Path $Root "app") -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
Set-Content -Path $PidFile -Value $Process.Id -Encoding ASCII
Start-Sleep -Seconds 3
if ($Process.HasExited) {
    Write-Host "AutoTrain failed to start. Check $Stderr" -ForegroundColor Red
    exit 1
}
Write-Host "AutoTrain started: http://localhost:$Port (PID $($Process.Id))" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process "http://localhost:$Port" }
