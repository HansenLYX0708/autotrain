param(
    [string]$Label = "",
    [string]$OutputDirectory = "",
    [string]$DefaultConfigsPath = "",
    [switch]$SkipChecks,
    [switch]$SkipBuild,
    [switch]$SkipSmokeTest,
    [int]$TscBaseline = 7
)
Set-StrictMode -Version 2
$ErrorActionPreference = "Stop"
$StartedAt = Get-Date

function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Invoke-Native([scriptblock]$Call, [string]$What) {
    # Native commands write warnings to stderr. With $ErrorActionPreference =
    # "Stop" any stderr reach via the pipeline becomes a terminating
    # NativeCommandError, so run them with Continue and check $LASTEXITCODE.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Call | Write-Host } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { Fail "$What failed with exit code $LASTEXITCODE." }
}

$ScriptRoot = $PSScriptRoot
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $ScriptRoot "..\.."))

# --- a. settings: CLI > local-upgrade.config.json > defaults -----------------
$ConfigFile = Join-Path $ScriptRoot "local-upgrade.config.json"
$Config = @{}
if (Test-Path $ConfigFile -PathType Leaf) {
    $Config = Get-Content -LiteralPath $ConfigFile -Raw -Encoding UTF8 | ConvertFrom-Json
}
# StrictMode throws on a missing property of a JSON object, so read keys
# through this helper: a partial config file must still work.
function Get-ConfigValue([string]$Name) {
    if ($Config -is [hashtable]) { return $Config[$Name] }
    $property = $Config.PSObject.Properties[$Name]
    if ($property) { return $property.Value }
    return $null
}
if (-not $PSBoundParameters.ContainsKey("OutputDirectory") -and (Get-ConfigValue "outputDirectory")) {
    $OutputDirectory = Get-ConfigValue "outputDirectory"
}
if (-not $PSBoundParameters.ContainsKey("DefaultConfigsPath") -and (Get-ConfigValue "defaultConfigsPath")) {
    $DefaultConfigsPath = Get-ConfigValue "defaultConfigsPath"
}
if (-not $PSBoundParameters.ContainsKey("TscBaseline") -and $null -ne (Get-ConfigValue "tscBaseline")) {
    $TscBaseline = [int](Get-ConfigValue "tscBaseline")
}
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $RepoRoot "dist" }
if (-not (Test-Path $RepoRoot)) { Fail "Repository root not found: $RepoRoot" }
if ($DefaultConfigsPath -and -not (Test-Path $DefaultConfigsPath -PathType Container)) {
    Fail "DefaultConfigsPath does not exist: $DefaultConfigsPath"
}
if ($Label -and $Label -notmatch '^[A-Za-z0-9._-]+$') {
    Fail "Label may only contain letters, digits, '.', '_' and '-': $Label"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)

# --- b. timestamp + label --------------------------------------------------
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ($Label) { $Timestamp = "$Timestamp-$Label" }

# --- c. git info -----------------------------------------------------------
$SourceRevision = ""
$Dirty = $false
Push-Location $RepoRoot
try {
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        $SourceRevision = (& git.exe rev-parse --short HEAD 2>$null | Select-Object -First 1)
        $porcelain = (& git.exe status --porcelain 2>$null)
        if ($porcelain) { $Dirty = $true }
    } finally { $ErrorActionPreference = $prev }
} finally { Pop-Location }
if ($Dirty) {
    Write-Host "WARNING: working tree is dirty; the package is built from the uncommitted state." -ForegroundColor Yellow
}
if ($SourceRevision) {
    if ($Dirty) { $SourceRevision = "$SourceRevision-dirty" }
    Write-Host "Source revision: $SourceRevision"
} else {
    Write-Host "WARNING: could not read git revision; manifest sourceRevision will be empty." -ForegroundColor Yellow
}

# --- d. checks -------------------------------------------------------------
if (-not $SkipChecks) {
    Write-Host "`n==> TypeScript check (baseline: $TscBaseline known errors)"
    $tscLines = @()
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        Push-Location $RepoRoot
        $tscLines = & npx.cmd tsc --noEmit --pretty false 2>&1 | Out-String -Stream
        Pop-Location
    } finally { $ErrorActionPreference = $prev }
    $errorLines = @($tscLines | Where-Object { $_ -match 'error TS' })
    if ($errorLines.Count -gt $TscBaseline) {
        Write-Host ($errorLines | Select-Object -First 20 | Out-String)
        Fail "tsc reports $($errorLines.Count) errors, baseline is $TscBaseline. Fix the code or update the baseline."
    }
    Write-Host "    tsc: $($errorLines.Count) error(s) within baseline."

    foreach ($Script in @("check-log-parsers.ts", "check-dataset-config-chain.ts", "check-console-log.ts", "check-labelme-images.ts")) {
        $ScriptPath = Join-Path $RepoRoot "scripts\$Script"
        if (-not (Test-Path $ScriptPath -PathType Leaf)) { continue }
        Write-Host "`n==> npx tsx scripts/$Script"
        Push-Location $RepoRoot
        try {
            Invoke-Native { & npx.cmd tsx $ScriptPath } "scripts/$Script"
        } finally { Pop-Location }
    }
}

# --- e. build the package --------------------------------------------------
Write-Host "`n==> Build-MinimalUpgrade.ps1 (Timestamp=$Timestamp)"
$BuildArgs = @{
    OutputDirectory = $OutputDirectory
    Timestamp = $Timestamp
    SourceRevision = $SourceRevision
}
if ($DefaultConfigsPath) { $BuildArgs.DefaultConfigsPath = $DefaultConfigsPath }
if ($SkipBuild) { $BuildArgs.SkipBuild = $true }
& (Join-Path $ScriptRoot "Build-MinimalUpgrade.ps1") @BuildArgs
if ($LASTEXITCODE -ne 0) { Fail "Build-MinimalUpgrade.ps1 failed with exit code $LASTEXITCODE." }

$PackageName = "AutoTrain-Minimal-Upgrade-$Timestamp"
$Archive = Join-Path $OutputDirectory "$PackageName.zip"
if (-not (Test-Path $Archive -PathType Leaf)) { Fail "Expected archive was not produced: $Archive" }

# --- f. verify the archive -------------------------------------------------
Write-Host "`n==> Verifying archive"
$shaFile = "$Archive.sha256"
if (-not (Test-Path $shaFile -PathType Leaf)) { Fail "Missing $shaFile" }
$expectedZip = (Get-Content -LiteralPath $shaFile -Encoding UTF8).Split(" ")[0]
$actualZip = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLower()
if ($expectedZip -ne $actualZip) { Fail "Zip SHA256 mismatch: file says $expectedZip, computed $actualZip" }
Write-Host "    zip checksum matches .zip.sha256"

$CheckRoot = Join-Path $env:TEMP "autotrain-upgrade-check-$Timestamp"
if (Test-Path $CheckRoot) { Remove-Item $CheckRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $CheckRoot | Out-Null
try {
    Push-Location $CheckRoot
    try {
        Invoke-Native { & tar.exe -xf $Archive -C $CheckRoot } "tar extract"
    } finally { Pop-Location }
    $PackageRoot = Join-Path $CheckRoot $PackageName
    $PayloadApp = Join-Path $PackageRoot "payload\app"

    foreach ($required in @((Join-Path $PayloadApp "server.js"), (Join-Path $PackageRoot "manifest.json"))) {
        if (-not (Test-Path $required -PathType Leaf)) { Fail "Package is missing $required" }
    }
    $leaked = @(Get-ChildItem $PayloadApp -Recurse -Force -File | Where-Object {
        $_.Name -match '^\.env($|\.)' -or $_.Name -match '\.db($|-(wal|shm|journal)$)'
    })
    if ($leaked.Count) { Fail "Package leaks environment/database files: $($leaked.FullName -join ', ')" }

    $ChecksumFile = Join-Path $PackageRoot "SHA256SUMS.txt"
    if (-not (Test-Path $ChecksumFile -PathType Leaf)) { Fail "Package checksums are missing." }
    $checked = 0
    foreach ($Line in Get-Content -LiteralPath $ChecksumFile -Encoding UTF8) {
        if ($Line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { Fail "Invalid checksum entry: $Line" }
        $Expected = $Matches[1]
        $File = [IO.Path]::GetFullPath((Join-Path $PackageRoot $Matches[2]))
        if (-not $File.StartsWith($PackageRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { Fail "Invalid package path." }
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { Fail "Missing package file: $File" }
        if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -ne $Expected) { Fail "Checksum mismatch: $File" }
        $checked++
    }
    Write-Host "    $checked packaged file checksums verified"

    if (-not $SkipSmokeTest) {
        Write-Host "`n==> Smoke test (packaged server on a scratch database)"
        $savedEnv = @{}
        foreach ($name in @("DATABASE_URL", "PORT", "HOSTNAME", "AUTH_COOKIE_SECURE", "NODE_ENV")) {
            $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name)
        }
        $server = $null
        try {
            $env:DATABASE_URL = "file:" + (Join-Path $CheckRoot "smoke.db").Replace("\", "/")
            Push-Location $RepoRoot
            try {
                Invoke-Native { & npx.cmd prisma db push --skip-generate } "prisma db push (smoke db)"
            } finally { Pop-Location }

            $port = 3020
            while ($port -lt 3100) {
                try {
                    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $port)
                    $listener.Start(); $listener.Stop(); break
                } catch { $port++ }
            }
            if ($port -ge 3100) { Fail "No free port in 3020-3099 for the smoke test" }

            $env:NODE_ENV = "production"
            $env:PORT = "$port"
            $env:HOSTNAME = "127.0.0.1"
            $env:AUTH_COOKIE_SECURE = "false"
            $outLog = Join-Path $CheckRoot "server-out.log"
            $errLog = Join-Path $CheckRoot "server-err.log"
            $server = Start-Process -FilePath (Get-Command node.exe).Source `
                -ArgumentList "`"$PayloadApp\server.js`"" `
                -WorkingDirectory $PayloadApp `
                -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
                -PassThru

            $deadline = (Get-Date).AddSeconds(60)
            $html = $null
            while ((Get-Date) -lt $deadline) {
                try {
                    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -UseBasicParsing -TimeoutSec 5
                    if ($r.StatusCode -eq 200) { $html = $r.Content; break }
                } catch { Start-Sleep -Milliseconds 800 }
            }
            if (-not $html) { Fail "Packaged server did not answer HTTP 200 within 60s (see $errLog)" }
            Write-Host "    GET / -> 200"

            $asset = [regex]::Match($html, '(?:src|href)="([^"]*/_next/static/[^"]+)"').Groups[1].Value
            if (-not $asset) { Fail "HTML contains no static asset reference" }
            $assetStatus = (Invoke-WebRequest -Uri "http://127.0.0.1:$port$asset" -UseBasicParsing -TimeoutSec 10).StatusCode
            if ($assetStatus -ne 200) { Fail "Static asset $asset returned $assetStatus" }
            Write-Host "    GET $asset -> 200"

            $apiStatus = 0
            try {
                (Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/datasets" -UseBasicParsing -TimeoutSec 10).StatusCode | Out-Null
            } catch {
                $apiStatus = [int]$_.Exception.Response.StatusCode
            }
            if ($apiStatus -ne 401) { Fail "GET /api/datasets returned $apiStatus, expected 401" }
            Write-Host "    GET /api/datasets -> 401 (unauthenticated, API routes load)"
            Write-Host "    smoke test passed on port $port"
        } finally {
            if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction Continue }
            foreach ($name in $savedEnv.Keys) {
                [Environment]::SetEnvironmentVariable($name, $savedEnv[$name])
            }
        }
    }
} finally {
    Remove-Item $CheckRoot -Recurse -Force -ErrorAction Continue
}

# --- g. summary ------------------------------------------------------------
$elapsed = [math]::Round(((Get-Date) - $StartedAt).TotalMinutes, 1)
$sizeMB = [math]::Round((Get-Item $Archive).Length / 1MB, 1)
Write-Host "`n================ UPDATE PACKAGE READY ================" -ForegroundColor Green
Write-Host "Archive        : $Archive"
Write-Host "Size           : $sizeMB MB"
Write-Host "SHA256         : $actualZip"
Write-Host "sourceRevision : $SourceRevision"
Write-Host "Elapsed        : $elapsed min"
Write-Host "Install on the target machine with:"
Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apply-MinimalUpgrade.ps1 -InstallRoot `"D:\AutoTrain`""
exit 0
