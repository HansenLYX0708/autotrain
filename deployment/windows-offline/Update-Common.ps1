Set-StrictMode -Version 2
$ErrorActionPreference = "Stop"

function Write-UpdateJson($Path, $Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
}

function Get-UpdateRoot([string]$Path) {
    $Full = [IO.Path]::GetFullPath($Path.Trim().Trim('"')).TrimEnd('\')
    if ($Full -notmatch '^[A-Za-z]:\\.+') { throw "Use an absolute local installation path, not a drive root or network share." }
    foreach ($Relative in @("app\server.js", "runtime\node\node.exe", "data\autotrain.db")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Full $Relative) -PathType Leaf)) { throw "Installation file missing: $Relative in $Full" }
    }
    return $Full
}

function Get-UpdateTaskName([string]$Root) {
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try { $Hash = [BitConverter]::ToString($Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Root.ToLowerInvariant()))).Replace("-", "").Substring(0, 16) }
    finally { $Hasher.Dispose() }
    return "AutoTrain-Update-$Hash"
}

function Expand-VerifiedUpdate([string]$Zip, [string]$Destination, [string]$ExpectedHash) {
    if ((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash -ne $ExpectedHash) { throw "The staged ZIP has changed; SHA-256 does not match." }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $Prefix = [IO.Path]::GetFullPath($Destination).TrimEnd('\') + '\'
        foreach ($Entry in $Archive.Entries) {
            $Target = [IO.Path]::GetFullPath((Join-Path $Destination $Entry.FullName))
            if (-not $Target.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase) -or $Entry.FullName.Contains(':')) {
                throw "Unsafe path in ZIP: $($Entry.FullName)"
            }
            if (($Entry.ExternalAttributes -shr 16 -band 0xF000) -eq 0xA000) { throw "ZIP symbolic links are not supported." }
        }
    } finally { $Archive.Dispose() }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Destination
    $Installers = @(Get-ChildItem -LiteralPath $Destination -Filter Apply-MinimalUpgrade.ps1 -Recurse -File)
    if ($Installers.Count -ne 1) { throw "Expected exactly one Apply-MinimalUpgrade.ps1 in the ZIP." }
    $Package = $Installers[0].Directory.FullName
    $ManifestPath = Join-Path $Package "manifest.json"
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Manifest.packageType -notin @("application-only-upgrade", "application-and-default-configs-upgrade")) { throw "This is not a supported minimal update package." }
    if ($Manifest.PSObject.Properties.Name -contains "databaseMigrationRequired" -and $Manifest.databaseMigrationRequired) { throw "Database migration packages require a separate upgrade procedure." }
    $Checksums = Join-Path $Package "SHA256SUMS.txt"
    $CheckedInstaller = $false
    foreach ($Line in Get-Content -LiteralPath $Checksums -Encoding UTF8) {
        if ($Line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw "Invalid package checksum entry." }
        $Expected = $Matches[1]
        $Relative = $Matches[2]
        $File = [IO.Path]::GetFullPath((Join-Path $Package $Relative))
        if (-not $File.StartsWith($Package + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe checksum path." }
        if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -ne $Expected) { throw "Package checksum mismatch: $Relative" }
        if ($Relative -eq "Apply-MinimalUpgrade.ps1") { $CheckedInstaller = $true }
    }
    if (-not $CheckedInstaller) { throw "The installer is not covered by the package checksums." }
    if (-not (Test-Path (Join-Path $Package "payload\app\server.js"))) { throw "The update has no application server." }
    return $Package
}

function Get-AutoTrainProcesses([string]$Root) {
    $Server = [regex]::Escape((Join-Path $Root "app\server.js"))
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "node.exe" -and $_.CommandLine -and
        $_.CommandLine.Replace('/', '\') -match "(?i)(?:^|[\s`"'])$Server(?:$|[\s`"'])"
    })
}

function Get-AutoTrainServices([string]$Root) {
    $Owners = New-Object 'System.Collections.Generic.HashSet[uint32]'
    foreach ($Server in @(Get-AutoTrainProcesses $Root)) {
        [void]$Owners.Add([uint32]$Server.ProcessId)
        [void]$Owners.Add([uint32]$Server.ParentProcessId)
    }
    @(Get-CimInstance Win32_Service | Where-Object {
        $MatchesPath = $_.PathName -and $_.PathName.Replace('/', '\').IndexOf($Root + '\', [StringComparison]::OrdinalIgnoreCase) -ge 0
        $MatchesProcess = $_.ProcessId -gt 0 -and $Owners.Contains([uint32]$_.ProcessId)
        # NSSM can live outside the installation and have no child while stopped.
        $Parameters = Get-ItemProperty -LiteralPath ("HKLM:\SYSTEM\CurrentControlSet\Services\" + $_.Name + "\Parameters") -ErrorAction SilentlyContinue
        $MatchesNssm = $false
        if ($Parameters) {
            foreach ($Property in $Parameters.PSObject.Properties | Where-Object { $_.Name -in @("Application", "AppParameters", "AppDirectory") }) {
                $Value = ([string]$Property.Value).Replace('/', '\')
                if ($Value.IndexOf($Root + '\', [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $Value.Trim('"').TrimEnd('\') -eq $Root) { $MatchesNssm = $true }
            }
        }
        $MatchesPath -or $MatchesProcess -or $MatchesNssm
    })
}

function Stop-AutoTrainInstance([string]$Root, [string[]]$ServiceNames) {
    foreach ($Name in $ServiceNames) {
        $Service = Get-Service -Name $Name
        if ($Service.Status -ne 'Stopped') {
            Write-Host "Stopping service: $Name"
            Stop-Service -Name $Name
            (Get-Service -Name $Name).WaitForStatus('Stopped', [TimeSpan]::FromSeconds(45))
        }
    }
    foreach ($AppProcess in @(Get-AutoTrainProcesses $Root)) {
        Write-Host "Stopping application PID $($AppProcess.ProcessId)"
        $Process = Get-Process -Id $AppProcess.ProcessId -ErrorAction SilentlyContinue
        if ($Process) {
            Stop-Process -Id $Process.Id
            if (-not $Process.WaitForExit(30000)) { throw "Application process did not stop." }
        }
    }
    if (@(Get-AutoTrainProcesses $Root).Count) { throw "Application is still running." }
    # Never pass a stale PID file to a package's older stop script.
    $PidFile = Join-Path $Root "data\autotrain.pid"
    if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
}

function Start-AutoTrainInstance([string]$Root, [string[]]$ServiceNames, [int]$Port) {
    if ($ServiceNames.Count) {
        foreach ($Name in $ServiceNames) {
            Start-Service -Name $Name
            (Get-Service -Name $Name).WaitForStatus('Running', [TimeSpan]::FromSeconds(45))
        }
        return
    }
    # Create an independent launcher through WMI. A direct child can keep a
    # task/native command's process tree and output handles open indefinitely.
    $Helper = Join-Path $PSScriptRoot "Start-UpdateServer.ps1"
    $Command = '"' + "$PSHOME\powershell.exe" + '" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $Helper + '" -InstallRoot "' + $Root + '" -Port ' + $Port
    $Created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $Command; CurrentDirectory = $Root }
    if ($Created.ReturnValue -ne 0) { throw "Independent server launcher failed: $($Created.ReturnValue)" }
    Write-Host "Launched independent server starter PID $($Created.ProcessId)."
}

function Start-AutoTrainServerProcess([string]$Root, [int]$Port) {
    # Same runtime environment as payload/Start.ps1; quote server.js for paths
    # containing spaces, including packages made with the older launcher.
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
    New-Item -ItemType Directory -Path (Join-Path $Root "logs") -Force | Out-Null
    $Node = Join-Path $Root "runtime\node\node.exe"
    $Server = Join-Path $Root "app\server.js"
    $Process = Start-Process -FilePath $Node -ArgumentList ('"' + $Server + '"') -WorkingDirectory (Join-Path $Root "app") -RedirectStandardOutput (Join-Path $Root "logs\server.log") -RedirectStandardError (Join-Path $Root "logs\server-error.log") -WindowStyle Hidden -PassThru
    [IO.File]::WriteAllText((Join-Path $Root "data\autotrain.pid"), [string]$Process.Id, [Text.Encoding]::ASCII)
    Write-Host "Started application PID $($Process.Id) on port $Port"
}

function Wait-AutoTrainHealthy([string]$Root, [int]$Port) {
    $Deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $Deadline) {
        $Ids = @((Get-AutoTrainProcesses $Root) | ForEach-Object { $_.ProcessId })
        $Listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -in $Ids })
        if ($Listeners.Count) {
            try {
                $Response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 5
                if ($Response.StatusCode -eq 200) { return }
            } catch {}
        }
        Start-Sleep -Seconds 2
    }
    throw "The installation did not become healthy at http://127.0.0.1:$Port/ within 90 seconds."
}
