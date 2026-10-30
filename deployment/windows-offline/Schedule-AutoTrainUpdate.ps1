param(
    [string]$PackagePath = "",
    [string]$InstallRoot = "",
    [string]$UpdateTime = ""
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Update-Common.ps1")
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]::new($Identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run Schedule-AutoTrainUpdate.cmd as administrator to register the SYSTEM scheduled task."
}
if (-not $PackagePath) { $PackagePath = Read-Host "Update ZIP path" }
if (-not $InstallRoot) { $InstallRoot = Read-Host "AutoTrain installation directory" }
if (-not $UpdateTime) { $UpdateTime = Read-Host "Update time (local yyyy-MM-dd HH:mm)" }
$PackagePath = [IO.Path]::GetFullPath($PackagePath.Trim().Trim('"'))
if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf) -or [IO.Path]::GetExtension($PackagePath) -ne '.zip') { throw "Select an existing update ZIP." }
$Root = Get-UpdateRoot $InstallRoot
$When = [datetime]::ParseExact($UpdateTime.Trim(), "yyyy-MM-dd HH:mm", [Globalization.CultureInfo]::InvariantCulture)
if ($When -le (Get-Date).AddMinutes(1)) { throw "Choose an execution time at least one minute in the future." }
$TaskName = Get-UpdateTaskName $Root
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing -and $Existing.State -eq "Running") { throw "An update for this installation is already running." }
$Base = Join-Path $Root "update-plans"
New-Item -ItemType Directory -Path $Base -Force | Out-Null
# These files will execute as SYSTEM. Do not inherit write access for ordinary users.
$Acl = New-Object Security.AccessControl.DirectorySecurity
$Acl.SetAccessRuleProtection($true, $false)
foreach ($Sid in @("S-1-5-18", "S-1-5-32-544")) {
    $Rule = New-Object Security.AccessControl.FileSystemAccessRule(
        (New-Object Security.Principal.SecurityIdentifier($Sid)), "FullControl",
        "ContainerInherit,ObjectInherit", "None", "Allow")
    $Acl.AddAccessRule($Rule)
}
$Acl.SetOwner((New-Object Security.Principal.SecurityIdentifier("S-1-5-32-544")))
Set-Acl -LiteralPath $Base -AclObject $Acl
$Directory = Join-Path $Base ((Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Directory | Out-Null
$StagedZip = Join-Path $Directory "update.zip"
Copy-Item -LiteralPath $PackagePath -Destination $StagedZip
foreach ($Name in @("Invoke-AutoTrainUpdate.ps1", "Update-Common.ps1", "Start-UpdateServer.ps1")) { Copy-Item -LiteralPath (Join-Path $PSScriptRoot $Name) -Destination $Directory }
$PlanPath = Join-Path $Directory "plan.json"
Write-UpdateJson $PlanPath ([ordered]@{
    installRoot = $Root
    packagePath = $StagedZip
    packageSha256 = (Get-FileHash -LiteralPath $StagedZip -Algorithm SHA256).Hash.ToLowerInvariant()
    scheduledAt = $When.ToString("o")
    taskName = $TaskName
})
$Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + (Join-Path $Directory "Invoke-AutoTrainUpdate.ps1") + '" -PlanPath "' + $PlanPath + '"'
$Action = New-ScheduledTaskAction -Execute "$PSHOME\powershell.exe" -Argument $Arguments -WorkingDirectory $Directory
$Trigger = New-ScheduledTaskTrigger -Once -At $When
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
$Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "AutoTrain application update: verify, stop, backup, apply, restart, and recover on failure."
Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null
Write-UpdateJson (Join-Path $Directory "status.json") @{ status = "scheduled"; scheduledAt = $When.ToString("o") }
Write-Host "Update scheduled for $($When.ToString('yyyy-MM-dd HH:mm'))" -ForegroundColor Green
Write-Host "Task: $TaskName"
Write-Host "Plan and logs: $Directory"
Write-Host "Cancel in Task Scheduler, or run: Unregister-ScheduledTask -TaskName '$TaskName'"
