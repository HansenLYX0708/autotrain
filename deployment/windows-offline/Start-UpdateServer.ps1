param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][int]$Port
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Update-Common.ps1")
try {
    $Root = Get-UpdateRoot $InstallRoot
    Start-AutoTrainServerProcess $Root $Port
} catch {
    [IO.File]::WriteAllText((Join-Path $InstallRoot "logs\update-start-error.log"), ($_ | Out-String), (New-Object Text.UTF8Encoding($false)))
    exit 1
}
