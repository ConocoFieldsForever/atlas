[CmdletBinding()]
param(
    [string]$AtlasRoot,
    [string]$GameData,
    [string]$AssetsRoot,
    [string]$TarkmapRoot,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LogPath = Join-Path $PSScriptRoot "build-all-atlas-maps.log"
$LockPath = Join-Path $PSScriptRoot "build-all-atlas-maps.running"
$LocalConfigPath = Join-Path $PSScriptRoot "build-all-atlas-maps.local.psd1"

# Keep machine-specific paths out of source control. A local data file is optional; explicit
# command-line parameters win, followed by the local file, then the standard EFT_* environment.
if (Test-Path -LiteralPath $LocalConfigPath) {
    $LocalConfig = Import-PowerShellDataFile -LiteralPath $LocalConfigPath
    if (-not $PSBoundParameters.ContainsKey("AtlasRoot")) { $AtlasRoot = $LocalConfig.AtlasRoot }
    if (-not $PSBoundParameters.ContainsKey("GameData")) { $GameData = $LocalConfig.GameData }
    if (-not $PSBoundParameters.ContainsKey("AssetsRoot")) { $AssetsRoot = $LocalConfig.AssetsRoot }
    if (-not $PSBoundParameters.ContainsKey("TarkmapRoot")) { $TarkmapRoot = $LocalConfig.TarkmapRoot }
}

if (-not $AtlasRoot) { $AtlasRoot = $env:EFT_ATLAS_ROOT }
if (-not $GameData) { $GameData = $env:EFT_GAME_DATA }
if (-not $AssetsRoot) { $AssetsRoot = $env:EFT_ASSETS_ROOT }
if (-not $TarkmapRoot) { $TarkmapRoot = $env:EFT_TARKMAP_ROOT }

function Write-Status {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $Line
    Add-Content -LiteralPath $LogPath -Value $Line
}

function Get-FreeSpaceGiB {
    param([string]$Path)
    try {
        $DriveRoot = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Path).Path)
        if (-not $DriveRoot) { return $null }
        $Drive = New-Object System.IO.DriveInfo($DriveRoot)
        return [math]::Round($Drive.AvailableFreeSpace / 1GB, 1)
    }
    catch {
        return $null
    }
}

Write-Host ""
Write-Host "ATLAS - BUILD ALL MAPS" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "This queue is resumable. Completed map packs are skipped automatically."
Write-Host ""

if (-not $AtlasRoot -or -not (Test-Path -LiteralPath $AtlasRoot)) {
    throw "Atlas release not found. Pass -AtlasRoot, set EFT_ATLAS_ROOT, or create '$LocalConfigPath'."
}
if (-not $GameData -or -not (Test-Path -LiteralPath $GameData)) {
    throw "EFT game-data directory not found. Pass -GameData, set EFT_GAME_DATA, or configure the local data file."
}
if (-not $AssetsRoot) {
    throw "Extracted-assets directory is not configured. Pass -AssetsRoot or set EFT_ASSETS_ROOT."
}
if (-not $TarkmapRoot) {
    throw "Writable map workspace is not configured. Pass -TarkmapRoot or set EFT_TARKMAP_ROOT."
}

$Python = Join-Path $AtlasRoot "python\python.exe"
$Builder = Join-Path $AtlasRoot "tools\build_map.py"
$AtlasExe = Join-Path $AtlasRoot "atlas.exe"

foreach ($RequiredPath in @($Python, $Builder, $AtlasExe)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}

# Read the supported roster from the release instead of freezing a version-specific map list in
# this helper. Factory is first for quick feedback; Streets is last because it is usually largest.
$MapRosterPath = Join-Path $AtlasRoot "extraction\maps\manifest.json"
if (-not (Test-Path -LiteralPath $MapRosterPath)) {
    throw "Atlas map roster not found: $MapRosterPath"
}
$MapIds = @((Get-Content -LiteralPath $MapRosterPath -Raw | ConvertFrom-Json).maps.id)
$Maps = @()
if ($MapIds -contains "factory_rework") { $Maps += "factory_rework" }
$Maps += @($MapIds | Where-Object { $_ -notin @("factory_rework", "streets") } | Sort-Object)
if ($MapIds -contains "streets") { $Maps += "streets" }

$Completed = @()
$Pending = @()
foreach ($Map in $Maps) {
    $Manifest = Join-Path $AtlasRoot ("packs\{0}.eftpack\manifest.json" -f $Map)
    if (Test-Path -LiteralPath $Manifest) {
        $Completed += $Map
    }
    else {
        $Pending += $Map
    }
}

Write-Host ("Release:   {0}" -f $AtlasRoot)
Write-Host ("Game:      {0}" -f $GameData)
Write-Host ("Assets:    {0}" -f $AssetsRoot)
Write-Host ("Workspace: {0}" -f $TarkmapRoot)
Write-Host ("Complete:  {0}/{1}" -f $Completed.Count, $Maps.Count) -ForegroundColor Green
Write-Host ("Pending:   {0}" -f (($Pending -join ", ")))

$AssetFree = Get-FreeSpaceGiB -Path $AssetsRoot
$PackFree = Get-FreeSpaceGiB -Path $AtlasRoot
if ($null -ne $AssetFree) { Write-Host ("Free on asset drive: {0} GiB" -f $AssetFree) }
if ($null -ne $PackFree) { Write-Host ("Free on pack drive:  {0} GiB" -f $PackFree) }
if (($null -ne $AssetFree -and $AssetFree -lt 80) -or ($null -ne $PackFree -and $PackFree -lt 80)) {
    Write-Warning "Building every self-contained pack may need tens of gigabytes. Free more space if a build reports disk-full."
}

if ($DryRun) {
    Write-Host "Dry run complete; no map builds were started." -ForegroundColor Yellow
    exit 0
}

if ($Pending.Count -eq 0) {
    Write-Host "All Atlas map packs are already installed." -ForegroundColor Green
    exit 0
}

if (Get-Process -Name "EscapeFromTarkov" -ErrorAction SilentlyContinue) {
    throw "Escape from Tarkov is running. Close the game before the one-time map extraction, then run this shortcut again."
}

$OwnsLock = $false
if (Test-Path -LiteralPath $LockPath) {
    $ExistingPid = 0
    $ExistingPidText = (Get-Content -LiteralPath $LockPath -Raw -ErrorAction SilentlyContinue)
    $ExistingProcess = $null
    if ($null -ne $ExistingPidText -and
        [int]::TryParse($ExistingPidText.Trim(), [ref]$ExistingPid)) {
        $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    }

    if ($null -ne $ExistingProcess -and $ExistingProcess.ProcessName -eq "powershell") {
        throw "Another Build All Maps queue is running as process $ExistingPid. Leave that window open and let it finish."
    }

    Write-Status "Removed a stale Build All Maps lock left by a stopped attempt."
    Remove-Item -LiteralPath $LockPath -Force
}

try {
    # CreateNew is atomic, preventing two rapidly double-clicked launchers from both starting.
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    $LockBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$PID)
    $LockStream.Write($LockBytes, 0, $LockBytes.Length)
    $LockStream.Dispose()
    $OwnsLock = $true
}
catch {
    throw "Another Build All Maps queue started at the same time. Leave its window open and let it finish."
}

try {
    New-Item -ItemType Directory -Force -Path $AssetsRoot, $TarkmapRoot | Out-Null

    $env:EFT_GAME_DATA = $GameData
    $env:EFT_ASSETS_ROOT = $AssetsRoot
    $env:EFT_TARKMAP_ROOT = $TarkmapRoot
    $env:EFT_ATLAS_EXE = $AtlasExe
    $env:EFT_BAKE_CPU = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Write-Status ("Starting resumable queue with {0} pending map(s)." -f $Pending.Count)

    $Number = 0
    foreach ($Map in $Maps) {
        $Number++
        $Manifest = Join-Path $AtlasRoot ("packs\{0}.eftpack\manifest.json" -f $Map)
        if (Test-Path -LiteralPath $Manifest) {
            Write-Status ("SKIP {0}/{1}: {2} is already complete." -f $Number, $Maps.Count, $Map)
            continue
        }

        Write-Status ("BUILD {0}/{1}: {2}" -f $Number, $Maps.Count, $Map)

        # Continue is intentional here: native stderr must be displayed and logged without
        # PowerShell converting it into a terminating NativeCommandError.
        $PreviousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python $Builder $Map "--self-contained" 2>&1 |
            Tee-Object -FilePath $LogPath -Append
        $BuildExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorAction

        if ($BuildExitCode -ne 0 -or -not (Test-Path -LiteralPath $Manifest)) {
            Write-Status ("FAILED: {0}. Fix the reported error, then run the shortcut again; completed maps will be skipped." -f $Map)
            exit 1
        }

        Write-Status ("READY: {0}" -f $Map)
    }

    Write-Status "All Atlas map packs are installed. Restart Atlas to refresh the map list."
    Write-Host ""
    Write-Host "ALL MAPS READY" -ForegroundColor Green
}
finally {
    if ($OwnsLock) {
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
}
