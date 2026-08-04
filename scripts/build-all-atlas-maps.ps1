[CmdletBinding()]
param(
    [string]$AtlasRoot,
    [string]$GameData,
    [string]$AssetsRoot,
    [string]$TarkmapRoot,
    [switch]$ForceCpu,
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
    if (-not $PSBoundParameters.ContainsKey("ForceCpu") -and $LocalConfig.ContainsKey("ForceCpu")) {
        $ForceCpu = [bool]$LocalConfig.ForceCpu
    }
}

if (-not $AtlasRoot) { $AtlasRoot = $env:EFT_ATLAS_ROOT }
if (-not $GameData) { $GameData = $env:EFT_GAME_DATA }
if (-not $AssetsRoot) { $AssetsRoot = $env:EFT_ASSETS_ROOT }
if (-not $TarkmapRoot) { $TarkmapRoot = $env:EFT_TARKMAP_ROOT }

function Write-Status {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $Line
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

function Test-PackComplete {
    param([string]$Map)
    $Manifest = Join-Path $AtlasRoot ("packs\{0}.eftpack\manifest.json" -f $Map)
    if (-not (Test-Path -LiteralPath $Manifest)) { return $false }
    try {
        $Data = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
        # sourceFingerprint is written by stage 9, after assembly, lighting, gameplay data, icons,
        # nav, and final manifest reconciliation. A bare manifest can exist during stage 4 and is
        # therefore not sufficient evidence that an interrupted pack is complete.
        return -not [string]::IsNullOrWhiteSpace([string]$Data.sourceFingerprint)
    }
    catch {
        return $false
    }
}

function Get-MapStageFraction {
    param(
        [int]$Stage,
        [int]$Total,
        [string]$Label,
        [Nullable[double]]$SubFraction,
        [bool]$FreshExtract
    )

    if ($Total -ne 9) {
        $Inner = if ($Label -match ": (done|skipped)") { 1.0 } elseif ($null -ne $SubFraction) { $SubFraction } else { 0.05 }
        return [math]::Max(0.0, [math]::Min(1.0, (($Stage - 1) + $Inner) / $Total))
    }

    if ($FreshExtract) {
        $Windows = @{
            1 = @(0.000, 0.636); 2 = @(0.636, 0.649); 4 = @(0.649, 0.732)
            3 = @(0.732, 0.917); 5 = @(0.917, 0.918); 6 = @(0.918, 0.941)
            7 = @(0.941, 0.955); 8 = @(0.955, 0.999); 9 = @(0.999, 1.000)
        }
    }
    else {
        $Windows = @{
            1 = @(0.000, 0.001); 2 = @(0.001, 0.030); 4 = @(0.030, 0.672)
            3 = @(0.672, 0.826); 5 = @(0.826, 0.827); 6 = @(0.827, 0.867)
            7 = @(0.867, 0.868); 8 = @(0.868, 0.999); 9 = @(0.999, 1.000)
        }
    }

    $Done = $Label -match ": (done|skipped)"
    if ($Done) {
        $Inner = 1.0
    }
    elseif ($null -ne $SubFraction) {
        $Inner = [math]::Max(0.0, [math]::Min(1.0, [double]$SubFraction))
    }
    else {
        $Inner = 0.02
    }

    # Stage 1 contains three serial first-extraction passes. Give their live subprogress separate
    # slices so the bar keeps moving through the longest part of a new map.
    if ($FreshExtract -and $Stage -eq 1 -and -not $Done) {
        if ($Label -match "extract dataset") { $Inner = 0.52 * $Inner }
        elseif ($Label -match "grass density") { $Inner = 0.52 + (0.01 * $Inner) }
        elseif ($Label -match "physics colliders") { $Inner = 0.53 + (0.47 * $Inner) }
        else { $Inner = 0.0 }
    }

    $Window = $Windows[$Stage]
    if ($null -eq $Window) { return 0.0 }
    return [double]$Window[0] + (([double]$Window[1] - [double]$Window[0]) * $Inner)
}

function Show-BuildProgress {
    param(
        [int]$MapNumber,
        [int]$MapCount,
        [string]$Map,
        [string]$StageLabel,
        [double]$MapFraction
    )
    $MapPercent = [math]::Round([math]::Max(0.0, [math]::Min(1.0, $MapFraction)) * 100)
    $Overall = [math]::Round((($MapNumber - 1) + $MapFraction) / $MapCount * 100)
    Write-Progress -Id 1 -Activity "Atlas - Build All Maps" -Status ("Map {0}/{1}: {2}" -f $MapNumber, $MapCount, $Map) -PercentComplete $Overall
    Write-Progress -Id 2 -ParentId 1 -Activity ("Building {0}" -f $Map) -Status $StageLabel -PercentComplete $MapPercent
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
    if (Test-PackComplete -Map $Map) {
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
Write-Host ("Processing: {0}" -f $(if ($ForceCpu) { "CPU fallback" } else { "GPU acceleration where supported" }))
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
    if ($ForceCpu) {
        $env:EFT_BAKE_CPU = "1"
    }
    else {
        Remove-Item Env:EFT_BAKE_CPU -ErrorAction SilentlyContinue
    }
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Write-Status ("Starting resumable queue with {0} pending map(s)." -f $Pending.Count)

    $Number = 0
    foreach ($Map in $Maps) {
        $Number++
        if (Test-PackComplete -Map $Map) {
            Write-Status ("SKIP {0}/{1}: {2} is already complete." -f $Number, $Maps.Count, $Map)
            Show-BuildProgress -MapNumber $Number -MapCount $Maps.Count -Map $Map -StageLabel "Already complete" -MapFraction 1.0
            continue
        }

        Write-Status ("BUILD {0}/{1}: {2}" -f $Number, $Maps.Count, $Map)
        $FreshExtract = $false
        $CurrentStage = 0
        $CurrentStageTotal = 9
        $CurrentStageLabel = "Starting"
        $CurrentSubFraction = $null
        $MaxMapFraction = 0.0
        Show-BuildProgress -MapNumber $Number -MapCount $Maps.Count -Map $Map -StageLabel $CurrentStageLabel -MapFraction 0.0

        # Continue is intentional here: native stderr must be displayed and logged without
        # PowerShell converting it into a terminating NativeCommandError.
        $PreviousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python $Builder $Map "--self-contained" 2>&1 | ForEach-Object {
            $BuildLine = $_.ToString()
            Write-Host $BuildLine
            Add-Content -LiteralPath $LogPath -Value $BuildLine -Encoding UTF8

            if ($BuildLine -match '^\[STAGE\s+(\d+)/(\d+)\]\s+(.+)$') {
                $CurrentStage = [int]$Matches[1]
                $CurrentStageTotal = [int]$Matches[2]
                $CurrentStageLabel = $Matches[3]
                $CurrentSubFraction = $null
                if ($CurrentStage -eq 1 -and $CurrentStageLabel -match 'extract dataset') {
                    $FreshExtract = $true
                }
            }
            elseif ($BuildLine -match '^\[SUBPROGRESS\].*?([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)\s*$') {
                $SubDone = [double]$Matches[1]
                $SubTotal = [double]$Matches[2]
                if ($SubTotal -gt 0) { $CurrentSubFraction = $SubDone / $SubTotal }
            }
            elseif ($BuildLine -match '^\[BUILD OK\]') {
                $MaxMapFraction = 1.0
            }

            if ($CurrentStage -gt 0 -and $MaxMapFraction -lt 1.0) {
                $Candidate = Get-MapStageFraction -Stage $CurrentStage -Total $CurrentStageTotal -Label $CurrentStageLabel -SubFraction $CurrentSubFraction -FreshExtract $FreshExtract
                $MaxMapFraction = [math]::Max($MaxMapFraction, $Candidate)
            }
            Show-BuildProgress -MapNumber $Number -MapCount $Maps.Count -Map $Map -StageLabel $CurrentStageLabel -MapFraction $MaxMapFraction
        }
        $BuildExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorAction

        if ($BuildExitCode -ne 0 -or -not (Test-PackComplete -Map $Map)) {
            Write-Progress -Id 2 -Activity ("Building {0}" -f $Map) -Completed
            Write-Status ("FAILED: {0}. Fix the reported error, then run the shortcut again; completed maps will be skipped." -f $Map)
            exit 1
        }

        Show-BuildProgress -MapNumber $Number -MapCount $Maps.Count -Map $Map -StageLabel "Ready" -MapFraction 1.0
        Write-Progress -Id 2 -Activity ("Building {0}" -f $Map) -Completed
        Write-Status ("READY: {0}" -f $Map)
    }

    Write-Progress -Id 1 -Activity "Atlas - Build All Maps" -Completed
    Write-Status "All Atlas map packs are installed. Restart Atlas on the VM to refresh the map list."
    Write-Host ""
    Write-Host "ALL MAPS READY" -ForegroundColor Green
}
finally {
    if ($OwnsLock) {
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
}
