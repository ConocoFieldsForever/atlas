# remote-screenshot-relay.ps1
#
# Run this on the Tarkov gaming PC. It watches EFT's screenshot directory and creates a zero-byte
# marker with the SAME filename in an Atlas inbox (normally an SMB share hosted by the render VM).
# Atlas parses the position/quaternion from the filename, so no screenshot pixels cross the LAN.

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $true)]
    [string]$InboxDir,

    [ValidateRange(100, 5000)]
    [int]$PollMilliseconds = 250
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
    throw "Screenshot source directory does not exist: $SourceDir"
}

if (-not (Test-Path -LiteralPath $InboxDir -PathType Container)) {
    New-Item -ItemType Directory -Path $InboxDir -Force | Out-Null
}

# Resolve to native filesystem paths. `Resolve-Path(...).Path` can return a provider-qualified
# UNC value such as `Microsoft.PowerShell.Core\FileSystem::\\server\share`, which PowerShell
# cmdlets accept but .NET's File.WriteAllBytes rejects as an unsupported path format.
$source = (Get-Item -LiteralPath $SourceDir -Force).FullName
$inbox = (Get-Item -LiteralPath $InboxDir -Force).FullName
$seen = @{}

# Existing screenshots are history. Seed the set so starting the relay never replays or touches
# them; only files appearing after this process starts become remote position fixes.
Get-ChildItem -LiteralPath $source -Filter "*.png" -File -ErrorAction SilentlyContinue |
    ForEach-Object { $seen[$_.FullName] = $true }

Write-Host "Atlas screenshot relay"
Write-Host "  source: $source"
Write-Host "  inbox:  $inbox"
Write-Host "Only zero-byte filename markers are sent. Press Ctrl+C to stop."

while ($true) {
    try {
        Get-ChildItem -LiteralPath $source -Filter "*.png" -File -ErrorAction Stop |
            Sort-Object LastWriteTimeUtc |
            ForEach-Object {
                if (-not $seen.ContainsKey($_.FullName)) {
                    $marker = Join-Path $inbox $_.Name
                    [System.IO.File]::WriteAllBytes($marker, [byte[]]@())
                    $seen[$_.FullName] = $true
                    Write-Host "[$(Get-Date -Format HH:mm:ss.fff)] relayed $($_.Name)"
                }
            }
    } catch {
        Write-Warning "relay poll failed: $($_.Exception.Message)"
    }
    Start-Sleep -Milliseconds $PollMilliseconds
}
