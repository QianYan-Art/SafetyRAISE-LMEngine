[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Prepare", "Collect")]
    [string]$Action,

    [Parameter(Mandatory = $true)]
    [ValidateSet("all", "dx12", "vulkan")]
    [string]$Backend,

    [string]$OutputDirectory = $(if ($env:RUNNER_TEMP) {
        Join-Path $env:RUNNER_TEMP "rsinfer-gpu-diagnostics"
    } else {
        Join-Path (Get-Location) ".gpu-diagnostics"
    })
)

$ErrorActionPreference = "Stop"
# WER LocalDumps 只支持 HKLM；本脚本仅在短命 GitHub Actions runner 上启用并恢复。
$werPath = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps"
$werBackupPath = Join-Path $OutputDirectory "wer-localdumps-before.json"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Write-OptionalCommandOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    try {
        & $Command @Arguments *> $Path
        "exit_code=$LASTEXITCODE" | Add-Content -Path $Path
    } catch {
        $_ | Out-String | Set-Content -Path $Path
    }
}

function Save-AdapterSnapshot {
    try {
        Get-CimInstance Win32_VideoController |
            Select-Object Name, DriverVersion, DriverDate, AdapterCompatibility, PNPDeviceID |
            ConvertTo-Json -Depth 4 |
            Set-Content -Path (Join-Path $OutputDirectory "video-controllers.json")
    } catch {
        $_ | Out-String | Set-Content -Path (Join-Path $OutputDirectory "video-controllers.error.txt")
    }

    Write-OptionalCommandOutput -Command "dxdiag.exe" -Arguments @("/t", (Join-Path $OutputDirectory "dxdiag.txt")) -Path (Join-Path $OutputDirectory "dxdiag-command.txt")
}

function Save-EnvironmentSnapshot {
    $snapshot = [ordered]@{}
    foreach ($name in @(
        "CI",
        "GITHUB_ACTIONS",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "WGPU_BACKEND",
        "RSINFER_GPU_DIAGNOSTICS",
        "RUST_BACKTRACE"
    )) {
        $snapshot[$name] = [Environment]::GetEnvironmentVariable($name)
    }
    $snapshot["backend_argument"] = $Backend
    $snapshot | ConvertTo-Json | Set-Content -Path (Join-Path $OutputDirectory "environment.json")
}

if ($Action -eq "Prepare") {
    if ($env:GITHUB_ACTIONS -ne "true") {
        throw "GPU WER 诊断只允许在 GITHUB_ACTIONS=true 的临时 runner 上启用。"
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "配置 WER LocalDumps 需要管理员权限；拒绝静默跳过。"
    }

    Save-AdapterSnapshot
    Save-EnvironmentSnapshot

    $before = [ordered]@{ exists = Test-Path $werPath }
    foreach ($name in @("DumpFolder", "DumpType", "DumpCount")) {
        $before[$name] = $null
    }
    if ($before.exists) {
        $properties = Get-ItemProperty -Path $werPath
        foreach ($name in @("DumpFolder", "DumpType", "DumpCount")) {
            $before[$name] = $properties.$name
        }
    }
    $before | ConvertTo-Json | Set-Content -Path $werBackupPath

    New-Item -ItemType Directory -Force -Path $werPath | Out-Null
    New-ItemProperty -Path $werPath -Name DumpFolder -PropertyType ExpandString -Value $OutputDirectory -Force | Out-Null
    New-ItemProperty -Path $werPath -Name DumpType -PropertyType DWord -Value 2 -Force | Out-Null
    New-ItemProperty -Path $werPath -Name DumpCount -PropertyType DWord -Value 3 -Force | Out-Null
    Set-Content -Path (Join-Path $OutputDirectory "wer-localdumps-enabled.txt") -Value "HKLM LocalDumps enabled for this ephemeral GitHub Actions runner; restored during collection."
    exit 0
}

Save-AdapterSnapshot
Save-EnvironmentSnapshot

foreach ($logName in @("Application", "System")) {
    try {
        Get-WinEvent -FilterHashtable @{ LogName = $logName; StartTime = (Get-Date).AddMinutes(-30) } |
            Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message |
            Format-List |
            Out-File -FilePath (Join-Path $OutputDirectory "event-$($logName.ToLowerInvariant()).txt") -Encoding utf8
    } catch {
        $_ | Out-String | Set-Content -Path (Join-Path $OutputDirectory "event-$($logName.ToLowerInvariant()).error.txt")
    }
}

try {
    Get-ChildItem -Path $OutputDirectory -Filter "*.dmp" -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object FullName, Length, LastWriteTime |
        ConvertTo-Json -Depth 3 |
        Set-Content -Path (Join-Path $OutputDirectory "dumps.json")
} catch {
    $_ | Out-String | Set-Content -Path (Join-Path $OutputDirectory "dumps.error.txt")
}

if (Test-Path $werBackupPath) {
    try {
        $before = Get-Content -Raw -Path $werBackupPath | ConvertFrom-Json
        Remove-ItemProperty -Path $werPath -Name DumpFolder, DumpType, DumpCount -ErrorAction SilentlyContinue
        if ($before.exists) {
            foreach ($name in @("DumpFolder", "DumpType", "DumpCount")) {
                if ($null -ne $before.$name) {
                    $propertyType = if ($name -eq "DumpFolder") { "ExpandString" } else { "DWord" }
                    New-ItemProperty -Path $werPath -Name $name -PropertyType $propertyType -Value $before.$name -Force | Out-Null
                }
            }
        } else {
            Remove-Item -Path $werPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        Set-Content -Path (Join-Path $OutputDirectory "wer-localdumps-restored.txt") -Value "HKLM LocalDumps settings restored after collection."
    } catch {
        $_ | Out-String | Set-Content -Path (Join-Path $OutputDirectory "wer-localdumps-restore.error.txt")
    }
}
