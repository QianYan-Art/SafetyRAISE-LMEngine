[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Prepare", "Collect", "Reproduce")]
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
$procdumpDownloadUrl = "https://download.sysinternals.com/files/Procdump.zip"
$firstGpuTest = "backend::wgpu_q8::tests::real_q8_decode_matches_quantized_prefill_and_budget_fallback"

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

function Install-ProcDump {
    $root = Join-Path $OutputDirectory "procdump"
    $zip = Join-Path $root "Procdump.zip"
    $executable = Join-Path $root "procdump64.exe"
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    Invoke-WebRequest -Uri $procdumpDownloadUrl -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $root -Force
    if (-not (Test-Path -LiteralPath $executable)) {
        throw "ProcDump 下载后缺少 procdump64.exe。"
    }

    $file = Get-Item -LiteralPath $executable
    $hash = Get-FileHash -LiteralPath $executable -Algorithm SHA256
    [ordered]@{
        url = $procdumpDownloadUrl
        name = $file.Name
        length = $file.Length
        sha256 = $hash.Hash
        file_version = $file.VersionInfo.FileVersion
        product_version = $file.VersionInfo.ProductVersion
    } | ConvertTo-Json | Set-Content -Path (Join-Path $OutputDirectory "procdump-version.json")
    Write-OptionalCommandOutput -Command $executable -Arguments @("-accepteula", "-?") -Path (Join-Path $OutputDirectory "procdump-help.txt")
    return $executable
}

function Get-TestExecutable {
    $jsonLog = Join-Path $OutputDirectory "cargo-test-no-run.jsonl"
    $cargoArguments = @(
        "test",
        "--release",
        "--features",
        "gpu-tests",
        "--lib",
        "--no-run",
        "--message-format=json"
    )
    & cargo @cargoArguments 2>&1 | Tee-Object -FilePath $jsonLog | Out-Host
    $cargoExit = $LASTEXITCODE
    if ($cargoExit -ne 0) {
        throw "cargo --no-run 失败，退出码 $cargoExit；详见 cargo-test-no-run.jsonl。"
    }

    $artifacts = @(
        Get-Content -LiteralPath $jsonLog | ForEach-Object {
            try {
                $message = $_ | ConvertFrom-Json
            } catch {
                return
            }
            if ($message.reason -eq "compiler-artifact" -and
                $message.profile.test -eq $true -and
                $message.executable -and
                $message.target.kind -contains "lib") {
                $message
            }
        }
    )
    if ($artifacts.Count -ne 1) {
        throw "未能唯一确定库测试 executable，候选数为 $($artifacts.Count)；详见 cargo-test-no-run.jsonl。"
    }
    $executable = (Resolve-Path -LiteralPath $artifacts[0].executable).Path
    [ordered]@{
        executable = $executable
        test_name = $firstGpuTest
        cargo_exit = $cargoExit
    } | ConvertTo-Json | Set-Content -Path (Join-Path $OutputDirectory "proc-dump-target.json")
    return $executable
}

function Invoke-ProcDumpReproduction {
    $procdump = Install-ProcDump
    $executable = Get-TestExecutable
    $dumpDirectory = Join-Path $OutputDirectory "dumps"
    New-Item -ItemType Directory -Force -Path $dumpDirectory | Out-Null
    $logPath = Join-Path $OutputDirectory "procdump-reproduction.log"
    $arguments = @(
        "-accepteula",
        "-ma",
        "-e",
        "-n",
        "1",
        "-x",
        $dumpDirectory,
        $executable,
        "--exact",
        $firstGpuTest,
        "--test-threads=1",
        "--nocapture"
    )
    & $procdump @arguments 2>&1 | Tee-Object -FilePath $logPath
    $procdumpExit = $LASTEXITCODE
    $log = Get-Content -Raw -LiteralPath $logPath
    $dumps = @(Get-ChildItem -LiteralPath $dumpDirectory -Filter "*.dmp" -File -ErrorAction SilentlyContinue)
    $summaryOk = $log -match "(?im)test result:\s+ok\."
    $summaryFailed = $log -match "(?im)test result:\s+FAILED\."
    $reproductionExit = if ($procdumpExit -eq 0 -and $summaryOk -and -not $summaryFailed -and $dumps.Count -eq 0) { 0 } else { 1 }
    [ordered]@{
        test_name = $firstGpuTest
        executable = $executable
        procdump_exit = $procdumpExit
        test_summary_ok = $summaryOk
        test_summary_failed = $summaryFailed
        dump_count = $dumps.Count
        reproduction_exit = $reproductionExit
        note = "ProcDump 退出码不代表子测试退出码；以测试 summary、dump 数量和 ProcDump 退出码共同判定。"
    } | ConvertTo-Json | Set-Content -Path (Join-Path $OutputDirectory "procdump-reproduction-summary.json")
    if ($reproductionExit -ne 0) {
        throw "ProcDump 复现未得到明确成功 summary（ProcDump=$procdumpExit，dumps=$($dumps.Count)）。"
    }
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

if ($Action -eq "Reproduce") {
    if ($env:GITHUB_ACTIONS -ne "true") {
        throw "ProcDump 复现只允许在 GITHUB_ACTIONS=true 的临时 runner 上运行。"
    }
    Invoke-ProcDumpReproduction
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

$dumpDirectory = Join-Path $OutputDirectory "dumps"
$dumps = @()
try {
    # WER 写入 OutputDirectory；ProcDump 写入其中的 dumps 子目录。
    $dumps = @(Get-ChildItem -Path $OutputDirectory -Filter "*.dmp" -File -Recurse -ErrorAction SilentlyContinue)
    $dumps |
        Select-Object FullName, Length, LastWriteTime |
        ConvertTo-Json -Depth 3 |
        Set-Content -Path (Join-Path $OutputDirectory "dumps.json")
} catch {
    $_ | Out-String | Set-Content -Path (Join-Path $OutputDirectory "dumps.error.txt")
}

$cdbPath = $null
$cdb = Get-Command cdb.exe -ErrorAction SilentlyContinue
if ($null -ne $cdb) {
    $cdbPath = if (-not [string]::IsNullOrWhiteSpace($cdb.Source)) { $cdb.Source } else { $cdb.Path }
}
if ([string]::IsNullOrWhiteSpace($cdbPath) -and -not [string]::IsNullOrWhiteSpace(${env:ProgramFiles(x86)})) {
    $kitsCdbPath = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Debuggers\x64\cdb.exe"
    if (Test-Path -LiteralPath $kitsCdbPath -PathType Leaf) {
        $cdbPath = $kitsCdbPath
    }
}

if ([string]::IsNullOrWhiteSpace($cdbPath)) {
    Set-Content -Path (Join-Path $OutputDirectory "cdb-not-found.txt") -Value "cdb.exe was not found on PATH or the standard Windows Kits x64 debugger path; dump stack/module analysis was skipped."
} else {
    $cdbDirectory = Join-Path $OutputDirectory "cdb"
    New-Item -ItemType Directory -Force -Path $cdbDirectory | Out-Null
    foreach ($dump in $dumps) {
        $safeName = ($dump.BaseName -replace "[^A-Za-z0-9._-]", "_")
        $analysisPath = Join-Path $cdbDirectory "$safeName.txt"
        try {
            $commands = ".ecxr; k; lm; q"
            & $cdbPath -z $dump.FullName -c $commands 2>&1 | Tee-Object -FilePath $analysisPath
            "exit_code=$LASTEXITCODE" | Add-Content -Path $analysisPath
        } catch {
            $_ | Out-String | Set-Content -Path $analysisPath
        }
    }
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
