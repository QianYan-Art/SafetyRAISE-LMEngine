[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SafetensorsModelDir,

    [string]$RsinferExe = "",
    [string]$LlamaExe = "",
    [string]$LlamaBenchExe = "",
    [string]$GgufModelPath = "",
    [string]$Prompt = "Say hello in one sentence.",
    [int]$MaxTokens = 128,
    [int]$Repeat = 1,
    [double]$Temperature = 0.6,
    [double]$TopP = 0.95,
    [int]$TopK = 20,
    [bool]$Chat = $true,
    [string[]]$RsinferDevices = @("cpu"),
    [int]$GpuLayers = 0,
    [int]$LlamaBenchPromptTokens = 8,
    [string[]]$LlamaExtraArgs = @(),
    [string]$OutputDir = "target\benchmarks",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Name does not exist: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Test-IsUnderPath {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Parent
    )

    $candidateFull = [System.IO.Path]::GetFullPath($Candidate).TrimEnd('\', '/')
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    return $candidateFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $candidateFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        $candidateFull.StartsWith($parentFull + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-SafetensorsModelDir {
    param([Parameter(Mandatory = $true)][string]$ModelDir)

    foreach ($required in @("config.json", "tokenizer.json")) {
        $path = Join-Path $ModelDir $required
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Safetensors model directory is missing ${required}: $ModelDir"
        }
    }

    $single = Join-Path $ModelDir "model.safetensors"
    $index = Join-Path $ModelDir "model.safetensors.index.json"
    if (-not (Test-Path -LiteralPath $single -PathType Leaf) -and -not (Test-Path -LiteralPath $index -PathType Leaf)) {
        throw "Safetensors model directory must contain model.safetensors or model.safetensors.index.json: $ModelDir"
    }
}

function Format-Command {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $quoted = @($FilePath) + $Arguments | ForEach-Object {
        if ($_ -match '\s|["]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }
    return ($quoted -join " ")
}

function Invoke-BenchmarkCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "[$Name] $commandText"

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $FilePath @Arguments 2>&1
    $exitCode = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 0 }
    $stopwatch.Stop()

    $output | Set-Content -LiteralPath $LogPath -Encoding UTF8
    return [pscustomobject]@{
        name = $Name
        command = $commandText
        exit_code = $exitCode
        wall_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        log_path = $LogPath
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$safetensorsDir = Resolve-ExistingPath -Path $SafetensorsModelDir -Name "SafetensorsModelDir"
Assert-SafetensorsModelDir -ModelDir $safetensorsDir

$ggufPath = ""
if ($GgufModelPath) {
    $ggufPath = Resolve-ExistingPath -Path $GgufModelPath -Name "GgufModelPath"
    if ((Get-Item -LiteralPath $ggufPath).PSIsContainer) {
        throw "GgufModelPath must be a file, not a directory: $ggufPath"
    }
}

$resolvedLlamaExe = ""
if ($LlamaExe) {
    $resolvedLlamaExe = Resolve-ExistingPath -Path $LlamaExe -Name "LlamaExe"
}

$resolvedLlamaBenchExe = ""
if ($LlamaBenchExe) {
    $resolvedLlamaBenchExe = Resolve-ExistingPath -Path $LlamaBenchExe -Name "LlamaBenchExe"
}

$resolvedRsinferExe = ""
if ($RsinferExe) {
    $resolvedRsinferExe = Resolve-ExistingPath -Path $RsinferExe -Name "RsinferExe"
} else {
    $candidate = Join-Path $repoRoot "target\release\rsinfer.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $resolvedRsinferExe = (Resolve-Path -LiteralPath $candidate).Path
    }
}

[string[]]$normalizedRsinferDevices = @(
    $RsinferDevices |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ }
) | Select-Object -Unique
if (@($normalizedRsinferDevices).Count -eq 0) {
    throw "RsinferDevices must contain at least one device."
}
foreach ($device in $normalizedRsinferDevices) {
    if ($device -notin @("cpu", "auto", "hybrid")) {
        throw "Unsupported rsinfer device '$device'. Expected one of: cpu, auto, hybrid."
    }
}
if ($GpuLayers -lt 0) {
    throw "GpuLayers must be non-negative."
}
if ($LlamaBenchPromptTokens -lt 1) {
    throw "LlamaBenchPromptTokens must be positive."
}

$outputFull = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
if (Test-IsUnderPath -Candidate $outputFull -Parent $safetensorsDir) {
    throw "OutputDir must not be inside SafetensorsModelDir: $outputFull"
}
if ($ggufPath) {
    $ggufParent = Split-Path -Parent $ggufPath
    if (Test-IsUnderPath -Candidate $outputFull -Parent $ggufParent) {
        throw "OutputDir must not be inside GGUF model directory: $outputFull"
    }
}

$commonRsinferArgs = @(
    "--model-path", $safetensorsDir,
    "-p", $Prompt,
    "-n", "$MaxTokens",
    "--temperature", "$Temperature",
    "--top-p", "$TopP",
    "--top-k", "$TopK",
    "--verbose"
)
if ($Chat) {
    $commonRsinferArgs += "--chat"
}

$rsinferCommands = New-Object System.Collections.Generic.List[object]
foreach ($device in $normalizedRsinferDevices) {
    $deviceArgs = $commonRsinferArgs + @("--device", $device)
    if ($device -ne "cpu" -and $GpuLayers -gt 0) {
        $deviceArgs += @("--gpu-layers", "$GpuLayers")
    }

    if ($resolvedRsinferExe) {
        $rsinferCommands.Add([pscustomobject]@{ name = "rsinfer-$device"; file = $resolvedRsinferExe; args = $deviceArgs })
    } else {
        $rsinferCommands.Add([pscustomobject]@{ name = "rsinfer-$device"; file = "cargo"; args = @("run", "--release", "--") + $deviceArgs })
    }
}

$llamaCommand = $null
$llamaBenchCommand = $null
if ($resolvedLlamaBenchExe -and $ggufPath) {
    $llamaBenchArgs = @(
        "-m", $ggufPath,
        "-p", "$LlamaBenchPromptTokens",
        "-n", "$MaxTokens",
        "-r", "1",
        "--no-warmup",
        "-o", "json"
    )
    $llamaBenchCommand = [pscustomobject]@{ file = $resolvedLlamaBenchExe; args = $llamaBenchArgs }
} elseif ($resolvedLlamaExe -and $ggufPath) {
    $llamaArgs = @(
        "-m", $ggufPath,
        "-p", $Prompt,
        "-n", "$MaxTokens",
        "--temp", "$Temperature",
        "--top-p", "$TopP",
        "--top-k", "$TopK"
    ) + $LlamaExtraArgs
    $llamaCommand = [pscustomobject]@{ file = $resolvedLlamaExe; args = $llamaArgs }
}

Write-Host "Read-only model inputs:"
Write-Host "  safetensors: $safetensorsDir"
if ($ggufPath) { Write-Host "  gguf:        $ggufPath" }
Write-Host "Benchmark output directory: $outputFull"
Write-Host "This script never writes into the model input directories."

Write-Host ""
Write-Host "Planned rsinfer commands:"
foreach ($command in $rsinferCommands) {
    Write-Host "  $($command.name): $(Format-Command -FilePath $command.file -Arguments $command.args)"
}
if ($llamaCommand) {
    Write-Host ""
    Write-Host "Planned llama.cpp command:"
    Write-Host (Format-Command -FilePath $llamaCommand.file -Arguments $llamaCommand.args)
}
if ($llamaBenchCommand) {
    Write-Host ""
    Write-Host "Planned llama.cpp benchmark command:"
    Write-Host (Format-Command -FilePath $llamaBenchCommand.file -Arguments $llamaBenchCommand.args)
} elseif ($LlamaExe -or $LlamaBenchExe -or $GgufModelPath) {
    Write-Host ""
    Write-Host "llama.cpp benchmark skipped: provide -LlamaBenchExe or both -LlamaExe and -GgufModelPath."
}

if ($DryRun) {
    Write-Host ""
    Write-Host "DryRun: no benchmark commands executed and no output files written."
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$results = New-Object System.Collections.Generic.List[object]

for ($i = 1; $i -le $Repeat; $i++) {
    foreach ($command in $rsinferCommands) {
        $results.Add((Invoke-BenchmarkCommand `
            -Name "$($command.name)-$i" `
            -FilePath $command.file `
            -Arguments $command.args `
            -LogPath (Join-Path $outputFull "$runStamp-$($command.name)-$i.log")))
    }

    if ($llamaCommand) {
        $results.Add((Invoke-BenchmarkCommand `
            -Name "llama-$i" `
            -FilePath $llamaCommand.file `
            -Arguments $llamaCommand.args `
            -LogPath (Join-Path $outputFull "$runStamp-llama-$i.log")))
    }

    if ($llamaBenchCommand) {
        $results.Add((Invoke-BenchmarkCommand `
            -Name "llama-bench-$i" `
            -FilePath $llamaBenchCommand.file `
            -Arguments $llamaBenchCommand.args `
            -LogPath (Join-Path $outputFull "$runStamp-llama-bench-$i.json")))
    }
}

$summaryPath = Join-Path $outputFull "$runStamp-summary.json"
$summary = [pscustomobject]@{
    created_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    safetensors_model_dir = $safetensorsDir
    gguf_model_path = $ggufPath
    prompt = $Prompt
    max_tokens = $MaxTokens
    repeat = $Repeat
    temperature = $Temperature
    top_p = $TopP
    top_k = $TopK
    chat = $Chat
    rsinfer_devices = $normalizedRsinferDevices
    gpu_layers = $GpuLayers
    llama_bench_prompt_tokens = $LlamaBenchPromptTokens
    results = $results
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "Summary written: $summaryPath"
$results | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
