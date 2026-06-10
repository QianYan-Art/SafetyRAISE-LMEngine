[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SafetensorsModelDir,

    [string]$RsinferExe = "",
    [string]$AExe = "",
    [string]$BExe = "",
    [string]$AName = "hybrid-q8",
    [string]$BName = "resident-hybrid-q8",
    [switch]$AResident,
    [switch]$BResident = $true,
    [string[]]$AExtraArgs = @(),
    [string[]]$BExtraArgs = @(),
    [string]$Prompt = "Say hello in one sentence.",
    [int]$MaxTokens = 32,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.95,
    [int]$TopK = 20,
    [int]$Runs = 6,
    [double]$TiePercent = 2.0,
    [int]$MaxGpuUtilPercent = 5,
    [string]$OutputDir = "target\benchmarks",
    [switch]$SkipGpuIdleCheck,
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

function Get-GpuUtilizationPercent {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw "nvidia-smi not found; use -SkipGpuIdleCheck to bypass."
    }

    $raw = & $nvidiaSmi.Source --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed: $raw"
    }

    $values = @($raw | ForEach-Object {
        $text = "$_".Trim()
        if ($text) { [int]$text }
    })
    if ($values.Count -eq 0) {
        throw "nvidia-smi returned no GPU utilization values."
    }
    return ($values | Measure-Object -Maximum).Maximum
}

function Get-Median {
    param([Parameter(Mandatory = $true)][double[]]$Values)

    if ($Values.Count -eq 0) {
        throw "Cannot compute median of empty values."
    }
    $sorted = @($Values | Sort-Object)
    $mid = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) {
        return [double]$sorted[$mid]
    }
    return [double](($sorted[$mid - 1] + $sorted[$mid]) / 2.0)
}

function Parse-AvgDecodeMs {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    $match = Select-String -LiteralPath $LogPath -Pattern "avg_decode_forward_per_token=([0-9]+(?:\.[0-9]+)?)" | Select-Object -Last 1
    if (-not $match) {
        throw "Missing avg_decode_forward_per_token in log: $LogPath"
    }
    return [double]$match.Matches[0].Groups[1].Value
}

function Invoke-RsinferVariant {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][bool]$Resident,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $oldResident = [Environment]::GetEnvironmentVariable("RSINFER_INTERNAL_RESIDENT_DECODE_PROTOTYPE", "Process")
    try {
        if ($Resident) {
            [Environment]::SetEnvironmentVariable("RSINFER_INTERNAL_RESIDENT_DECODE_PROTOTYPE", "1", "Process")
        } else {
            [Environment]::SetEnvironmentVariable("RSINFER_INTERNAL_RESIDENT_DECODE_PROTOTYPE", $null, "Process")
        }

        Write-Host "[$Variant] $(Format-Command -FilePath $FilePath -Arguments $Arguments)"
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $FilePath @Arguments 2>&1
        $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
        $stopwatch.Stop()
        $output | Set-Content -LiteralPath $LogPath -Encoding UTF8

        if ($exitCode -ne 0) {
            throw "$Variant failed with exit code $exitCode; log: $LogPath"
        }

        return [pscustomobject]@{
            variant = $Variant
            resident = $Resident
            wall_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
            avg_decode_forward_per_token = Parse-AvgDecodeMs -LogPath $LogPath
            log_path = $LogPath
        }
    } finally {
        [Environment]::SetEnvironmentVariable("RSINFER_INTERNAL_RESIDENT_DECODE_PROTOTYPE", $oldResident, "Process")
    }
}

if ($Runs -lt 2) {
    throw "Runs must be at least 2 because the first run is discarded as warm-up."
}
if ($MaxTokens -lt 1) {
    throw "MaxTokens must be positive."
}
if ($TiePercent -lt 0) {
    throw "TiePercent must be non-negative."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$safetensorsDir = Resolve-ExistingPath -Path $SafetensorsModelDir -Name "SafetensorsModelDir"
Assert-SafetensorsModelDir -ModelDir $safetensorsDir

$resolvedRsinferExe = ""
if ($RsinferExe) {
    $resolvedRsinferExe = Resolve-ExistingPath -Path $RsinferExe -Name "RsinferExe"
} else {
    $candidate = Join-Path $repoRoot "target\release\rsinfer.exe"
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "RsinferExe not provided and target release binary not found. Run cargo build --release first."
    }
    $resolvedRsinferExe = (Resolve-Path -LiteralPath $candidate).Path
}
$resolvedAExe = if ($AExe) {
    Resolve-ExistingPath -Path $AExe -Name "AExe"
} else {
    $resolvedRsinferExe
}
$resolvedBExe = if ($BExe) {
    Resolve-ExistingPath -Path $BExe -Name "BExe"
} else {
    $resolvedRsinferExe
}

$outputFull = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
if (Test-IsUnderPath -Candidate $outputFull -Parent $safetensorsDir) {
    throw "OutputDir must not be inside SafetensorsModelDir: $outputFull"
}

$commonArgs = @(
    "--model-path", $safetensorsDir,
    "--chat",
    "-p", $Prompt,
    "-n", "$MaxTokens",
    "--temperature", "$Temperature",
    "--top-p", "$TopP",
    "--top-k", "$TopK",
    "--device", "hybrid",
    "--quantization", "q8",
    "--profile-tokens",
    "--verbose"
)
$aArgs = $commonArgs + $AExtraArgs
$bArgs = $commonArgs + $BExtraArgs

Write-Host "A/B decode benchmark protocol:"
Write-Host "  model: $safetensorsDir"
Write-Host "  exe:   $resolvedRsinferExe"
Write-Host "  A exe: $resolvedAExe"
Write-Host "  B exe: $resolvedBExe"
Write-Host "  prompt: $Prompt"
Write-Host "  runs: $Runs each, interleaved; discard first run per variant"
Write-Host "  decision: median avg_decode_forward_per_token; <= $TiePercent% is TIE"
Write-Host "  A: $AName resident=$([bool]$AResident)"
Write-Host "  B: $BName resident=$([bool]$BResident)"
Write-Host "  output: $outputFull"

if (-not $SkipGpuIdleCheck) {
    $gpuUtil = Get-GpuUtilizationPercent
    Write-Host "  gpu_utilization_percent: $gpuUtil"
    if ($gpuUtil -ge $MaxGpuUtilPercent) {
        throw "GPU utilization is ${gpuUtil}%, expected < ${MaxGpuUtilPercent}%. Close GPU-heavy apps or use -SkipGpuIdleCheck."
    }
}

if ($DryRun) {
    Write-Host ""
    Write-Host "DryRun commands:"
    Write-Host "  A: $(Format-Command -FilePath $resolvedAExe -Arguments $aArgs)"
    Write-Host "  B: $(Format-Command -FilePath $resolvedBExe -Arguments $bArgs)"
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$results = New-Object System.Collections.Generic.List[object]

for ($i = 1; $i -le $Runs; $i++) {
    $results.Add((Invoke-RsinferVariant `
        -Variant "$AName-$i" `
        -FilePath $resolvedAExe `
        -Arguments $aArgs `
        -Resident ([bool]$AResident) `
        -LogPath (Join-Path $outputFull "$runStamp-A-$i-$AName.log")))

    $results.Add((Invoke-RsinferVariant `
        -Variant "$BName-$i" `
        -FilePath $resolvedBExe `
        -Arguments $bArgs `
        -Resident ([bool]$BResident) `
        -LogPath (Join-Path $outputFull "$runStamp-B-$i-$BName.log")))
}

$aRaw = @($results | Where-Object { $_.variant -like "$AName-*" } | ForEach-Object { [double]$_.avg_decode_forward_per_token })
$bRaw = @($results | Where-Object { $_.variant -like "$BName-*" } | ForEach-Object { [double]$_.avg_decode_forward_per_token })
$aEffective = @($aRaw | Select-Object -Skip 1)
$bEffective = @($bRaw | Select-Object -Skip 1)
$aMedian = Get-Median -Values $aEffective
$bMedian = Get-Median -Values $bEffective
$deltaPercent = (($bMedian - $aMedian) / $aMedian) * 100.0
$decision = if ([math]::Abs($deltaPercent) -le $TiePercent) {
    "TIE"
} elseif ($deltaPercent -lt 0.0) {
    "WIN"
} else {
    "LOSS"
}

$summaryPath = Join-Path $outputFull "$runStamp-ab-summary.json"
$summary = [pscustomobject]@{
    created_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    safetensors_model_dir = $safetensorsDir
    rsinfer_exe = $resolvedRsinferExe
    a_exe = $resolvedAExe
    b_exe = $resolvedBExe
    prompt = $Prompt
    max_tokens = $MaxTokens
    temperature = $Temperature
    top_p = $TopP
    top_k = $TopK
    runs = $Runs
    warmup_discarded_per_variant = 1
    tie_percent = $TiePercent
    a = [pscustomobject]@{
        name = $AName
        resident = [bool]$AResident
        raw_ms = $aRaw
        effective_ms = $aEffective
        median_ms = [math]::Round($aMedian, 3)
    }
    b = [pscustomobject]@{
        name = $BName
        resident = [bool]$BResident
        raw_ms = $bRaw
        effective_ms = $bEffective
        median_ms = [math]::Round($bMedian, 3)
    }
    delta_percent = [math]::Round($deltaPercent, 3)
    decision = $decision
    results = $results
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "A raw ms: $($aRaw -join ', ')"
Write-Host "B raw ms: $($bRaw -join ', ')"
Write-Host "A median ms: $([math]::Round($aMedian, 3))"
Write-Host "B median ms: $([math]::Round($bMedian, 3))"
Write-Host "Delta percent (B vs A): $([math]::Round($deltaPercent, 3))%"
Write-Host "Decision: $decision"
Write-Host "Summary written: $summaryPath"
