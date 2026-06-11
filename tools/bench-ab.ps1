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
    [int]$MaxTokens = 256,
    [switch]$Quick,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.95,
    [int]$TopK = 20,
    [int]$Runs = 6,
    [double]$TiePercent = 2.0,
    [int]$MaxGpuUtilPercent = 5,
    [int]$RepeatNGramThreshold = 8,
    [int]$RepeatNGramSize = 4,
    [int]$RepeatTailTokens = 128,
    [int]$SentinelCompareTokens = 64,
    [int]$CpuAnchorCalibrationTokens = 256,
    [string]$OutputDir = "target\benchmarks",
    [switch]$SkipGpuIdleCheck,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ChineseSmokePrompt = -join ([char[]](0x4F60, 0x597D))

if ($Quick -and -not $PSBoundParameters.ContainsKey("MaxTokens")) {
    $MaxTokens = 32
}

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

function Parse-GeneratedTokenIds {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    $match = Select-String -LiteralPath $LogPath -Pattern "^profile\.generated_token_ids\s*(.*)$" | Select-Object -Last 1
    if (-not $match) {
        throw "Missing profile.generated_token_ids in log: $LogPath"
    }
    $raw = $match.Matches[0].Groups[1].Value.Trim()
    if (-not $raw) {
        return @()
    }
    return @($raw -split "," | Where-Object { $_ -ne "" } | ForEach-Object { [uint32]$_ })
}

function Parse-GeneratedText {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    $text = Get-Content -Raw -LiteralPath $LogPath
    $startMarker = "=== 开始生成 ==="
    $endMarker = "--- 生成完成"
    if ($text.Contains($startMarker)) {
        $text = ($text -split [regex]::Escape($startMarker), 2)[1]
    }
    if ($text.Contains($endMarker)) {
        $text = ($text -split [regex]::Escape($endMarker), 2)[0]
    }
    $text = ($text -replace "(?m)^profile\..*$", "").Trim()
    return $text
}

function Test-RepeatedNGram {
    param(
        [Parameter(Mandatory = $true)][uint32[]]$TokenIds,
        [int]$N = 4,
        [int]$TailTokens = 128,
        [int]$Threshold = 8
    )

    $tailStart = [Math]::Max(0, $TokenIds.Count - $TailTokens)
    $tail = @($TokenIds | Select-Object -Skip $tailStart)
    $counts = @{}
    $maxCount = 0
    $maxGram = ""
    if ($tail.Count -ge $N) {
        for ($i = 0; $i -le $tail.Count - $N; $i++) {
            $gram = (@($tail[$i..($i + $N - 1)]) -join ",")
            if (-not $counts.ContainsKey($gram)) {
                $counts[$gram] = 0
            }
            $counts[$gram]++
            if ($counts[$gram] -gt $maxCount) {
                $maxCount = $counts[$gram]
                $maxGram = $gram
            }
        }
    }

    return [pscustomobject]@{
        pass = ($maxCount -lt $Threshold)
        max_repeat = $maxCount
        max_ngram = $maxGram
        threshold = $Threshold
        ngram_size = $N
        tail_tokens = $TailTokens
    }
}

function Get-FirstTokenDifference {
    param(
        [Parameter(Mandatory = $true)][uint32[]]$A,
        [Parameter(Mandatory = $true)][uint32[]]$B
    )

    $limit = [Math]::Min($A.Count, $B.Count)
    for ($i = 0; $i -lt $limit; $i++) {
        if ($A[$i] -ne $B[$i]) {
            return ($i + 1)
        }
    }
    if ($A.Count -ne $B.Count) {
        return ($limit + 1)
    }
    return $null
}

function Test-TokenPrefixEqual {
    param(
        [Parameter(Mandatory = $true)][uint32[]]$A,
        [Parameter(Mandatory = $true)][uint32[]]$B,
        [Parameter(Mandatory = $true)][int]$Length
    )

    if ($A.Count -lt $Length -or $B.Count -lt $Length) {
        return $false
    }
    for ($i = 0; $i -lt $Length; $i++) {
        if ($A[$i] -ne $B[$i]) {
            return $false
        }
    }
    return $true
}

function New-RsinferArgs {
    param(
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][int]$TokenCount,
        [Parameter(Mandatory = $true)][double]$Temp,
        [Parameter(Mandatory = $true)][double]$TopPValue,
        [Parameter(Mandatory = $true)][int]$TopKValue,
        [Parameter(Mandatory = $true)][bool]$Resident,
        [string[]]$ExtraArgs = @(),
        [string]$Device = "hybrid",
        [string]$Quantization = "q8",
        [switch]$ProfileLayers
    )

    $args = @(
        "--model-path", $ModelDir,
        "--chat",
        "-p", $PromptText,
        "-n", "$TokenCount",
        "--temperature", "$Temp",
        "--top-p", "$TopPValue",
        "--top-k", "$TopKValue",
        "--device", $Device,
        "--quantization", $Quantization,
        "--profile-tokens",
        "--verbose"
    )
    if ($ProfileLayers) {
        $args += @("--profile-layers")
    }
    if ($Resident) {
        $args += @("--resident")
    } else {
        $args += @("--no-resident")
    }
    return $args + $ExtraArgs
}

function Invoke-RsinferVariant {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][bool]$Resident,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    Write-Host "[$Variant] $(Format-Command -FilePath $FilePath -Arguments $Arguments)"
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $FilePath @Arguments 2>&1
    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    $stopwatch.Stop()
    $output | Set-Content -LiteralPath $LogPath -Encoding UTF8

    if ($exitCode -ne 0) {
        throw "$Variant failed with exit code $exitCode; log: $LogPath"
    }

    $tokenIds = @(Parse-GeneratedTokenIds -LogPath $LogPath)
    return [pscustomobject]@{
        variant = $Variant
        resident = $Resident
        wall_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        avg_decode_forward_per_token = Parse-AvgDecodeMs -LogPath $LogPath
        token_count = $tokenIds.Count
        token_ids = $tokenIds
        log_path = $LogPath
    }
}

function Invoke-Sentinels {
    param(
        [Parameter(Mandatory = $true)][object[]]$BenchmarkResults,
        [Parameter(Mandatory = $true)][string]$RunStamp,
        [Parameter(Mandatory = $true)][string]$OutputFull,
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][string[]]$AArgs,
        [Parameter(Mandatory = $true)][string[]]$BArgs
    )

    $perRun = @($BenchmarkResults | ForEach-Object {
        $ids = [uint32[]]@($_.token_ids)
        $rep = Test-RepeatedNGram -TokenIds $ids -N $RepeatNGramSize -TailTokens $RepeatTailTokens -Threshold $RepeatNGramThreshold
        [pscustomobject]@{
            variant = $_.variant
            pass = $rep.pass
            max_repeat = $rep.max_repeat
            max_ngram = $rep.max_ngram
            log_path = $_.log_path
        }
    })

    $abAPath = Join-Path $OutputFull "$RunStamp-sentinel-ab-A.log"
    $abBPath = Join-Path $OutputFull "$RunStamp-sentinel-ab-B.log"
    $abAArgs = New-RsinferArgs -ModelDir $ModelDir -PromptText $PromptText -TokenCount $SentinelCompareTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident ([bool]$AResident) -ExtraArgs $AExtraArgs
    $abBArgs = New-RsinferArgs -ModelDir $ModelDir -PromptText $PromptText -TokenCount $SentinelCompareTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident ([bool]$BResident) -ExtraArgs $BExtraArgs
    $abA = Invoke-RsinferVariant -Variant "sentinel-ab-$AName" -FilePath $Exe -Arguments $abAArgs -Resident ([bool]$AResident) -LogPath $abAPath
    $abB = Invoke-RsinferVariant -Variant "sentinel-ab-$BName" -FilePath $Exe -Arguments $abBArgs -Resident ([bool]$BResident) -LogPath $abBPath
    $abDiff = Get-FirstTokenDifference -A ([uint32[]]@($abA.token_ids)) -B ([uint32[]]@($abB.token_ids))
    $abPass = ($null -eq $abDiff) -and ($abA.token_ids.Count -eq $SentinelCompareTokens) -and ($abB.token_ids.Count -eq $SentinelCompareTokens)

    $cpuNoResidentPath = Join-Path $OutputFull "$RunStamp-sentinel-cpu-anchor-no-resident.log"
    $cpuAnchorPath = Join-Path $OutputFull "$RunStamp-sentinel-cpu-anchor-gpu-layers0.log"
    $noResidentArgs = New-RsinferArgs -ModelDir $ModelDir -PromptText $PromptText -TokenCount $CpuAnchorCalibrationTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident $false -ExtraArgs $AExtraArgs
    $cpuAnchorArgs = New-RsinferArgs -ModelDir $ModelDir -PromptText $PromptText -TokenCount $CpuAnchorCalibrationTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident $false -ExtraArgs @("--gpu-layers", "0")
    $noResident = Invoke-RsinferVariant -Variant "sentinel-cpu-anchor-no-resident" -FilePath $Exe -Arguments $noResidentArgs -Resident $false -LogPath $cpuNoResidentPath
    $cpuAnchor = Invoke-RsinferVariant -Variant "sentinel-cpu-anchor-gpu-layers0" -FilePath $Exe -Arguments $cpuAnchorArgs -Resident $false -LogPath $cpuAnchorPath
    $nCal = Get-FirstTokenDifference -A ([uint32[]]@($noResident.token_ids)) -B ([uint32[]]@($cpuAnchor.token_ids))
    $anchorCompareLength = if ($null -eq $nCal -or $nCal -ge 64) {
        64
    } elseif ($nCal -lt 16) {
        0
    } else {
        [int]$nCal
    }
    $anchorFailJudging = ($anchorCompareLength -gt 0)
    $anchorPass = if ($anchorCompareLength -eq 0) {
        $true
    } else {
        Test-TokenPrefixEqual -A ([uint32[]]@($noResident.token_ids)) -B ([uint32[]]@($cpuAnchor.token_ids)) -Length $anchorCompareLength
    }

    $smokePath = Join-Path $OutputFull "$RunStamp-sentinel-chinese-smoke.log"
    $smokeArgs = New-RsinferArgs -ModelDir $ModelDir -PromptText $ChineseSmokePrompt -TokenCount $SentinelCompareTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident $false
    $smoke = Invoke-RsinferVariant -Variant "sentinel-chinese-smoke" -FilePath $Exe -Arguments $smokeArgs -Resident $false -LogPath $smokePath
    $smokeText = Parse-GeneratedText -LogPath $smokePath
    $smokeRep = Test-RepeatedNGram -TokenIds ([uint32[]]@($smoke.token_ids)) -N $RepeatNGramSize -TailTokens $RepeatTailTokens -Threshold $RepeatNGramThreshold
    $smokePass = ($smokeText.Length -gt 0) -and $smokeRep.pass

    $pass = (-not (@($perRun | Where-Object { -not $_.pass }))) -and $abPass -and $anchorPass -and $smokePass
    return [pscustomobject]@{
        pass = [bool]$pass
        per_run_repetition = $perRun
        ab_greedy_equal = [pscustomobject]@{
            pass = [bool]$abPass
            compare_tokens = $SentinelCompareTokens
            first_difference_position = $abDiff
            a_log_path = $abAPath
            b_log_path = $abBPath
        }
        cpu_anchor = [pscustomobject]@{
            pass = [bool]$anchorPass
            calibration_tokens = $CpuAnchorCalibrationTokens
            first_difference_position = $nCal
            compare_length = $anchorCompareLength
            fail_judging_enabled = [bool]$anchorFailJudging
            no_resident_log_path = $cpuNoResidentPath
            cpu_anchor_log_path = $cpuAnchorPath
        }
        chinese_smoke = [pscustomobject]@{
            pass = [bool]$smokePass
            nonempty = [bool]($smokeText.Length -gt 0)
            repetition = $smokeRep
            log_path = $smokePath
        }
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
$resolvedAExe = if ($AExe) { Resolve-ExistingPath -Path $AExe -Name "AExe" } else { $resolvedRsinferExe }
$resolvedBExe = if ($BExe) { Resolve-ExistingPath -Path $BExe -Name "BExe" } else { $resolvedRsinferExe }

$outputFull = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
if (Test-IsUnderPath -Candidate $outputFull -Parent $safetensorsDir) {
    throw "OutputDir must not be inside SafetensorsModelDir: $outputFull"
}

$aArgs = New-RsinferArgs -ModelDir $safetensorsDir -PromptText $Prompt -TokenCount $MaxTokens -Temp $Temperature -TopPValue $TopP -TopKValue $TopK -Resident ([bool]$AResident) -ExtraArgs $AExtraArgs
$bArgs = New-RsinferArgs -ModelDir $safetensorsDir -PromptText $Prompt -TokenCount $MaxTokens -Temp $Temperature -TopPValue $TopP -TopKValue $TopK -Resident ([bool]$BResident) -ExtraArgs $BExtraArgs

Write-Host "A/B decode benchmark protocol v2:"
Write-Host "  model: $safetensorsDir"
Write-Host "  exe:   $resolvedRsinferExe"
Write-Host "  A exe: $resolvedAExe"
Write-Host "  B exe: $resolvedBExe"
Write-Host "  prompt: $Prompt"
Write-Host "  max_tokens: $MaxTokens$(if ($Quick) { ' (Quick)' } else { '' })"
Write-Host "  runs: $Runs each, interleaved; discard first run per variant"
Write-Host "  decision: median avg_decode_forward_per_token; <= $TiePercent% is TIE"
Write-Host "  sentinels: per-run ${RepeatNGramSize}-gram repeat < $RepeatNGramThreshold in last $RepeatTailTokens tokens; A/B greedy $SentinelCompareTokens-token equality; CPU anchor calibration; Chinese smoke"
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
    Write-Host "  sentinel A/B A: $(Format-Command -FilePath $resolvedAExe -Arguments (New-RsinferArgs -ModelDir $safetensorsDir -PromptText $Prompt -TokenCount $SentinelCompareTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident ([bool]$AResident) -ExtraArgs $AExtraArgs))"
    Write-Host "  sentinel A/B B: $(Format-Command -FilePath $resolvedBExe -Arguments (New-RsinferArgs -ModelDir $safetensorsDir -PromptText $Prompt -TokenCount $SentinelCompareTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident ([bool]$BResident) -ExtraArgs $BExtraArgs))"
    Write-Host "  sentinel CPU anchor no-resident: $(Format-Command -FilePath $resolvedRsinferExe -Arguments (New-RsinferArgs -ModelDir $safetensorsDir -PromptText $Prompt -TokenCount $CpuAnchorCalibrationTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident $false -ExtraArgs $AExtraArgs))"
    Write-Host "  sentinel CPU anchor gpu-layers0: $(Format-Command -FilePath $resolvedRsinferExe -Arguments (New-RsinferArgs -ModelDir $safetensorsDir -PromptText $Prompt -TokenCount $CpuAnchorCalibrationTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident $false -ExtraArgs @('--gpu-layers','0')))"
    Write-Host "  sentinel Chinese smoke: $(Format-Command -FilePath $resolvedRsinferExe -Arguments (New-RsinferArgs -ModelDir $safetensorsDir -PromptText $ChineseSmokePrompt -TokenCount $SentinelCompareTokens -Temp 0.0 -TopPValue $TopP -TopKValue $TopK -Resident $false))"
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

$sentinel = Invoke-Sentinels -BenchmarkResults ($results.ToArray()) -RunStamp $runStamp -OutputFull $outputFull -Exe $resolvedRsinferExe -ModelDir $safetensorsDir -PromptText $Prompt -AArgs $aArgs -BArgs $bArgs

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
if (-not $sentinel.pass) {
    $decision = "SENTINEL_FAIL"
}

$summaryPath = Join-Path $outputFull "$runStamp-ab-summary.json"
$summary = [pscustomobject]@{
    protocol = "v2"
    created_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    safetensors_model_dir = $safetensorsDir
    rsinfer_exe = $resolvedRsinferExe
    a_exe = $resolvedAExe
    b_exe = $resolvedBExe
    prompt = $Prompt
    max_tokens = $MaxTokens
    quick = [bool]$Quick
    temperature = $Temperature
    top_p = $TopP
    top_k = $TopK
    runs = $Runs
    warmup_discarded_per_variant = 1
    tie_percent = $TiePercent
    sentinel = $sentinel
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
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "A raw ms: $($aRaw -join ', ')"
Write-Host "B raw ms: $($bRaw -join ', ')"
Write-Host "A median ms: $([math]::Round($aMedian, 3))"
Write-Host "B median ms: $([math]::Round($bMedian, 3))"
Write-Host "Delta percent (B vs A): $([math]::Round($deltaPercent, 3))%"
Write-Host "Sentinel pass: $($sentinel.pass)"
Write-Host "Decision: $decision"
Write-Host "Summary written: $summaryPath"

if (-not $sentinel.pass) {
    throw "Sentinel failed; run is invalid. Summary written: $summaryPath"
}
