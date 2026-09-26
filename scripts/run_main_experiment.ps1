param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("local", "service")]
    [string]$Profile,
    [string]$BaseUrl,
    [ValidateRange(1, 690)]
    [int]$BlockLimit = 1,
    [ValidateRange(1, 4)]
    [int]$ShardCount = 1,
    [ValidateRange(0, 3)]
    [int]$ShardIndex = 0,
    [string]$ReferenceTokenizerBaseUrl = "http://host.docker.internal:8091/v1",
    [ValidateSet("disabled", "resume-once")]
    [string]$CheckpointPolicy = "resume-once",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$schedulePath = Join-Path $projectRoot "manifests\main_schedule.json"
$activationPath = Join-Path $projectRoot "manifests\main_experiment_activation_v3.json"
$parallelConfigPath = Join-Path $projectRoot "configs\execution\parallel_execution_v2.json"
$catalog = Join-Path $projectRoot "manifests\agent_tasks\main.jsonl"
$dataset = Join-Path $workspaceRoot "datasets\LinuxFLBench\dataset\LINUXFLBENCH_dataset.jsonl"
$pythonExe = "D:\Anaconda\envs\SANER\python.exe"
$image = "python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534"
$sourceVolume = "saner-main-kernel-sources"

if ($Profile -eq "local") {
    $model = "Qwen3-14B"
    if ([string]::IsNullOrWhiteSpace($BaseUrl)) { $BaseUrl = "http://host.docker.internal:8000/v1" }
}
else {
    $model = "qwen-plus-2025-12-01"
    if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
        throw "Service profile requires the verified Alibaba Cloud BaseUrl"
    }
}
$baseUrl = $BaseUrl.Trim().TrimEnd("/")
$parsedUrl = $null
if (-not [Uri]::TryCreate($baseUrl, [UriKind]::Absolute, [ref]$parsedUrl) -or
    $parsedUrl.Scheme -notin @("http", "https") -or -not $baseUrl.EndsWith("/v1")) {
    throw "BaseUrl must be an absolute HTTP(S) OpenAI-compatible URL ending in /v1"
}

$schedule = Get-Content -LiteralPath $schedulePath -Raw | ConvertFrom-Json
if ($schedule.status -ne "frozen_execution_blocked_pending_validation_gates" -or
    [int]$schedule.task_count -ne 230 -or [int]$schedule.block_count -ne 1380 -or
    [int]$schedule.episode_count -ne 9660) {
    throw "Main schedule is not the frozen 230-task design"
}
$allModelBlocks = @($schedule.blocks | Where-Object { $_.model -eq $model })
if ($allModelBlocks.Count -ne 690) { throw "Frozen schedule does not contain exactly 690 $model blocks" }
$parallelConfig = Get-Content -LiteralPath $parallelConfigPath -Raw | ConvertFrom-Json
if ($Profile -eq "local") {
    if ($ShardCount -ne 1 -or $ShardIndex -ne 0) {
        throw "Local execution remains frozen at one GPU worker: use -ShardCount 1 -ShardIndex 0"
    }
}
else {
    if ($ShardCount -ne [int]$parallelConfig.service_shard_count) {
        throw "Service execution requires the predeclared $($parallelConfig.service_shard_count)-shard plan"
    }
    if ($ShardIndex -ge $ShardCount) { throw "ShardIndex must be smaller than ShardCount" }
}
$blocks = @()
for ($ordinal = 0; $ordinal -lt $allModelBlocks.Count; $ordinal += 1) {
    if (($ordinal % $ShardCount) -eq $ShardIndex) { $blocks += $allModelBlocks[$ordinal] }
}
$expectedShardBlocks = @($parallelConfig.service_blocks_per_shard)[$ShardIndex]
if ($Profile -eq "local") { $expectedShardBlocks = 690 }
if ($blocks.Count -ne [int]$expectedShardBlocks) {
    throw "Shard partition is not balanced: expected $expectedShardBlocks blocks, found $($blocks.Count)"
}

$pending = @($blocks | Where-Object {
    $dir = Join-Path $workspaceRoot ($_.output_directory -replace "/", "\")
    -not (Test-Path -LiteralPath (Join-Path $dir "summary.json"))
} | Select-Object -First $BlockLimit)
if ($PlanOnly) {
    [pscustomobject]@{
        mode = "plan_only"
        experiment_id = "saner-main-v1"
        profile = $Profile
        model = $model
        total_model_blocks = $allModelBlocks.Count
        shard_count = $ShardCount
        shard_index = $ShardIndex
        shard_blocks = $blocks.Count
        completed_blocks = $blocks.Count - @($blocks | Where-Object {
            $dir = Join-Path $workspaceRoot ($_.output_directory -replace "/", "\")
            -not (Test-Path -LiteralPath (Join-Path $dir "summary.json"))
        }).Count
        pending_blocks_selected = $pending.Count
        blocks = @($pending | Select-Object block_index,task_id,kernel_version,repetition,output_directory)
    } | ConvertTo-Json -Depth 5
    exit 0
}

if (-not (Test-Path -LiteralPath $activationPath -PathType Leaf)) {
    throw "Main execution activation record is missing; complete all three launch gates first"
}
$activation = Get-Content -LiteralPath $activationPath -Raw | ConvertFrom-Json
if ($activation.status -ne "ready_for_execution" -or $activation.experiment_id -ne "saner-main-v1") {
    throw "Main execution activation record is not ready"
}
& $pythonExe (Join-Path $projectRoot "scripts\activate_main_experiment_v3.py") --check
if ($LASTEXITCODE -ne 0) { throw "Main activation record or executable configuration changed" }
& $pythonExe (Join-Path $projectRoot "scripts\freeze_main_experiment.py") --check
if ($LASTEXITCODE -ne 0) { throw "Main schedule or frozen configuration changed" }

docker info --format "{{.ServerVersion}} {{.OSType}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop Linux engine is not available" }
$volume = docker volume inspect $sourceVolume --format "{{.Name}}" 2>$null
if ($LASTEXITCODE -ne 0 -or $volume.Trim() -ne $sourceVolume) {
    throw "Audited main source volume is unavailable: $sourceVolume"
}
if ($Profile -eq "service" -and [string]::IsNullOrWhiteSpace($env:SANER_SERVICE_API_KEY)) {
    throw "SANER_SERVICE_API_KEY is empty"
}

$attempted = 0
$lockRoot = Join-Path $workspaceRoot "runs\main\saner-main-v1\.launcher-locks"
New-Item -ItemType Directory -Path $lockRoot -Force | Out-Null
$lockPath = Join-Path $lockRoot "$Profile-shard-$ShardIndex-of-$ShardCount.lock"
$lockStream = $null
try {
    $lockStream = [IO.File]::Open(
        $lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None
    )
}
catch [IO.IOException] {
    throw "This $Profile shard is already running: $ShardIndex of $ShardCount"
}
try {
foreach ($block in $blocks) {
    $outputDir = Join-Path $workspaceRoot ($block.output_directory -replace "/", "\")
    $summary = Join-Path $outputDir "summary.json"
    $scores = Join-Path $outputDir "scores.json"
    if (Test-Path -LiteralPath $summary) {
        if (-not (Test-Path -LiteralPath $scores)) {
            & $pythonExe (Join-Path $projectRoot "scripts\score_main_run.py") `
                --summary $summary --dataset $dataset `
                --dataset-manifest (Join-Path $projectRoot "manifests\dataset_manifest.json") `
                --output $scores
            if ($LASTEXITCODE -ne 0) { throw "Deferred main scoring failed" }
        }
        continue
    }
    if ($attempted -ge $BlockLimit) { break }
    $attempted += 1
    $sourceInfo = $schedule.sources.PSObject.Properties[$block.kernel_version].Value
    $functionIndexContainer = "/project/" + ($sourceInfo.function_index -replace "\\", "/")
    $sourceContainer = $sourceInfo.source_directory
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "[$attempted/$BlockLimit] $Profile task $($block.task_id), repetition $($block.repetition)"

    $dockerArgs = @(
        "run", "--rm", "--pull=never", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--add-host", "host.docker.internal:host-gateway",
        "--mount", "type=bind,source=$projectRoot,target=/project,readonly",
        "--mount", "type=bind,source=$catalog,target=/input/tasks.jsonl,readonly",
        "--mount", "type=bind,source=$outputDir,target=/output",
        "--mount", "type=volume,source=$sourceVolume,target=/sources,readonly",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m",
        "--env", "PYTHONPATH=/project/src"
    )
    if ($Profile -eq "service") { $dockerArgs += @("--env", "SANER_SERVICE_API_KEY") }
    $dockerArgs += @(
        $image, "python", "/project/scripts/run_main_block.py",
        "--profile", $Profile, "--source", $sourceContainer,
        "--function-index", $functionIndexContainer,
        "--task-catalog", "/input/tasks.jsonl", "--schedule", "/project/manifests/main_schedule.json",
        "--task-id", $block.task_id, "--base-url", $baseUrl, "--model", $model,
        "--repetition", [string]$block.repetition, "--output-dir", "/output",
        "--token-counter", "offline-reference", "--tokenizer-base-url", $ReferenceTokenizerBaseUrl,
        "--tokenizer-model", "Qwen3-14B", "--checkpoint-policy", $CheckpointPolicy
    )
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Main block failed; logs and stable checkpoint were retained"
    }
    & $pythonExe (Join-Path $projectRoot "scripts\score_main_run.py") `
        --summary $summary --dataset $dataset `
        --dataset-manifest (Join-Path $projectRoot "manifests\dataset_manifest.json") `
        --output $scores
    if ($LASTEXITCODE -ne 0) { throw "Deferred main scoring failed" }
}
Write-Host "Completed $attempted new $Profile main block(s) in shard $ShardIndex/$ShardCount; BlockLimit was $BlockLimit."
}
finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
}
