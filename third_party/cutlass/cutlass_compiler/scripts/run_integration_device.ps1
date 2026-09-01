[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,
    [Parameter(Position = 1)]
    [string]$TargetSM = "",
    [Parameter(Position = 2)]
    [ValidateSet("isa", "bin", "fatbin", "llvm")]
    [string]$CompilationTarget = "bin",
    [Parameter(Position = 3)]
    [string]$Entry = "main"
)

$ErrorActionPreference = "Stop"
$InputPath = [IO.Path]::GetFullPath($InputPath)
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
    throw "input file not found: $InputPath"
}

$buildRoot = $env:CUTLASS_COMPILER_BUILD_DIR
if ([string]::IsNullOrWhiteSpace($buildRoot)) {
    throw "CUTLASS_COMPILER_BUILD_DIR is not set"
}
if ([string]::IsNullOrWhiteSpace($TargetSM)) {
    $TargetSM = $env:CUTLASS_COMPILER_DEVICE_SM
}
if ([string]::IsNullOrWhiteSpace($TargetSM)) {
    $TargetSM = $env:TEST_GPU_ARCH
}
if ([string]::IsNullOrWhiteSpace($TargetSM)) {
    $TargetSM = "sm_90"
}

function Resolve-Executable([string]$Override, [string[]]$Candidates, [string]$Name) {
    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        return $Override
    }
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "$Name was not found"
}

$compiler = Resolve-Executable $env:CUTLASS_COMPILER @(
    (Join-Path $buildRoot "tools\cutlass-compiler\cutlass-compiler.exe"),
    (Join-Path $buildRoot "cutlass_compiler\tools\cutlass-compiler\cutlass-compiler.exe")
) "cutlass-compiler"
$runnerCandidates = @()
if (-not [string]::IsNullOrWhiteSpace($env:LLVM_TOOLS_DIR)) {
    $runnerCandidates += Join-Path $env:LLVM_TOOLS_DIR "mlir-runner.exe"
}
$runner = Resolve-Executable $env:MLIR_RUNNER $runnerCandidates "mlir-runner"

foreach ($required in @(
    @{ Name = "MLIR_CUDA_RUNTIME"; Value = $env:MLIR_CUDA_RUNTIME },
    @{ Name = "MLIR_RUNNER_UTILS"; Value = $env:MLIR_RUNNER_UTILS },
    @{ Name = "MLIR_C_RUNNER_UTILS"; Value = $env:MLIR_C_RUNNER_UTILS }
)) {
    if ([string]::IsNullOrWhiteSpace($required.Value) -or
        -not (Test-Path -LiteralPath $required.Value -PathType Leaf)) {
        throw "$($required.Name) was not found"
    }
}

$tempDir = Join-Path ([IO.Path]::GetTempPath()) ("cutlass-integration-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir | Out-Null
$compiled = Join-Path $tempDir "compiled.mlir"
try {
    & $compiler `
        -cute-fold-static -cute-expand-ops -cute-to-base `
        -base-prepare -one-shot-convert-to-llvm `
        "-attach-nvvm-target=chip=$TargetSM" `
        "-emit-gpu-binary=compilation-target=$CompilationTarget" `
        $InputPath -o $compiled
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $runner -e $Entry -entry-point-result=void `
        "--shared-libs=$($env:MLIR_CUDA_RUNTIME)" `
        "--shared-libs=$($env:MLIR_RUNNER_UTILS)" `
        "--shared-libs=$($env:MLIR_C_RUNNER_UTILS)" `
        $compiled
    exit $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
