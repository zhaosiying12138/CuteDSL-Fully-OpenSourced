[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,
    [Parameter(Position = 1)]
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

$tempDir = Join-Path ([IO.Path]::GetTempPath()) ("cutlass-integration-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir | Out-Null
$compiled = Join-Path $tempDir "compiled.mlir"
try {
    & $compiler `
        -cute-fold-static -cute-expand-ops -cute-to-base `
        -base-prepare -one-shot-convert-to-llvm `
        $InputPath -o $compiled
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $runner -e $Entry -entry-point-result=void $compiled
    exit $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
