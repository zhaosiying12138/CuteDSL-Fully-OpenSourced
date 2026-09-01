[CmdletBinding()]
param(
    [switch]$IncludeSm120
)

# Native Windows counterpart of run_correctness.sh.  Runtime sm120 tests are
# opt-in because an sm86 device can generate sm120a PTX but cannot execute it.
$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $root
$python = (Get-Command python.exe -ErrorAction Stop).Source
if (-not (Test-Path $python)) {
    throw "the current Python environment is not available"
}

$failed = $false
Write-Host "=== [1/3] host/compiler tests (excluding sm120) ==="
& $python -m pytest -m "not sm120" -q
if ($LASTEXITCODE -ne 0) { $failed = $true }

if ($IncludeSm120) {
    Write-Host "=== [2/3] sm120 runtime tests (must run on a compatible sm120 GPU) ==="
    & $python -m pytest -m sm120 -q
    if ($LASTEXITCODE -ne 0) { $failed = $true }
} else {
    Write-Host "=== [2/3] sm120 runtime tests skipped (use -IncludeSm120 only on sm120) ==="
}

Write-Host "=== [3/3] selfcute dialect LIT ==="
& cmake --build (Join-Path $root "build-selfcute") --target check-selfcute-lit
if ($LASTEXITCODE -ne 0) { $failed = $true }

if ($failed) {
    Write-Host "RESULT: FAILURES"
    exit 1
}
Write-Host "RESULT: correctness gates passed"
exit 0
