[CmdletBinding()]
param(
    [string]$Python = "",
    [int]$Jobs = 0,
    [switch]$NoCcache
)

# Native Windows/MSVC counterpart of build.sh.  The oracle is a small
# nanobind extension over header-only cutegen; it does not need CUDA or LLVM.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}
$Python = (Resolve-Path $Python).Path
$build = Join-Path $root "build-oracle"

foreach ($tool in @("cmake.exe", "ninja.exe", "cl.exe")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool was not found. Run this script from a VS Developer PowerShell."
    }
}
$launcherArgs = @()
if (-not $NoCcache) {
    $ccache = Get-Command ccache.exe -ErrorAction SilentlyContinue
    if (-not $ccache) {
        throw "ccache.exe was not found. Install ccache or pass -NoCcache explicitly."
    }
    $env:CCACHE_COMPILERCHECK = "content"
    $launcherArgs = @(
        "-DCMAKE_C_COMPILER_LAUNCHER=$($ccache.Source)",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=$($ccache.Source)"
    )
    Write-Host "Using ccache: $($ccache.Source)"
} else {
    # Clear launchers from an earlier configure when -NoCcache is used
    # with an existing build tree.
    $launcherArgs = @(
        "-DCMAKE_C_COMPILER_LAUNCHER=",
        "-DCMAKE_CXX_COMPILER_LAUNCHER="
    )
}
if (-not (Test-Path $Python)) {
    throw "Python was not found at '$Python'. Pass -Python <path> or put python.exe on PATH."
}
& $Python -c "import nanobind"
if ($LASTEXITCODE -ne 0) { throw "nanobind is not installed in $Python" }
if ($Jobs -le 0) { $Jobs = 4 }

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", (Join-Path $PSScriptRoot "."),
    "-B", $build,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DPython3_EXECUTABLE=$Python"
) + $launcherArgs
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cutegen oracle CMake configure failed" }
& cmake --build $build --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "cutegen oracle build failed" }

& $Python -c @"
import sys
sys.path.insert(0, r'$build')
import _cutegen_oracle as O
r = O.selfcheck()
assert r == '(32,4):(1,8)', r
d = O.count_dynamics('(256,?):(?,1)')
assert d == 2, d
print('[cutegen_oracle] selfcheck OK:', r, '| dynamics =', d)
"@
if ($LASTEXITCODE -ne 0) { throw "cutegen oracle self-check failed" }
Write-Host "Built $build\_cutegen_oracle.pyd for $Python"
