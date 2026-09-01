[CmdletBinding()]
param(
    [int]$Jobs = 0,
    [string]$LLVMSource = "C:\llvm-project",
    [switch]$SkipSelfCuteTests,
    [switch]$NoCcache
)

# Native Windows/MSVC counterpart of build_compiler.sh.  The pinned LLVM build
# must already exist; this script does not download dependencies.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llvm = Join-Path $root "build-llvm"
$llvmSource = (Resolve-Path $LLVMSource).Path
$compiler = Join-Path $root "build-compiler"
$selfcute = Join-Path $root "build-selfcute"
$python = (Get-Command python.exe -ErrorAction Stop).Source
$llvmCmake = $llvm.Replace('\', '/')
$llvmSourceCmake = $llvmSource.Replace('\', '/')
$pythonCmake = $python.Replace('\', '/')

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
if (-not (Test-Path (Join-Path $llvm "bin\mlir-opt.exe"))) {
    throw "Pinned LLVM is missing. Run tools\build_pinned_llvm.ps1 first."
}
if ($Jobs -le 0) {
    $Jobs = 15
    if ($env:LLVM_BUILD_JOBS -and [int]::TryParse($env:LLVM_BUILD_JOBS, [ref]$parsed)) {
        $Jobs = $parsed
    }
}

$common = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLVM_DIR=$llvmCmake/lib/cmake/llvm",
    "-DMLIR_DIR=$llvmCmake/lib/cmake/mlir",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DCMAKE_CXX_STANDARD=20",
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
    "-DCMAKE_CXX_EXTENSIONS=OFF",
    "-DCMAKE_C_FLAGS=/utf-8",
    "-DCMAKE_CXX_FLAGS=/utf-8 /bigobj /permissive- /Zc:__cplusplus /Zc:preprocessor /Zc:externConstexpr /Zc:inline",
    "-DPython3_EXECUTABLE=$pythonCmake"
) + $launcherArgs

$compilerArgs = @(
    "-S", (Join-Path $root "third_party\cutlass\cutlass_compiler"),
    "-B", $compiler
) + $common + @(
    "-DCUTLASS_COMPILER_GTEST_SOURCE_DIR=$llvmSourceCmake/third-party/unittest"
)
& cmake @compilerArgs
if ($LASTEXITCODE -ne 0) { throw "cutlass_compiler CMake configure failed" }
& cmake --build $compiler --target cute-opt base-opt cutlass-compiler --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "cutlass_compiler build failed" }

# selfcute is deliberately a separate tree.  Its CMake now resolves the
# cutlass_compiler libraries as .a on Unix or .lib on MSVC.
$selfcuteArgs = @(
    "-S", (Join-Path $root "compiler"),
    "-B", $selfcute
) + $common + @(
    "-DSELF_CUTE_CUTLASS_COMPILER_ROOT=$compiler"
)
& cmake @selfcuteArgs
if ($LASTEXITCODE -ne 0) { throw "selfcute CMake configure failed" }
$targets = @("selfcute-opt")
if (-not $SkipSelfCuteTests) { $targets += "check-selfcute-lit" }
& cmake --build $selfcute --target $targets --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "selfcute build/test failed" }

Write-Host "Compiler tools are ready:"
Write-Host "  $compiler\tools\cutlass-compiler\cutlass-compiler.exe"
Write-Host "  $compiler\cute_ir\tools\cute-opt\cute-opt.exe"
Write-Host "  $selfcute\bin\selfcute-opt.exe"
