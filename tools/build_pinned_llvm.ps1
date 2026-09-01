[CmdletBinding()]
param(
    [string]$LLVMCommit = "",
    [string]$LLVMSource = "C:\llvm-project",
    [int]$Jobs = 0,
    [switch]$ConfigureOnly,
    [switch]$NoCcache
)

# Native Windows/MSVC counterpart of build_pinned_llvm.sh.
# Run from a Visual Studio Developer PowerShell so cl.exe, link.exe and rc.exe
# are available to CMake.  This uses the existing C:\llvm-project checkout;
# it never creates a second LLVM repository.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llvmSource = (Resolve-Path $LLVMSource).Path
$build = Join-Path $root "build-llvm"

foreach ($tool in @("cmake.exe", "ninja.exe", "git.exe", "cl.exe")) {
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
        "-DCMAKE_CXX_COMPILER_LAUNCHER=$($ccache.Source)",
        "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON"
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

if ([string]::IsNullOrWhiteSpace($LLVMCommit)) {
    $LLVMCommit = (Get-Content (Join-Path $root "third_party\cutlass\cutlass_compiler\LLVM_COMMIT") -Raw).Trim()
}
if ($LLVMCommit -notmatch "^[0-9a-fA-F]{40}$") {
    throw "LLVM commit is not a 40-character commit id: $LLVMCommit"
}

if (-not (Test-Path (Join-Path $llvmSource ".git")) -or
    -not (Test-Path (Join-Path $llvmSource "llvm\CMakeLists.txt"))) {
    throw "$llvmSource is not an LLVM git checkout"
}
$dirty = & git -C $llvmSource status --porcelain
if ($dirty) {
    throw "$llvmSource has local changes; refusing to change its checkout"
}

& git -C $llvmSource cat-file -e "$LLVMCommit`^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Fetching LLVM commit $LLVMCommit into $llvmSource ..."
    & git -C $llvmSource fetch --depth 1 origin $LLVMCommit
    if ($LASTEXITCODE -ne 0) { throw "could not fetch LLVM commit $LLVMCommit" }
}
& git -C $llvmSource checkout --detach $LLVMCommit
if ($LASTEXITCODE -ne 0) { throw "could not check out LLVM commit $LLVMCommit" }

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", (Join-Path $llvmSource "llvm"),
    "-B", $build,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_STANDARD=20",
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
    "-DCMAKE_CXX_EXTENSIONS=OFF",
    "-DCMAKE_C_FLAGS=/utf-8",
    "-DCMAKE_CXX_FLAGS=/utf-8 /permissive- /Zc:__cplusplus /Zc:preprocessor /Zc:externConstexpr /Zc:inline",
    "-DLLVM_ENABLE_PROJECTS=mlir",
    "-DLLVM_TARGETS_TO_BUILD=Native;NVPTX",
    "-DLLVM_ENABLE_ASSERTIONS=ON",
    "-DLLVM_BUILD_UTILS=ON",
    "-DLLVM_BUILD_TOOLS=ON",
    "-DLLVM_INCLUDE_EXAMPLES=OFF",
    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF",
    "-DLLVM_INCLUDE_TESTS=OFF",
    "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF",
    "-DLLVM_ENABLE_DIA_SDK=OFF",
    "-DLLVM_ENABLE_ZSTD=OFF"
) + $launcherArgs
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "LLVM CMake configure failed" }
if ($ConfigureOnly) { return }

if ($Jobs -le 0) {
    $Jobs = 15
    if ($env:LLVM_BUILD_JOBS -and [int]::TryParse($env:LLVM_BUILD_JOBS, [ref]$parsed)) {
        $Jobs = $parsed
    }
}
Write-Host "Building the minimal LLVM/MLIR/NVPTX set with MSVC using $Jobs parallel jobs ..."
$targets = @(
    "llvm-libraries",
    "mlir-libraries",
    "FileCheck",
    "llvm-tblgen",
    "llc",
    "mlir-opt",
    "mlir-tblgen",
    "mlir-translate",
    "llvm-config"
)
& cmake --build $build --target $targets --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "LLVM build failed" }

Write-Host "LLVM is ready:"
Write-Host "  $build\bin\mlir-opt.exe"
Write-Host "  $build\bin\mlir-tblgen.exe"
Write-Host "  $build\bin\mlir-translate.exe"
Write-Host "  $build\bin\llc.exe"
