[CmdletBinding()]
param([string]$Python = "")

# Windows compatibility name retained for the Linux make_envs.sh entry point.
# On Windows, use the already-configured current Python environment; do not
# create a second venv or install the proprietary reference wheel.
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}
$Python = (Resolve-Path $Python).Path

& $Python -c @"
import sys
import nanobind
import torch
from cuda.bindings import driver
print('Python:', sys.executable)
print('nanobind:', nanobind.__version__)
print('torch:', torch.__version__)
print('cuda-python bindings:', driver.__file__)
"@
if ($LASTEXITCODE -ne 0) {
    throw "the current Python environment is missing nanobind, torch, or cuda-python"
}
Write-Host "Using current Python environment: $Python"
