param(
    [Parameter(Position=0)][string]$Command = "check",
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$MoreArgs
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required. Install it and rerun." }
    & ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & ".venv\Scripts\python.exe" -m pip install -e ".\CausalForge[mujoco]" "imageio>=2.34" pillow
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    New-Item ".venv\\.ready" -ItemType File -Force | Out-Null
}
& ".venv\Scripts\python.exe" "CausalForge\tools\mujoco_lab.py" $Command @MoreArgs
exit $LASTEXITCODE
