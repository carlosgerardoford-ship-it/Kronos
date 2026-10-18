# Kronos - Windows setup (CPU only).
# Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File install_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Python 3.10-3.12 with tkinter. 3.13 is not supported: pandas==2.2.2 (pinned in requirements.txt) has no wheel for it.
function Test-Py($exe, $pyargs) {
    try {
        $out = & $exe @pyargs -c "import sys, tkinter; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return "$out".Trim()
    } catch {
        return $null
    }
}

$cands = @(
    @{ exe = "py"; pyargs = @("-3.12") },
    @{ exe = "py"; pyargs = @("-3.11") },
    @{ exe = "py"; pyargs = @("-3.10") },
    @{ exe = "python"; pyargs = @() }
)
$py = $null
foreach ($c in $cands) {
    $v = Test-Py $c.exe $c.pyargs
    if ($v -in @("3.10", "3.11", "3.12")) { $py = $c; Write-Host "Using Python $v"; break }
}
if (-not $py) {
    Write-Host "Python 3.10, 3.11 or 3.12 (with tkinter) was not found."
    Write-Host "Install Python 3.12 from https://www.python.org/downloads/ (keep 'tcl/tk and IDLE' checked) and run this again."
    exit 1
}

function Run-Checked($exe, $cmdargs) {
    & $exe @cmdargs
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $exe $cmdargs" }
}

$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Run-Checked $py.exe ($py.pyargs + @("-m", "venv", ".venv"))
}
Run-Checked $vpy @("-m", "pip", "install", "--upgrade", "pip")
# CPU build of PyTorch first, so requirements.txt does not pull the CUDA build
Run-Checked $vpy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
Run-Checked $vpy @("-m", "pip", "install", "-r", "requirements.txt")
Run-Checked $vpy @("-c", "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())")

Write-Host ""
Write-Host "Done. Open the interface with run_gui.bat"
