# Install the PII Proxy plugin for Claude Code (Windows)
#
# Run: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProxyDir  = Join-Path $ScriptDir "proxy"
$Venv      = Join-Path $ProxyDir ".venv"
$VenvPy    = Join-Path $Venv "Scripts\python.exe"
$Port      = if ($env:PII_PROXY_PORT) { $env:PII_PROXY_PORT } else { "5599" }
$ProxyUrl  = "http://127.0.0.1:$Port"
$ClaudeDir = Join-Path $env:USERPROFILE ".claude"

Write-Host "=== PII Proxy Installer (Windows) ==="
Write-Host ""

Write-Host "[1/4] Checking Python..."
try {
    $pyVersion = python --version 2>&1
    Write-Host "  Found $pyVersion"
} catch {
    Write-Host "  ERROR: Python not found. Install Python 3.9+ and try again."
    exit 1
}

Write-Host "[2/4] Creating venv and installing deps..."
if (-not (Test-Path $VenvPy)) {
    python -m venv $Venv
}
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install -r (Join-Path $ProxyDir "requirements.txt") --quiet
Write-Host "  Deps installed in $Venv"
Write-Host "  (For NVIDIA GPUs install onnxruntime-gpu instead:"
Write-Host "     $VenvPy -m pip uninstall -y onnxruntime; $VenvPy -m pip install onnxruntime-gpu)"

# Wire into Claude Code via HTTPS_PROXY + NODE_EXTRA_CA_CERTS (NOT
# ANTHROPIC_BASE_URL, which trips the Remote Control / GrowthBook gate).
Write-Host "[3/4] Configuring Claude Code settings.json..."
$SettingsFile = Join-Path $ClaudeDir "settings.json"
if (-not (Test-Path $ClaudeDir)) {
    New-Item -ItemType Directory -Path $ClaudeDir -Force | Out-Null
}
try {
    $wireOut = & $VenvPy (Join-Path $ProxyDir "wire.py") --name pii-proxy --settings $SettingsFile 2>&1
    foreach ($line in $wireOut) { Write-Host "  $line" }
    Write-Host "  Settings written to $SettingsFile"
} catch {
    Write-Host "  ERROR: Could not update settings.json: $_"
    exit 1
}

Write-Host "[4/4] Registering Claude Code plugin..."
$PluginsDir = Join-Path $ClaudeDir "plugins"
$PluginLink = Join-Path $PluginsDir "pii-proxy"
if (-not (Test-Path $PluginsDir)) {
    New-Item -ItemType Directory -Path $PluginsDir -Force | Out-Null
}
if (Test-Path $PluginLink) {
    Remove-Item $PluginLink -Recurse -Force
}
cmd /c mklink /J "$PluginLink" "$ScriptDir" | Out-Null
Write-Host "  Plugin linked at $PluginLink"

Write-Host ""
Write-Host "=== Installation Complete ==="
Write-Host ""
Write-Host "The proxy will auto-start when you launch Claude Code."
Write-Host "Manual start:  $VenvPy $ProxyDir\server.py"
Write-Host ""
Write-Host "First request will download the openai/privacy-filter ONNX weights"
Write-Host "(quantized variant preferred — usually 50-150MB)."
Write-Host ""
Write-Host "Start a new Claude Code session to activate."
