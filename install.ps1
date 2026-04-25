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

Write-Host "[3/4] Configuring Claude Code settings.json..."
$SettingsFile = Join-Path $ClaudeDir "settings.json"
if (-not (Test-Path $ClaudeDir)) {
    New-Item -ItemType Directory -Path $ClaudeDir -Force | Out-Null
}
try {
    if (Test-Path $SettingsFile) {
        $settings = Get-Content $SettingsFile -Raw | ConvertFrom-Json
    } else {
        $settings = [PSCustomObject]@{}
    }
    if (-not ($settings | Get-Member -Name "env" -MemberType NoteProperty)) {
        $settings | Add-Member -NotePropertyName "env" -NotePropertyValue ([PSCustomObject]@{})
    }
    $existing = $null
    if ($settings.env | Get-Member -Name "ANTHROPIC_BASE_URL" -MemberType NoteProperty) {
        $existing = $settings.env.ANTHROPIC_BASE_URL
    }
    if (-not $existing) {
        $settings.env | Add-Member -NotePropertyName "ANTHROPIC_BASE_URL" -NotePropertyValue $ProxyUrl -Force
        Write-Host "  Set ANTHROPIC_BASE_URL=$ProxyUrl"
    } elseif ($existing -notmatch "127\.0\.0\.1.*$Port") {
        $settings.env | Add-Member -NotePropertyName "PII_PROXY_UPSTREAM" -NotePropertyValue $existing -Force
        $settings.env | Add-Member -NotePropertyName "ANTHROPIC_BASE_URL" -NotePropertyValue $ProxyUrl -Force
        Write-Host "  Chaining: ANTHROPIC_BASE_URL=$ProxyUrl -> upstream=$existing"
    } else {
        Write-Host "  ANTHROPIC_BASE_URL already set"
    }
    $defaults = @{
        "PII_PROXY_PORT"      = "5599"
        "PII_PROXY_MIN_SCORE" = "0.5"
        "PII_PROXY_MODEL"     = "openai/privacy-filter"
        "PII_PROXY_WARMUP"    = "1"
    }
    foreach ($key in $defaults.Keys) {
        if (-not ($settings.env | Get-Member -Name $key -MemberType NoteProperty)) {
            $settings.env | Add-Member -NotePropertyName $key -NotePropertyValue $defaults[$key]
        }
    }
    $settings | ConvertTo-Json -Depth 10 | Set-Content $SettingsFile -Encoding UTF8
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
