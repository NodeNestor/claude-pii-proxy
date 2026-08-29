# Ensure the PII proxy is running (Windows)
# Uses the venv created by install.ps1 (proxy\.venv\Scripts\python.exe)

$ErrorActionPreference = "SilentlyContinue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProxyDir = Join-Path $ScriptDir "..\proxy"
$VenvPython = Join-Path $ProxyDir ".venv\Scripts\python.exe"
$ClaudeDir = Join-Path $env:USERPROFILE ".claude"
$PidFile = Join-Path $ClaudeDir "pii-proxy.pid"
$VerFile = Join-Path $ClaudeDir "pii-proxy.version"
$HookLog = Join-Path $ClaudeDir "pii-proxy-hook.log"
$ProxyLog = Join-Path $ClaudeDir "pii-proxy.log"
$Port = if ($env:PII_PROXY_PORT) { $env:PII_PROXY_PORT } else { "5599" }
$ProxyUrl = "http://127.0.0.1:$Port"
$PluginJson = Join-Path $ScriptDir "..\.claude-plugin\plugin.json"
$CurrentVersion = if (Test-Path $PluginJson) { (Get-Content $PluginJson -Raw | ConvertFrom-Json).version } else { "unknown" }

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $HookLog -Value "[$ts] $msg"
}

Log "Hook started. ProxyDir=$ProxyDir"

# Wire ourselves into Claude Code via HTTPS_PROXY + NODE_EXTRA_CA_CERTS (NOT
# ANTHROPIC_BASE_URL, which trips the Remote Control / GrowthBook gate). CA
# generation, HTTPS_PROXY single-owner ownership, chaining with rolling-context,
# plugin defaults and stale-base_url cleanup all live in wire.py — one tested
# implementation shared with the sh hook and rolling-context.
$SettingsFile = Join-Path $ClaudeDir "settings.json"
$WirePython = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
try {
    $wireOut = & $WirePython (Join-Path $ProxyDir "wire.py") --name pii-proxy --settings $SettingsFile 2>&1
    foreach ($line in $wireOut) { Log "wire: $line" }
} catch {
    Log "WARNING: wire.py failed to update settings.json: $_"
}

# Pick the python interpreter
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
Log "Using interpreter: $Python"

# Already running?
if (Test-Path $PidFile) {
    $savedPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($savedPid) {
        $proc = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
        if ($proc) {
            $runningVersion = if (Test-Path $VerFile) { Get-Content $VerFile -ErrorAction SilentlyContinue } else { "" }
            if ($runningVersion -eq $CurrentVersion) {
                Log "Proxy already running (PID $savedPid, v$runningVersion)"
                exit 0
            }
            Log "Version changed ($runningVersion -> $CurrentVersion), restarting (PID $savedPid)"
            Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item $VerFile -Force -ErrorAction SilentlyContinue
}

Log "Starting proxy with $Python ..."
$proc = Start-Process -FilePath $Python -ArgumentList "server.py" `
    -WorkingDirectory $ProxyDir `
    -RedirectStandardOutput $ProxyLog -RedirectStandardError "$ProxyLog.err" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File -FilePath $PidFile -NoNewline
$CurrentVersion | Out-File -FilePath $VerFile -NoNewline
Log "Proxy started with PID $($proc.Id) (v$CurrentVersion)"

exit 0
