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

# Always update settings.json first (even if proxy is already running)
$SettingsFile = Join-Path $ClaudeDir "settings.json"
try {
    if (Test-Path $SettingsFile) {
        $settings = Get-Content $SettingsFile -Raw | ConvertFrom-Json
    } else {
        $settings = [PSCustomObject]@{}
    }

    if (-not ($settings | Get-Member -Name "env" -MemberType NoteProperty)) {
        $settings | Add-Member -NotePropertyName "env" -NotePropertyValue ([PSCustomObject]@{})
    }

    $existingUrl = $null
    if ($settings.env | Get-Member -Name "ANTHROPIC_BASE_URL" -MemberType NoteProperty) {
        $existingUrl = $settings.env.ANTHROPIC_BASE_URL
    }

    if (-not $existingUrl) {
        $settings.env | Add-Member -NotePropertyName "ANTHROPIC_BASE_URL" -NotePropertyValue $ProxyUrl -Force
        Log "Set ANTHROPIC_BASE_URL=$ProxyUrl (settings.json)"
    } elseif ($existingUrl -notmatch "127\.0\.0\.1.*$Port") {
        # Save existing URL as upstream so we chain transparently
        $settings.env | Add-Member -NotePropertyName "PII_PROXY_UPSTREAM" -NotePropertyValue $existingUrl -Force
        $settings.env | Add-Member -NotePropertyName "ANTHROPIC_BASE_URL" -NotePropertyValue $ProxyUrl -Force
        Log "Chaining: upstream=$existingUrl"
    } else {
        Log "ANTHROPIC_BASE_URL already set to PII proxy"
    }

    # Plugin defaults
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
} catch {
    Log "WARNING: Could not update settings.json: $_"
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
