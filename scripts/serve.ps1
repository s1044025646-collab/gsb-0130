$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root 'data'
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$logFile = Join-Path $dataDir 'server.log'
$pidFile = Join-Path $dataDir 'server.pid'
if (Test-Path $pidFile) {
  $oldPid = [int](Get-Content $pidFile)
  $existing = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
  if ($existing) {
    Write-Output "服务已在运行 (pid=$oldPid)"
    exit 0
  }
}
$env:HMM_DB_PATH = Join-Path $dataDir 'app.sqlite'
$process = Start-Process -FilePath 'node' `
  -ArgumentList (Join-Path $root 'node_modules/tsx/dist/cli.mjs'), (Join-Path $root 'src/api/server.ts') `
  -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $logFile -RedirectStandardError (Join-Path $dataDir 'server.err.log')
Set-Content -Path $pidFile -Value $process.Id
Start-Sleep -Milliseconds 800
Get-Content $logFile
Write-Output "服务 pid=$($process.Id)，仅监听 127.0.0.1 自动端口"
