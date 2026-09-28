$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root 'data/server.pid'
if (-not (Test-Path $pidFile)) {
  Write-Output '未找到本项目服务的 pid 文件，无需停止'
  exit 0
}
$serverPid = [int](Get-Content $pidFile)
$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($process) {
  Stop-Process -Id $serverPid
  Write-Output "已停止本项目服务 pid=$serverPid"
} else {
  Write-Output "进程 pid=$serverPid 已不存在"
}
Remove-Item -Force $pidFile
