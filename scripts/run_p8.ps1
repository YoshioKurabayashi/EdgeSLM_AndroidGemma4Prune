# P8: 「3 つがそろったときだけ実行する」流れ（BERT が対象・LM ヘッドの確信度・引数が取り出せる）で、評価セット v2 を測り直す。
# 学習しない・端末も使わない（PC だけ・数十秒）。P7 の結果（work\p7）が要る。
# 設計: DoC/設計書_P8_三つがそろったときだけ実行.md
#
#   .\scripts\run_p8.ps1
# 進み具合は work\logs\p8-<日時>.log
param(
    [string]$Layers = "17,26,35"
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
# numpy と PyYAML だけで動く。EdgeSLM_MultiHead_Multi の venv を使うだけ（Dispatcher の決定的な処理を読むため）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p8-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P8 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p8_gate' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P8 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P8\gate.md" -ForegroundColor Green
