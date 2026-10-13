# P14: 対象の語の食い違いを確かめる規則（S4）を足して、評価セット v3・v2 で測る。学習しない・端末も使わない（PC だけ・1〜2 分）。
# 設計: DoC/設計書_P14_対象の語の食い違いを確かめる.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# P13 の結果（work\p13\）が要る。
#
#   .\scripts\run_p14.ps1
# 進み具合は work\logs\p14-<日時>.log
param(
    [string]$Layers = "17,26,35"
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
# numpy と PyYAML だけで動く。EdgeSLM_MultiHead_Multi の venv を使うだけ
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p14-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P14 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p14_eval' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P14 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P14\mismatch.md" -ForegroundColor Green
