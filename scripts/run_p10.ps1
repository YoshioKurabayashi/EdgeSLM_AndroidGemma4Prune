# P10: 棄却の閾値の決め方を変えて、不要な聞き返しと重大な誤りの釣り合いを見る。学習しない・端末も使わない（PC だけ・1〜2 分）。
# 設計: DoC/設計書_P10_棄却の閾値の決め方.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
#
#   .\scripts\run_p10.ps1
# 進み具合は work\logs\p10-<日時>.log
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
$log = Join-Path $logDir ("p10-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P10 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p10_reject' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P10 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P10\reject_rule.md" -ForegroundColor Green
