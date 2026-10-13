# P16: 部品ごとの分類の精度（BERT の 2 値・LM ヘッドの 9 クラス・組み合わせ）を、評価セット v3 で分けて出す。学習しない・端末も使わない（PC だけ・1 分）。
# 設計: DoC/設計書_P16_部品ごとの分類精度.md
# P13 の結果（work\p13\）が要る。
#
#   .\scripts\run_p16.ps1
# 進み具合は work\logs\p16-<日時>.log
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
$log = Join-Path $logDir ("p16-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P16 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p16_parts' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P16 が失敗しました（ログ: $log）" }
& $py '-B' '-m' 'app.p16_stopper' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P16（止める力の比べ）が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P16\parts.md・stopper.md" -ForegroundColor Green
