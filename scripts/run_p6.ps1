# P6: 学習済みのヘッドを seed・種類（mlp・linear）ごとに評価セット（challenge_v1）へ当て直す。学習しない・端末も使わない（1〜2 分）。続けて、閾値を動かしたときの正答率も出す。
# 設計: DoC/設計書_P6_ヘッドのseedを変えた判定.md
#
#   .\scripts\run_p6.ps1
# 進み具合は work\logs\p6-<日時>.log
param(
    [string]$Layers = "17,26,35"
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
# numpy だけで動く（torch は使わない）。EdgeSLM_Android の venv を使うだけ
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p6-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P6 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p6_seeds' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P6（seed ごとの判定）が失敗しました（ログ: $log）" }
& $py '-B' '-m' 'app.p6_sweep' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P6（閾値を動かす）が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P6\seeds_table.md・threshold_sweep.md" -ForegroundColor Green
