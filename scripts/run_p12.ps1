# P12: BERT の断る閾値の決め方と、フラグを効かせる範囲を変えて、対象を断る件数と KPI を見る。学習しない・端末も使わない（PC だけ・2〜3 分）。
# 設計: DoC/設計書_P12_BERTの断る閾値の決め方.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# P11 の結果（work\p11\text_scores_*.json）が要る。
#
#   .\scripts\run_p12.ps1
#   .\scripts\run_p12.ps1 -Berts "C"      # BERT の案をしぼる（既定は C,A）
# 進み具合は work\logs\p12-<日時>.log
param(
    [string]$Layers = "17,26,35",
    [string]$Berts = "C,A"
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
$log = Join-Path $logDir ("p12-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P12 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p12_block' '--layers' $Layers '--berts' $Berts 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P12 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P12\block_rule.md" -ForegroundColor Green
