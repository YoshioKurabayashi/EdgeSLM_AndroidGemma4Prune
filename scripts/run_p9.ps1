# P9: 引数の取り出しを補う層（app\p9_arguments.py）の効果を測る。学習しない・端末も使わない（PC だけ・数十秒）。P7・P8 の結果が要る。
# 設計: DoC/設計書_P9_引数の取り出しを補う.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
#
#   .\scripts\run_p9.ps1
# 進み具合は work\logs\p9-<日時>.log
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
$log = Join-Path $logDir ("p9-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P9 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p9_args' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P9 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P9\args_compare.md" -ForegroundColor Green
