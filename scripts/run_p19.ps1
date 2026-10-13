# P19: 構成を 1 つに決める（BERT の使うモデル・棄却の閾値・規則 S5）。学習なし・端末なし（P18 の出力を読む）。
# 設計: DoC/設計書_P19_構成を確定する.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# 評価セット v3・v4 は「直す用」。ここの数字は合否ではない。
#
#   .\scripts\run_p19.ps1                 # 2 分ほど
# 進み具合は work\logs\p19-<日時>.log
param(
    [string]$Layers = "17,26,35"
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
# numpy と PyYAML だけ使う（EdgeSLM_MultiHead_Multi の venv を使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p19-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P19 =====（ログ: $log）" -ForegroundColor Cyan
& $py -B -m app.p19_config --layers $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P19 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P19\config.md" -ForegroundColor Green
