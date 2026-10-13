# P17: 流れのヘッドを 1,101 行に切り替える案と、不要な聞き返しを減らす手当て（引数の取り出し・規則の例外・確信度の閾値）を、評価セット v3 で測る。
# 学習しない・端末も使わない（PC だけ・1〜2 分）。設計: DoC/設計書_P17_不要な聞き返しを減らす.md
# P13 の結果（work\p13\）と、P7 のヘッド（work\p7\）が要る。
#
#   .\scripts\run_p17.ps1
# 進み具合は work\logs\p17-<日時>.log
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
$log = Join-Path $logDir ("p17-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P17 =====（ログ: $log）" -ForegroundColor Cyan
& $py '-B' '-m' 'app.p17_tune' '--layers' $Layers 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
if ($LASTEXITCODE -ne 0) { throw "P17 が失敗しました（ログ: $log）" }
Write-Host "完了  結果: results\P17\tune.md" -ForegroundColor Green
