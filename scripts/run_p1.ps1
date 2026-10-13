# P1: Gemma 4 E2B の層を減らしても分類が落ちないか（お試し・PC だけ）。
# 設計: DoC/設計書_P1_層ごとの分類性能.md
#
#   1) 35 層それぞれの h を取り出す（GPU・数分）           app.p1_extract_layers
#   2) 層の数ごとにヘッドの学習・評価・交差検証（1 つ 3 分ほど）  app.p1_heads
#   3) 並べる（CPU・数秒）                                  app.p1_compare
#   4) 誤実行・過剰な聞き返しを行ごとに調べる（-Errors のとき。1 つの層で 3 分ほど）  app.p1_errors
#
# 実行:  .\scripts\run_p1.ps1                      # 1)〜3)（1) は作成済みなら飛ばす）
#        .\scripts\run_p1.ps1 -Layers "14,15,16"   # 層の数を足す
#        .\scripts\run_p1.ps1 -Force               # 1) の取り出しとヘッドを作り直す
#        .\scripts\run_p1.ps1 -Layers "16,18,19,25,26,27" -Errors -ErrorLayers "17,24,28,35"   # 足して、誤りも調べる
# ログは work\logs\p1-<日時>.log
param(
    [string]$Layers = "",
    [switch]$Force,
    [switch]$Errors,
    [string]$ErrorLayers = "17,24,28,35",
    [string]$Device = "cuda"
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# EdgeSLM_MultiHead_Multi の venv（GPU の torch・transformers・ヘッドのコード）を使う（venv は読むだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "EdgeSLM_MultiHead_Multi の venv がありません: $py" }
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p1-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))

function Run-Step([string]$Title, [string[]]$A) {
    Write-Host "`n===== $Title =====" -ForegroundColor Cyan
    Add-Content -Path $log -Value "===== $Title =====" -Encoding utf8
    $t0 = Get-Date
    & $py @A 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
    $code = $LASTEXITCODE
    $s = "[{0}] exit={1} 所要={2} 分" -f $Title, $code, [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Add-Content -Path $log -Value $s -Encoding utf8
    Write-Host $s -ForegroundColor DarkGray
    if ($code -ne 0) { throw "$Title が失敗しました（ログ: $log）" }
}

$feat = Join-Path $repo 'work\p1\all_layers.npz'
if ($Force -or -not (Test-Path $feat)) { Run-Step "1) 35 層それぞれの h を取り出す（GPU）" @('-B', '-m', 'app.p1_extract_layers', '--device', $Device) }
else { Write-Host "1) 35 層の h は作成済みなので使い回す（作り直すなら -Force）" -ForegroundColor DarkGray }
$h = @('-B', '-m', 'app.p1_heads')
if ($Layers) { $h += @('--layers', $Layers) }
if ($Force) { $h += '--force' }
Run-Step "2) 層の数ごとにヘッドの学習・評価・交差検証" $h
Run-Step "3) 並べる" @('-B', '-m', 'app.p1_compare')
if ($Errors) { Run-Step "4) 誤実行・過剰な聞き返しを行ごとに" @('-B', '-m', 'app.p1_errors', '--layers', $ErrorLayers) }
Write-Host "完了  ログ: $log  結果: results\P1\compare_table.md" -ForegroundColor Green
