# P22・P23: 引数の値を測って直す（規則でどこまで届くか）。学習なし・端末なし。
# 設計: DoC/設計書_P22_引数の値を測って直す.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# 引数の段だけを測る（正しいツールを選べたとして、値が合うか）。端末は要らない。
#
#   .\scripts\run_p22.ps1                 # すべて（5 分ほど）
#   .\scripts\run_p22.ps1 -MeasureOnly    # 見ていない行（arg_eval_v1・v2）の測り直しだけ（1 分）
# 結果: results\P22\fix.md・argeval.md（P22）、results\P23\fix.md・argeval.md（P23）
# 進み具合は work\logs\p22-<日時>.log
param(
    [switch]$MeasureOnly
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
$log = Join-Path $logDir ("p22-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P22・P23 =====（ログ: $log）" -ForegroundColor Cyan
function Step([string]$Title, [string[]]$A) {
    Write-Host "`n--- $Title ---" -ForegroundColor Cyan
    Add-Content -Path $log -Value "--- $Title ---" -Encoding utf8
    $t0 = Get-Date
    & $py @A 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
    $code = $LASTEXITCODE
    $s = "[{0}] exit={1} 所要={2} 分" -f $Title, $code, [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Add-Content -Path $log -Value $s -Encoding utf8
    Write-Host $s -ForegroundColor DarkGray
    if ($code -ne 0) { throw "$Title が失敗しました（ログ: $log）" }
}
if (-not $MeasureOnly) {
    Step "正解の引数を書き出す（評価セット v5・v6 の操作系の対象）" @('-B', '-m', 'app.p22_labels')
    Step "P22 の直しを、これまでの行で確かめる" @('-B', '-m', 'app.p22_fix')
    Step "見ていない行 その 1（arg_eval_v1）を作る" @('-B', '-m', 'app.p22_argeval', 'build')
}
Step "見ていない行 その 1 で測る（P21・P22）" @('-B', '-m', 'app.p22_argeval', 'measure')
if (-not $MeasureOnly) {
    Step "P23 の整える層を、これまでの行で確かめる" @('-B', '-m', 'app.p23_fix')
    Step "見ていない行 その 2（arg_eval_v2）を作る" @('-B', '-m', 'app.p23_argeval', 'build')
}
Step "見ていない行 その 2 で測る（P21・P22・P23）" @('-B', '-m', 'app.p23_argeval', 'measure')
Write-Host "完了  結果: results\P22\argeval.md・results\P23\argeval.md" -ForegroundColor Green
