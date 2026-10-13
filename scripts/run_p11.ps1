# P11: BERT のフラグが対象の依頼（空調など）に立つ問題。BERT の学習データを 4 通りに変えて学習し直し、フラグの立ち方と KPI を比べる。
# 設計: DoC/設計書_P11_BERTのフラグが対象の依頼に立つ問題.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# 端末は使わない。ヘッドも学習し直さない（BERT だけ。PC の GPU で 15 分ほど）。
#
#   .\scripts\run_p11.ps1                 # 足す文を作る → BERT を学習（4 案 × seed 3 つ。A の seed 0・1 は P7 の結果を写す）→ 判定
#   .\scripts\run_p11.ps1 -JudgeOnly      # 判定と集計だけやり直す（1〜2 分）
#   .\scripts\run_p11.ps1 -Online         # 日本語 BERT が PC に無いとき、取りに行くのを許す（既定は通信しない）
#   .\scripts\run_p11.ps1 -Seeds "0,1"    # seed を減らす（既定は 0,1,2）
# 途中で止めても、学習が済んだ seed は work\p11\text_scores_<案>.json に残る（もう一度実行すると続きから）。
# 進み具合は work\logs\p11-<日時>.log
param(
    [string]$Layers = "17,26,35",
    [string]$Seeds = "0,1,2",
    [switch]$JudgeOnly,
    [switch]$Online
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# どの手順も EdgeSLM_MultiHead_Multi の venv（torch・transformers・numpy・PyYAML）を使うだけ
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p11-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P11 =====（ログ: $log）" -ForegroundColor Cyan
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
if (-not $JudgeOnly) {
    Step "BERT の学習だけに足す、対象側の文を作る（重なりを確かめる）" @('-B', '-m', 'app.p11_data')
    if ($Online) { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue; Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue }
    Step "BERT を 4 案 × seed で学習（交差検証 5 回＋全体 1 回ずつ）" @('-B', '-m', 'app.p11_bert', 'text', '--seeds', $Seeds)
    $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
}
Step "フラグの立ち方と KPI を比べる" @('-B', '-m', 'app.p11_bert', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P11\bert_variants.md" -ForegroundColor Green
