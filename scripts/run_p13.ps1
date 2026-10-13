# P13: 大きい評価セット（challenge_v3・800 行）で、いまの構成と、重大な誤りを減らす 2 つの手当て（BERT に機能を足す・規則を補う）を測る。
# 設計: DoC/設計書_P13_評価セットを増やし重大な誤りを減らす.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う）。ヘッドは学習し直さない。
#
#   .\scripts\run_p13.ps1                 # データ → トークン化 → 端末（800 行 × 3 つのグラフ）→ BERT（GPU）→ 判定（20 分ほど）
#   .\scripts\run_p13.ps1 -JudgeOnly      # 判定と集計だけやり直す（1 分）
#   .\scripts\run_p13.ps1 -Online         # 日本語 BERT が PC に無いとき、取りに行くのを許す（既定は通信しない）
# 自分で書いた評価用の発話を足すとき: data\challenge_v3\user_rows.tsv に「発話<TAB>正解」を 1 行ずつ（UTF-8）。
#   正解は 機能なし・不明・雑談・hvac・media・nav・status・manual・poi・minfo・cam のどれか（書き方の例: data\challenge_v3\user_rows.sample.tsv）。
#   置いてから実行する。後から足したときは、もう一度そのまま実行すれば、足りない分だけ端末と BERT をやり直す
# 途中で止めても、端末の済んだ分（work\p13\parts）と BERT の済んだ seed は残る。
# 進み具合は work\logs\p13-<日時>.log
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
# トークン化・端末・判定は EdgeSLM_Android の venv、BERT は EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p13-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P13 =====（ログ: $log）" -ForegroundColor Cyan
function Step([string]$Py, [string]$Title, [string[]]$A) {
    Write-Host "`n--- $Title ---" -ForegroundColor Cyan
    Add-Content -Path $log -Value "--- $Title ---" -Encoding utf8
    $t0 = Get-Date
    & $Py @A 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
    $code = $LASTEXITCODE
    $s = "[{0}] exit={1} 所要={2} 分" -f $Title, $code, [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Add-Content -Path $log -Value $s -Encoding utf8
    Write-Host $s -ForegroundColor DarkGray
    if ($code -ne 0) { throw "$Title が失敗しました（ログ: $log）" }
}
if (-not $JudgeOnly) {
    Step $py "評価セット v3 と、BERT に足す行を作る（重なりを確かめる）" @('-B', '-m', 'app.p13_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p13_eval', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で h を集める（評価セット v3）" @('-B', '-m', 'app.p13_eval', 'device', '--layers', $Layers)
    if ($Online) { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue; Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue }
    Step $pyGpu "BERT を 2 案 × seed で学習（交差検証 5 回＋全体 1 回ずつ）" @('-B', '-m', 'app.p13_eval', 'text', '--seeds', $Seeds)
    $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
}
Step $pyGpu "評価セット v3 で 4 つの構成を比べる" @('-B', '-m', 'app.p13_eval', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P13\eval_v3.md" -ForegroundColor Green
