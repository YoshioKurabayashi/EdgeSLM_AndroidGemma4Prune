# P34: P33（何を返すかで絞る・引数の直し）を入れた構成の合否を、新しい評価セット（challenge_v10・600 行）で測る。数え方は P33 の改定後（機能にない依頼を雑談で受けたら正解）。
# 設計: DoC/設計書_P34_新しい評価セットv10で合否.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md ｜ 測る前の記録: results\P34\preregistration.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う。判定デモアプリは使わない）。VLM の作業と同時に動かさない。
# ヘッドは学習し直さない（P28 で作った 1,656 行のヘッドをそのまま使う）。BERT は P20 で保存したモデルで、v10 の確率を出すだけ。
#
#   .\scripts\run_p34.ps1                 # データの確かめ → トークン化 → 端末（600 行 × 2 つのグラフ）→ BERT の確率 → 判定（15 分ほど）
#   .\scripts\run_p34.ps1 -JudgeOnly      # 判定と集計だけやり直す（3〜5 分）
# 途中で止めても、端末の済んだ分（work\p34\parts）は残る。もう一度実行すれば続きから。
# 進み具合は work\logs\p34-<日時>.log
param(
    [string]$Layers = "17,26",
    [string]$Seeds = "0,1,2",
    [switch]$JudgeOnly
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# トークン化・端末は EdgeSLM_Android の venv、BERT・判定は EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p34-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P34 =====（ログ: $log）" -ForegroundColor Cyan
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
    Step $py "評価セット v10 を作る（重なりを確かめる。測る前の記録の md5 7ee3c3ac… と同じになるはず）" @('-B', '-m', 'app.p34_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p34_verdict', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で h を集める（評価セット v10 600 行）" @('-B', '-m', 'app.p34_verdict', 'device', '--layers', $Layers)
    Step $pyGpu "保存した BERT（案 F × seed）で、評価セット v10 の確率を出す（v9 の行で P32 と同じかも確かめる）" @('-B', '-m', 'app.p34_verdict', 'text', '--seeds', $Seeds)
}
Step $pyGpu "評価セット v10 で合否（本命 P33 ↔ P31 の構成 ↔ いまのアプリ）" @('-B', '-m', 'app.p34_verdict', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P34\verdict.md" -ForegroundColor Green
