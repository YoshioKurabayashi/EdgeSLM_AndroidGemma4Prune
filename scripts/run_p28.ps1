# P28: 空調 ↔ メディアの区別を上げる行（72 行）を足してヘッドを学習し直し、新しい評価セット（challenge_v8・600 行）で流れ全体の合否を測る。
# 設計: DoC/設計書_P28_空調とメディアの区別を学習で上げる.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md ｜ 測る前の記録: results\P28\preregistration.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う。判定デモアプリは使わない）。VLM の作業と同時に動かさない。
# 学習し直すのはヘッドだけ（数分）。BERT は P20 で保存したモデルで、評価セット v8 の確率を出すだけ。
#
#   .\scripts\run_p28.ps1                 # データの確かめ → トークン化 → 端末（672 行 × 2 つのグラフ）→ ヘッドの学習 → BERT の確率 → 判定（15 分ほど）
#   .\scripts\run_p28.ps1 -JudgeOnly      # 判定と集計だけやり直す（3〜5 分）
#   .\scripts\run_p28.ps1 -Force          # ヘッドを作り直す（端末の済んだ分はそのまま使う）
# 途中で止めても、端末の済んだ分（work\p28\parts）は残る。もう一度実行すれば続きから。
# 進み具合は work\logs\p28-<日時>.log
param(
    [string]$Layers = "17,26",
    [string]$Seeds = "0,1,2",
    [switch]$JudgeOnly,
    [switch]$Force
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# トークン化・端末は EdgeSLM_Android の venv、ヘッド・BERT・判定は EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p28-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P28 =====（ログ: $log）" -ForegroundColor Cyan
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
    Step $py "足す行（72 行）と評価セット v8 を作る（重なりを確かめる。同じ中身なら md5 も同じ）" @('-B', '-m', 'app.p28_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p28_verdict', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で h を集める（足す行 72・評価セット v8 600）" @('-B', '-m', 'app.p28_verdict', 'device', '--layers', $Layers)
    $a = @('-B', '-m', 'app.p28_verdict', 'heads', '--layers', $Layers)
    if ($Force) { $a += '--force' }
    Step $pyGpu "1,584 行＋足す行（1,656 行）でヘッドを学習・評価" $a
    Step $pyGpu "保存した BERT（案 F × seed）で、評価セット v8 の確率を出す" @('-B', '-m', 'app.p28_verdict', 'text', '--seeds', $Seeds)
}
Step $pyGpu "評価セット v8（合否）と v7・v6（直す用）で、ヘッド 1,584 行 ↔ 1,656 行を比べる" @('-B', '-m', 'app.p28_verdict', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P28\verdict.md" -ForegroundColor Green
