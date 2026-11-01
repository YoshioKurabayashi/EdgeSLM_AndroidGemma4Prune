# P37: P35 の構成（アプリ v2 と同じ）の合否を、新しい評価セット（challenge_v11・600 行）で測る。あわせて、v11 をアプリ v2 の中だけで判定させ、PC と突き合わせる。
# 設計: DoC/設計書_P37_新しい評価セットv11で合否.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md ｜ 測る前の記録: results\P37\preregistration.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」と、P36 で入れた「Gemma4 判定デモ v2」を使う）。VLM の作業と同時に動かさない。学習はしない。
#
#   .\scripts\run_p37.ps1                 # データの確かめ → トークン化 → 端末で h → BERT の確率 → 判定（合否）→ アプリ v2 で v11 を判定して突き合わせ（20 分ほど）
#   .\scripts\run_p37.ps1 -JudgeOnly      # 合否の判定と集計だけやり直す（3〜5 分）
#   .\scripts\run_p37.ps1 -AppOnly        # アプリ v2 での突き合わせだけやり直す（5 分ほど）
# 途中で止めても、端末の済んだ分（work\p37\parts）は残る。もう一度実行すれば続きから。
# 進み具合は work\logs\p37-<日時>.log
param(
    [string]$Layers = "17,26",
    [string]$Seeds = "0,1,2",
    [switch]$JudgeOnly,
    [switch]$AppOnly
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
$log = Join-Path $logDir ("p37-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P37 =====（ログ: $log）" -ForegroundColor Cyan
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
if (-not $JudgeOnly -and -not $AppOnly) {
    Step $py "評価セット v11 を作る（重なりを確かめる。測る前の記録の md5 354de4fd… と同じになるはず）" @('-B', '-m', 'app.p37_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p37_verdict', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "お試しアプリで h を集める（評価セット v11 600 行）" @('-B', '-m', 'app.p37_verdict', 'device', '--layers', $Layers)
    Step $pyGpu "保存した BERT（案 F × seed）で、評価セット v11 の確率を出す（v10 の行で P34 と同じかも確かめる）" @('-B', '-m', 'app.p37_verdict', 'text', '--seeds', $Seeds)
}
if (-not $AppOnly) {
    Step $pyGpu "評価セット v11 で合否（本命 P35 ↔ P33 の構成 ↔ アプリ v1）" @('-B', '-m', 'app.p37_verdict', 'judge', '--layers', $Layers)
}
if (-not $JudgeOnly) {
    Step $pyGpu "アプリ v2 用の確かめの行を作る（PC の判定・Gemma の ID 列）" @('-B', '-m', 'app.p37_app', 'prep')
    Step $pyGpu "PC の BertJapaneseTokenizer の ID 列（v11。端末の BERT の入力の正解）" @('-B', '-m', 'app.p37_app', 'truth')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "アプリ v2 へ確かめの行を送る" @('-B', '-m', 'app.p37_app', 'deploy')
    Step $py "アプリ v2 の中だけで v11 を判定させ、PC と突き合わせる" @('-B', '-m', 'app.p37_app', 'check', '--layers', $Layers)
}
Write-Host "完了  結果: results\P37\verdict.md（合否）・results\P37\app_check.md（アプリ v2）" -ForegroundColor Green
