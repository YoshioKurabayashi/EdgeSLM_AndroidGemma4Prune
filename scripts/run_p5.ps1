# P5: 学習データを足して（train_add_v1・219 行）ヘッドを学習し直し、評価セット（challenge_v1）で前と後を比べる。
# 設計: DoC/設計書_P5_学習データを足してヘッドを学習し直す.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う）。評価セットの h は P4 のものを使う。
#
#   .\scripts\run_p5.ps1                  # データ → トークン化 → 端末（足した行）→ ヘッド（GPU の venv）→ 判定（20 分ほど）
#   .\scripts\run_p5.ps1 -JudgeOnly       # 判定と集計だけやり直す
#   .\scripts\run_p5.ps1 -Force           # ヘッドを学習し直す（端末の結果はそのまま）
# 進み具合は work\logs\p5-<日時>.log
param(
    [string]$Layers = "17,26,35",
    [switch]$JudgeOnly,
    [switch]$Force
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# トークン化・端末・判定は EdgeSLM_Android の venv、ヘッドは EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p5-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P5 =====（ログ: $log）" -ForegroundColor Cyan
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
    Step $py "学習に足す発話を作る（714 行・評価セットとの重なりを確かめる）" @('-B', '-m', 'app.p5_train_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p5_retrain', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で足した行の h を集める" @('-B', '-m', 'app.p5_retrain', 'device', '--layers', $Layers)
    $a = @('-B', '-m', 'app.p5_retrain', 'heads', '--layers', $Layers)
    if ($Force) { $a += '--force' }
    Step $pyGpu "714 行＋足した行でヘッドを学習・評価・交差検証 → 書き出し" $a
}
Step $py "評価セットで前と後を比べる" @('-B', '-m', 'app.p5_retrain', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P5\challenge_before_after.md" -ForegroundColor Green
