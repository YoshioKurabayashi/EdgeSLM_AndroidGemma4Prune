# P4: データの外の発話（challenge_v1・360 行）を、端末の前半 17 層と 35 層のグラフで分類して比べる。
# 設計: DoC/設計書_P4_データの外の発話での評価.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う）。
#
#   .\scripts\run_p4.ps1                  # データ → ヘッドの書き出し（層 17・35）→ トークン化 → 端末 → 判定（10 分ほど）
#   .\scripts\run_p4.ps1 -JudgeOnly       # 端末の結果はそのままで、判定と集計だけやり直す
# 進み具合は work\logs\p4-<日時>.log
param(
    [string]$Layers = "17,35",
    [switch]$JudgeOnly
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p4-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P4 =====（ログ: $log）" -ForegroundColor Cyan
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
    Step $py "評価セットを作る（714 行との重なりも確かめる）" @('-B', '-m', 'app.p4_challenge_data')
    Step $pyGpu "ヘッドを書き出す（714 行で学習済みのもの。学習し直さない）" @('-B', '-m', 'app.p3_assets', '--layers', $Layers)
    Step $py "トークン化" @('-B', '-m', 'app.p4_eval', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で流す" @('-B', '-m', 'app.p4_eval', 'device', '--layers', $Layers)
}
Step $py "判定と集計" @('-B', '-m', 'app.p4_eval', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P4\challenge_table.md" -ForegroundColor Green
