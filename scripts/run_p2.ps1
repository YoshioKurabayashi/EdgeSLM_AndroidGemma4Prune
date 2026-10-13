# P2: 層を減らした分類専用のまとめ読みグラフ（層 17・層 26）を作り、端末で 35 層と比べる（お試し）。
# 設計: DoC/設計書_P2_層を減らしたグラフを端末で.md
#
#   .\scripts\run_p2.ps1 -Stage build      # PC（30 分ほど）: 最後のかたまり（N17・N26）→ 繋ぐ → PC の ORT で確かめる
#   .\scripts\run_p2.ps1 -Stage convert    # PC（数時間）: QAIRT → EPContext（層 17・26）
#   ── 端末（先に .\scripts\p2_build_app.ps1。USB で接続。ほかのセッションが端末を使っていないとき）──
#   .\scripts\run_p2.ps1 -Stage device     # 送る → 16 行で速さ → 714 行の h を集める（3 つのグラフ。20 分ほど）
#   .\scripts\run_p2.ps1 -Stage heads      # 端末の h でヘッド（GPU の venv・10 分ほど）→ PC の HF と比べる
#   -Layers "17"  で層の数をしぼれる（build・convert は 17,26、device・heads は 17,26,35 が既定）
# 進み具合は work\logs\p2-<段>-<日時>.log（各手順の詳しいログは work\logs\p2_*.log）
param(
    [ValidateSet("build", "convert", "device", "heads")][string]$Stage = "build",
    [string]$Layers = "",
    [switch]$Force
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# 変換・端末は EdgeSLM_Android の venv、ヘッドは EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "EdgeSLM_Android の venv がありません: $py" }
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p2-{0}-{1}.log" -f $Stage, (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P2 $Stage =====（ログ: $log）" -ForegroundColor Cyan
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
$lb = if ($Layers) { $Layers } else { "17,26" }
$ld = if ($Layers) { $Layers } else { "17,26,35" }
if ($Stage -eq "build") {
    $a = @('-B', '-m', 'app.p2_build', '--steps', 'chunks,merge,ortcheck', '--layers', $lb)
    if ($Force) { $a += '--force' }
    Step $py "最後のかたまり → 繋ぐ → PC の ORT" $a
}
if ($Stage -eq "convert") {
    Step $py "QAIRT で context binary" @('-B', '-m', 'app.p2_build', '--steps', 'convert', '--layers', $lb)
}
if ($Stage -in @("device")) {
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "お試しアプリへ送る" @('-B', '-m', 'app.p2_device', 'deploy', '--layers', $ld)
    Step $py "16 行で速さを比べる" @('-B', '-m', 'app.p2_device', 'speed', '--layers', $ld)
    Step $py "714 行の h を集める" @('-B', '-m', 'app.p2_device', 'collect', '--layers', $ld)
}
if ($Stage -eq "heads") {
    $a = @('-B', '-m', 'app.p2_heads', '--layers', $ld)
    if ($Force) { $a += '--force' }
    Step $pyGpu "端末の h でヘッド → PC の HF と比べる" $a
}
Write-Host "完了" -ForegroundColor Green
