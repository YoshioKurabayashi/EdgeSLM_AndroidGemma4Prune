# P39: 判定 1 回の時間を、部品ごとに分けて測る（BERT・Gemma の層（NPU）・埋め込み/PLE の表引き・RoPE・毎回の準備・受け渡し）。
# 設計: DoC/設計書_P39_時間の内訳を測る.md
# 使うのは測定用の複製アプリ「Gemma4 判定デモ v3 測定」（org.edgeslm.gemma4router3・android_app_router3/）。判定デモ v2 とその置き場は変えない。
# 端末を USB で接続。VLM の作業と同時に動かさない。学習はしない。測定中は端末に触らない（画面は点いたまま）。
#
#   .\scripts\run_p39.ps1                   # ビルド → 設定 → 送る → 測る → まとめ（ビルド 5 分・送る 5〜10 分・測る 15〜25 分）
#   .\scripts\run_p39.ps1 -Stage build      # アプリをビルドして入れる
#   .\scripts\run_p39.ps1 -Stage deploy     # 設定を作って資材を送る（v2・お試しアプリの置き場にある同じものは端末の中で写す）
#   .\scripts\run_p39.ps1 -Stage bench      # 測る（層 17・26・35 × par・serial、各 300 行。最後に QNN の計測を 30 行ずつ）
#   .\scripts\run_p39.ps1 -Stage report     # まとめだけやり直す
#   .\scripts\run_p39.ps1 -Stage bench -Layers "17,26" -Rows 100 -QnnProf off   # 短く試す
# 進み具合は work\logs\p39-<日時>.log
param(
    [ValidateSet('all', 'build', 'deploy', 'bench', 'report')][string]$Stage = 'all',
    [string]$Layers = "17,26,35",
    [int]$Rows = 300,
    [string]$Modes = "par,serial",
    [ValidateSet('off', 'basic', 'detailed')][string]$QnnProf = 'detailed',
    [string]$JavaHome = 'D:\Android\jdk17'
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'   # 使うだけ（入れない）
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p39-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P39（$Stage）=====（ログ: $log）" -ForegroundColor Cyan
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
function Want($name) { return ($Stage -eq 'all' -or $Stage -eq $name) }
& 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null

if (Want 'build') {
    & (Join-Path $PSScriptRoot 'p39_build_app.ps1') -JavaHome $JavaHome
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) { throw "ビルドかインストールに失敗しました" }
}
if (Want 'deploy') {
    Step "v3 の設定を作る（P37 の設定に 35 層を足す。既定の層 17・確かめ用の行 v11）" @('-B', '-m', 'app.p39_latency', 'prep')
    Step "資材を v3 の置き場へ送る（グラフ 3 つ・表・BERT・ヘッド。全部で約 4.3GB を端末の中で写す）" @('-B', '-m', 'app.p39_latency', 'deploy')
}
if (Want 'bench') {
    Step "端末で測る（層 $Layers × $Modes・各 $Rows 行・QNN の計測 $QnnProf）" @('-B', '-m', 'app.p39_latency', 'bench', '--layers', $Layers, '--rows', "$Rows", '--modes', $Modes, '--qnnprof', $QnnProf)
}
if (Want 'report') {
    Step "まとめ（部品ごとの時間・QNN の計測）" @('-B', '-m', 'app.p39_latency', 'report')
}
Write-Host "完了  結果: results\P39\latency.md" -ForegroundColor Green
