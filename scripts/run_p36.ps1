# P36: 判定デモアプリ v2「Gemma4 判定デモ v2」（org.edgeslm.gemma4router2、android_app_router2/）を作り、端末の判定を PC の判定と突き合わせる。
# 設計: DoC/設計書_P36_判定デモアプリv2.md
# いまの構成（P35: 規則 P31・引数 P35（歯止め・機能の定義で絞る・範囲外は断る）・ヘッド 1,656 行・S1 の seed）を、スマホの中だけで動かすアプリ。
# P24 の判定デモ（android_app_router/・org.edgeslm.gemma4router）はそのまま残す（別のアプリとして入る）。既存のアプリとその置き場には書き込まない。学習はしない。VLM の作業と同時に動かさない。
#
#   .\scripts\run_p36.ps1                  # prep → build → deploy → check → open を順に（20〜30 分。端末を USB で接続）
#   .\scripts\run_p36.ps1 -Stage prep      # 資材を作り、Java の判定が PC と同じかを確かめる（端末なし・5 分ほど）
#   .\scripts\run_p36.ps1 -Stage build     # アプリをビルドして入れる
#   .\scripts\run_p36.ps1 -Stage deploy    # 資材（グラフ・ヘッド・BERT・辞書）を端末へ送る
#   .\scripts\run_p36.ps1 -Stage check     # 端末で評価セット v10 の 600 行を層 17・26 で判定させ、PC と突き合わせる（5 分ほど）
#   .\scripts\run_p36.ps1 -Stage open      # 画面を開く（手で試す）
#   .\scripts\run_p36.ps1 -Stage open -Prompts "近くのホテルの宿泊料金を調べて|26:|曲をシャッフルしてください"
#   -Force を付けると、prep で、すでにある資材も作り直す
# 進み具合は work\logs\p36-<日時>.log
param(
    [ValidateSet('all', 'prep', 'build', 'deploy', 'check', 'open')][string]$Stage = 'all',
    [string]$Layers = "17,26",
    [string]$Prompts = "",
    [switch]$Force,
    [string]$JavaHome = 'D:\Android\jdk17'
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# 端末は EdgeSLM_Android の venv、PC の判定の表・入力の正解（fugashi が要る）は EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$javac = Join-Path $JavaHome 'bin\javac.exe'
$java = Join-Path $JavaHome 'bin\java.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p36-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P36（$Stage）=====（ログ: $log）" -ForegroundColor Cyan
function Step([string]$Exe, [string]$Title, [string[]]$A, [switch]$Soft) {
    Write-Host "`n--- $Title ---" -ForegroundColor Cyan
    Add-Content -Path $log -Value "--- $Title ---" -Encoding utf8
    $t0 = Get-Date
    & $Exe @A 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
    $code = $LASTEXITCODE
    $s = "[{0}] exit={1} 所要={2} 分" -f $Title, $code, [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Add-Content -Path $log -Value $s -Encoding utf8
    Write-Host $s -ForegroundColor DarkGray
    if ($code -ne 0) {
        if (-not $Soft) { throw "$Title が失敗しました（ログ: $log）" }
        $script:warn += "★$Title で食い違いがありました（ログ: $log）。止めずに先へ進みます"
        Write-Host $script:warn[-1] -ForegroundColor Yellow
    }
}
function Want($name) { return ($Stage -eq 'all' -or $Stage -eq $name) }
$warn = @()

if (Want 'prep') {
    if (-not (Test-Path 'work\p24\device\router\bert\model.onnx')) { throw "P24 の BERT（work\p24\device\router\bert\model.onnx）が無い。先に .\scripts\run_p24.ps1 -Stage prep" }
    if ($Force -or -not (Test-Path 'work\p36\fixtures\flow_L26.tsv')) {
        Step $pyGpu "PC の判定の表を作る（規則 P31・引数 P35・流れ。Java と突き合わせる正解）" @('-B', '-m', 'app.p36_fixtures')
    } else { Write-Host "PC の判定の表（work\p36\fixtures）は、あるので作り直さない（-Force で作り直す）" -ForegroundColor DarkGray }
    if ($Force -or -not (Test-Path 'work\p36\device\router\router.json')) {
        Step $pyGpu "ヘッド（1,656 行・S1 の seed）・設定・確かめ用の行を、端末の形に書き出す" @('-B', '-m', 'app.p36_assets', 'heads')
    } else { Write-Host "ヘッド・設定（work\p36\device\router）は、あるので作り直さない（-Force で作り直す）" -ForegroundColor DarkGray }
    Step $pyGpu "PC の BertJapaneseTokenizer の ID 列を出す（評価セット v10。端末の BERT の入力の正解）" @('-B', '-m', 'app.p36_assets', 'truth')
    # ---- アプリに入れる判定（Java）が、PC（Python）と同じ結果を返すか
    $out = 'work\p36\java_out'
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $src = @((Get-ChildItem 'android_app_router2\app\src\main\java\org\edgeslm\router\*.java').FullName) + @((Get-ChildItem 'tools\p36_java\org\edgeslm\router\*.java').FullName)
    Step $javac "アプリ用の判定（Java）をコンパイル" (@('--release', '8', '-encoding', 'UTF-8', '-Xlint:-options', '-d', $out) + $src)
    $jv = @('-Dfile.encoding=UTF-8', '-Dsun.stdout.encoding=UTF-8', '-Dstdout.encoding=UTF-8', '-cp', $out)
    Step $java "Java ↔ PC: 規則・引数・検査・判定の流れ・単語の分け方" ($jv + @('org.edgeslm.router.ParityTest2', 'work\p36\fixtures', '12', 'work\p24\device\router\bert'))
    Step $java "Java ↔ PC: ヘッド（1,656 行）の計算と、h からの判定" ($jv + @('org.edgeslm.router.HeadTest2', 'work\p36\fixtures', 'work\p36\device\router'))
    Write-Host "prep 完了  結果: results\P36\decide_check.md" -ForegroundColor Green
}
if (Want 'build') {
    Write-Host "`n--- アプリをビルドして入れる ---" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'p36_build_app.ps1') -JavaHome $JavaHome
}
if ((Want 'deploy') -or (Want 'check') -or (Want 'open')) { & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null }
if (Want 'deploy') { Step $py "資材を端末へ送る（グラフ 2 つ・ヘッド・BERT・辞書）" @('-B', '-m', 'app.p36_device', 'deploy') }
if (Want 'check') {
    Step $py "端末で評価セット v10 を判定させ、PC の判定と突き合わせる" @('-B', '-m', 'app.p36_device', 'check', '--layers', $Layers)
    Write-Host "check 完了  結果: results\P36\device_check.md" -ForegroundColor Green
}
if (Want 'open') {
    $a = @('-B', '-m', 'app.p36_device', 'open')
    if ($Prompts) { $a += @('--prompts', $Prompts) }
    Step $py "判定デモ v2 の画面を開く" $a
}
foreach ($w in $warn) { Write-Host $w -ForegroundColor Yellow }
Write-Host "完了" -ForegroundColor Green
