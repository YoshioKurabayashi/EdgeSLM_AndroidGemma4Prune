# P24: 判定デモアプリ「Gemma4 判定デモ」（org.edgeslm.gemma4router）を作り、端末の判定を PC の判定と突き合わせる。
# 設計: DoC/設計書_P24_判定デモアプリ.md
# 発話 1 つについて、いまの構成（Gemma 4 の前半 17 層か 26 層＋ヘッド・BERT・規則・引数・検査）の最終の判定までを、スマホの中だけで出すアプリ。
# 既存のアプリとその置き場には書き込まない。学習はしない。VLM の作業と同時に動かさない。
#
#   .\scripts\run_p24.ps1                  # prep → build → deploy → check → open を順に（20 分ほど。端末を USB で接続）
#   .\scripts\run_p24.ps1 -Stage prep      # 資材を作り、Java の判定が PC と同じかを確かめる（端末なし・5 分ほど）
#   .\scripts\run_p24.ps1 -Stage build     # アプリをビルドして入れる
#   .\scripts\run_p24.ps1 -Stage deploy    # 資材（グラフ・ヘッド・BERT・辞書）を端末へ送る
#   .\scripts\run_p24.ps1 -Stage check     # 端末で評価セット v6 の 600 行を層 17・26 で判定させ、PC と突き合わせる（5 分ほど）
#   .\scripts\run_p24.ps1 -Stage open      # 画面を開く（手で試す）
#   .\scripts\run_p24.ps1 -Stage open -Prompts "エアコンを25度にして|26:|エアコンを25度にして"   # 順に判定（「26:」はその層へ切り替え）
#   -Force を付けると、prep で、すでにある資材も作り直す
# 進み具合は work\logs\p24-<日時>.log
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
# ONNX への書き出し・端末は EdgeSLM_Android の venv、PC の入力の正解（fugashi が要る）は EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$javac = Join-Path $JavaHome 'bin\javac.exe'
$java = Join-Path $JavaHome 'bin\java.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p24-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P24（$Stage）=====（ログ: $log）" -ForegroundColor Cyan
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
    if ($Force -or -not (Test-Path 'work\p24\fixtures\flow_L26.tsv')) {
        Step $pyGpu "PC の判定の表を作る（規則・引数・流れ。Java と突き合わせる正解）" @('-B', '-m', 'app.p24_fixtures')
    } else { Write-Host "PC の判定の表（work\p24\fixtures）は、あるので作り直さない（-Force で作り直す）" -ForegroundColor DarkGray }
    if ($Force -or -not (Test-Path 'work\p24\device\router\router.json')) {
        Step $pyGpu "ヘッド・設定・BERT の辞書を、端末の形に書き出す" @('-B', '-m', 'app.p24_assets', 'heads')
    } else { Write-Host "ヘッド・設定・辞書（work\p24\device\router）は、あるので作り直さない（-Force で作り直す）" -ForegroundColor DarkGray }
    Step $pyGpu "PC の BertJapaneseTokenizer の ID 列を出す（端末用の入力作りの正解）" @('-B', '-m', 'app.p24_assets', 'truth')
    if ($Force -or -not (Test-Path 'work\p24\device\router\bert\model.onnx')) {
        Step $py "BERT（案 F・seed 0）を ONNX にして、torch・保存済みの確率と比べる" @('-B', '-m', 'app.p24_assets', 'bert')
    } else { Write-Host "BERT の ONNX（work\p24\device\router\bert\model.onnx）は、あるので作り直さない（-Force で作り直す）" -ForegroundColor DarkGray }
    # ---- アプリに入れる判定（Java）が、PC（Python）と同じ結果を返すか
    $out = 'work\p24\java_out'
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $src = @((Get-ChildItem 'android_app_router\app\src\main\java\org\edgeslm\router\*.java').FullName) + @((Get-ChildItem 'tools\p24_java\org\edgeslm\router\*.java').FullName)
    Step $javac "アプリ用の判定（Java）をコンパイル" (@('--release', '8', '-encoding', 'UTF-8', '-Xlint:-options', '-d', $out) + $src)
    $jv = @('-Dfile.encoding=UTF-8', '-Dsun.stdout.encoding=UTF-8', '-Dstdout.encoding=UTF-8', '-cp', $out)
    Step $java "Java ↔ PC: 規則・引数・検査・判定の流れ" ($jv + @('org.edgeslm.router.ParityTest', 'work\p24\fixtures'))
    Step $java "Java ↔ PC: ヘッドの計算と、h からの判定" ($jv + @('org.edgeslm.router.HeadTest', 'work\p24\fixtures', 'work\p24\device\router'))
    # 入力作りは、PC の正解と初めて突き合わせる所。食い違いがあっても止めない（端末の確かめで、判定への影響を見る）
    Step $java "Java ↔ PC: BERT の入力作り（MeCab＋WordPiece）" ($jv + @('org.edgeslm.router.TokTest', 'work\p24\device\router\bert', 'work\p24\tok_truth.tsv')) -Soft
    Write-Host "prep 完了  結果: results\P24\bert_onnx.md" -ForegroundColor Green
}
if (Want 'build') {
    Write-Host "`n--- アプリをビルドして入れる ---" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'p24_build_app.ps1') -JavaHome $JavaHome
}
if ((Want 'deploy') -or (Want 'check') -or (Want 'open')) { & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null }
if (Want 'deploy') { Step $py "資材を端末へ送る（グラフ 2 つ・ヘッド・BERT・辞書）" @('-B', '-m', 'app.p24_device', 'deploy') }
if (Want 'check') {
    Step $py "端末で評価セット v6 を判定させ、PC の判定と突き合わせる" @('-B', '-m', 'app.p24_device', 'check', '--layers', $Layers)
    Write-Host "check 完了  結果: results\P24\device_check.md" -ForegroundColor Green
}
if (Want 'open') {
    $a = @('-B', '-m', 'app.p24_device', 'open')
    if ($Prompts) { $a += @('--prompts', $Prompts) }
    Step $py "判定デモの画面を開く" $a
}
foreach ($w in $warn) { Write-Host $w -ForegroundColor Yellow }
Write-Host "完了" -ForegroundColor Green
