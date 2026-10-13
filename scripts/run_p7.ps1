# P7: 機能にない依頼の止め方を 2 通り試す（ヘッドの学習データを増やす／文だけを読む外の部品 BERT・n-gram）。新しい評価セット challenge_v2 で比べる。
# 設計: DoC/設計書_P7_機能にない依頼の止め方.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う）。
#
#   .\scripts\run_p7.ps1                  # データ → トークン化 → 端末 → ヘッド（GPU）→ 文の分類器（GPU）→ 判定（35 分ほど）
#   .\scripts\run_p7.ps1 -JudgeOnly       # 判定と集計だけやり直す
#   .\scripts\run_p7.ps1 -Force           # ヘッドを学習し直す（端末の結果はそのまま）
#   .\scripts\run_p7.ps1 -Online          # 日本語 BERT が PC に無いとき、取りに行くのを許す（既定は通信しない）
# 自分で書いた評価用の発話を足すとき: data\challenge_v2\user_rows.tsv に「発話<TAB>正解」を 1 行ずつ（UTF-8）。
#   正解は 機能なし・不明・雑談・hvac・media・nav・status・manual・poi・minfo・cam のどれか。置いてから実行する
# 進み具合は work\logs\p7-<日時>.log
param(
    [string]$Layers = "17,26,35",
    [switch]$JudgeOnly,
    [switch]$Force,
    [switch]$Online
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
# トークン化・端末・判定は EdgeSLM_Android の venv、ヘッドと文の分類器は EdgeSLM_MultiHead_Multi の venv（どちらも使うだけ）
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'work\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("p7-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P7 =====（ログ: $log）" -ForegroundColor Cyan
function Step([string]$Py, [string]$Title, [string[]]$A, [switch]$Optional) {
    Write-Host "`n--- $Title ---" -ForegroundColor Cyan
    Add-Content -Path $log -Value "--- $Title ---" -Encoding utf8
    $t0 = Get-Date
    & $Py @A 2>&1 | ForEach-Object { $l = "$_"; Add-Content -Path $log -Value $l -Encoding utf8; $l } | Out-Host
    $code = $LASTEXITCODE
    $s = "[{0}] exit={1} 所要={2} 分" -f $Title, $code, [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Add-Content -Path $log -Value $s -Encoding utf8
    Write-Host $s -ForegroundColor DarkGray
    if ($code -ne 0) {
        if ($Optional) { Write-Host "★ $Title が失敗しました。続けます（判定はヘッドだけの表になります）" -ForegroundColor Yellow }
        else { throw "$Title が失敗しました（ログ: $log）" }
    }
}
if (-not $JudgeOnly) {
    Step $py "評価セット v2 と、学習に足す行 v2 を作る（重なりを確かめる）" @('-B', '-m', 'app.p7_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p7_block', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で h を集める（足す行 168・評価セット v2 200）" @('-B', '-m', 'app.p7_block', 'device', '--layers', $Layers)
    $a = @('-B', '-m', 'app.p7_block', 'heads', '--layers', $Layers)
    if ($Force) { $a += '--force' }
    Step $pyGpu "933 行＋足す行でヘッドを学習・評価・交差検証" $a
    if ($Online) { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue; Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue }
    Step $pyGpu "文だけの分類器（n-gram・日本語 BERT）を学習" @('-B', '-m', 'app.p7_block', 'text') -Optional
    $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
}
Step $py "評価セット v2 で比べる" @('-B', '-m', 'app.p7_block', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P7\block_compare.md" -ForegroundColor Green
