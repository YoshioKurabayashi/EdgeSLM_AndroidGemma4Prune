# P21: 規則と引数の誤りを直した構成の合否を、新しい評価セット（challenge_v6・600 行）で測る。
# 設計: DoC/設計書_P21_規則と引数の誤りを直す.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md
# 端末を USB で接続（お試しアプリ「Gemma4 層削減お試し」とモデルは P2・P3 で入れたものを使う）。VLM の作業と同時に動かさない。
# ヘッドも BERT も学習し直さない。BERT は P20 で保存したモデル（work\p20\bert_F）で、評価セット v6 の確率を出すだけ。
#
#   .\scripts\run_p21.ps1                 # データ → トークン化 → 端末（600 行 × 3 つのグラフ）→ BERT の確率 → 判定（8 分ほど）
#   .\scripts\run_p21.ps1 -JudgeOnly      # 判定と集計だけやり直す（1 分）
#   .\scripts\run_p21.ps1 -FixOnly        # 直しの確かめ（これまでの行で、直す前と後を比べる。results\P21\fix.md）だけやり直す（3 分）
# 自分で書いた評価用の発話を足すとき: data\challenge_v6\user_rows.tsv に「発話<TAB>正解」を 1 行ずつ（UTF-8）。
#   正解は 機能なし・不明・雑談・hvac・media・nav・status・manual・poi・minfo・cam のどれか（書き方の例: data\challenge_v6\user_rows.sample.tsv）。
#   置いてから実行する（後から足したときも、そのままもう一度実行すればよい。端末は足した行だけ流れる）
# 途中で止めても、端末の済んだ分（work\p21\parts）は残る。
# 進み具合は work\logs\p21-<日時>.log
param(
    [string]$Layers = "17,26,35",
    [string]$Seeds = "0,1,2",
    [switch]$JudgeOnly,
    [switch]$FixOnly
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
$log = Join-Path $logDir ("p21-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "===== P21 =====（ログ: $log）" -ForegroundColor Cyan
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
if ($FixOnly) {
    Step $pyGpu "直しの確かめ（これまでの行で、直す前と後を比べる）" @('-B', '-m', 'app.p21_fix', '--layers', $Layers)
    Write-Host "完了  結果: results\P21\fix.md" -ForegroundColor Green
    return
}
if (-not $JudgeOnly) {
    Step $py "評価セット v6 を作る（重なりを確かめる）" @('-B', '-m', 'app.p21_data')
    Step $py "トークン化（714 行と同じ符号化かを確かめる）" @('-B', '-m', 'app.p21_verdict', 'encode')
    & 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null
    Step $py "端末で h を集める（評価セット v6 600 行）" @('-B', '-m', 'app.p21_verdict', 'device', '--layers', $Layers)
    Step $pyGpu "保存した BERT（案 F × seed）で、評価セット v6 の確率を出す" @('-B', '-m', 'app.p21_verdict', 'text', '--seeds', $Seeds)
}
Step $pyGpu "評価セット v6 で、直した構成の合否を出す" @('-B', '-m', 'app.p21_verdict', 'judge', '--layers', $Layers)
Write-Host "完了  結果: results\P21\verdict.md" -ForegroundColor Green
