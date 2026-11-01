# P38: 判定デモアプリ v2（Gemma4 判定デモ v2）を前半 17 層で動かし、判定の様子を録画する。
# 設計: DoC/設計書_P38_層17の判定デモ動画.md
# 端末を USB で接続。アプリ v2 と資材は P36・P37 で送り済みのものを使う（設定が P37 のものと違うときだけ送り直す）。VLM の作業と同時に動かさない。
# 録画中は端末に触らない（画面に通知が出ると映り込む）。
#
#   .\scripts\run_p38.ps1                     # 既定の 15 文を層 17 で録画（2〜3 分）
#   .\scripts\run_p38.ps1 -OpenOnly           # 録画せず、画面を開いて 15 文を流すだけ（下見）
#   .\scripts\run_p38.ps1 -Layer 26 -Name demo_L26
#   .\scripts\run_p38.ps1 -Prompts "車内を21度に設定してください|次の曲にしてください" -Name short
param(
    [int]$Layer = 17,
    [string]$Prompts = "",
    [string]$Name = "demo",
    [switch]$OpenOnly
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'   # 使うだけ（入れない）
& 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null

$a = @('-B', '-m', 'app.p38_record', '--layer', "$Layer", '--name', $Name)
if ($Prompts) { $a += @('--prompts', $Prompts) }
if ($OpenOnly) { $a += '--open-only' }
Write-Host "=== アプリ v2 を層 $Layer で開いて録画 ===" -ForegroundColor Cyan
& $py @a
if ($LASTEXITCODE -ne 0) { throw "最後まで終わらなかったか、文の数が合いません（results\P38_20261015_L17_demo のログを確認）" }
Write-Host "完了  結果: results\P38_20261015_L17_demo\README.md" -ForegroundColor Green
