# P3: 層 17 のグラフをお試しアプリの分類画面に載せ、デモを録画する。
# 設計: DoC/設計書_P3_層17の分類デモ.md
# 先に .\scripts\p2_build_app.ps1 でアプリを入れ直す（画面のソースを P3 用に変えたため）。端末を USB で接続。
#
#   .\scripts\run_p3.ps1                 # 資材を作る（GPU の venv）→ 送る → 既定の 6 文で録画
#   .\scripts\run_p3.ps1 -OpenOnly       # 資材を作って送り、画面を開くだけ（手で試す）
#   .\scripts\run_p3.ps1 -Layers 26 -Name demo_L26
#   .\scripts\run_p3.ps1 -Prompts "エアコンを22度にして|日本の首都はどこ" -Name short
param(
    [int]$Layers = 17,
    [string]$Prompts = "",
    [string]$Name = "demo",
    [switch]$OpenOnly
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
$py = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_Android\.venv\Scripts\python.exe'
$pyGpu = Join-Path (Split-Path -Parent $repo) 'EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe'
& 'D:\Android\sdk\platform-tools\adb.exe' start-server 2>&1 | Out-Null

Write-Host "=== 資材を作る（層 $Layers 個の端末の h で学習したヘッド）===" -ForegroundColor Cyan
& $pyGpu -B -m app.p3_assets --layers $Layers
if ($LASTEXITCODE -ne 0) { throw "資材の作成に失敗しました" }

$a = @('-B', '-m', 'app.p3_demo', '--layers', "$Layers", '--name', $Name)
if ($Prompts) { $a += @('--prompts', $Prompts) }
if ($OpenOnly) { $a += '--open-only' }
Write-Host "=== 資材を送る → 分類の画面 ===" -ForegroundColor Cyan
& $py @a
if ($LASTEXITCODE -ne 0) { throw "最後まで終わりませんでした（results\P3_20261002_demo のログを確認）" }
Write-Host "完了" -ForegroundColor Green
