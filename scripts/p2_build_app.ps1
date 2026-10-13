# P2: お試しアプリ「Gemma4 層削減お試し」（org.edgeslm.gemma4prune、android_app_prune/）をビルドして端末に入れる。
# 設計: DoC/設計書_P2_層を減らしたグラフを端末で.md の 4 節。
# EdgeSLM_AndroidGemma4Head の scripts/b3_build_app.ps1 の複製（フォルダとアプリ名だけ替えた）。既存のアプリとは別に入る（上書きしない）。
#
#   .\scripts\p2_build_app.ps1
param(
    [string]$Adb = 'D:\Android\sdk\platform-tools\adb.exe',
    [string]$JavaHome = 'D:\Android\jdk17',
    [switch]$Online            # --offline を外す（キャッシュに無い依存を取りに行く。必要なときだけ）
)
# adb や gradle は stderr に普通の出力を書くので、Stop にすると PowerShell 5.1 ではそれが例外になる。終了コードで判定する
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:JAVA_HOME = $JavaHome

Write-Host "=== ビルド（元のアプリと同じ版の Gradle・AGP）===" -ForegroundColor Cyan
Push-Location android_app_prune
$gargs = @('assembleDebug')
if (-not $Online) { $gargs = @('--offline') + $gargs }
& .\gradlew.bat @gargs
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "ビルドに失敗しました（exit=$code）" }
$apk = 'android_app_prune\app\build\outputs\apk\debug\app-debug.apk'
Get-Item $apk | Format-Table Name, Length, LastWriteTime

Write-Host "=== インストール（org.edgeslm.gemma4prune）===" -ForegroundColor Cyan
& $Adb start-server 2>&1 | Out-Null
& $Adb install -r $apk
if ($LASTEXITCODE -ne 0) { throw "インストールに失敗しました（端末が USB でつながっているか確認）" }
# 通知の権限（無くても動くが、Foreground Service の通知を出すため。元のアプリと同じ）
& $Adb shell pm grant org.edgeslm.gemma4prune android.permission.POST_NOTIFICATIONS 2>&1 | Out-Null
& $Adb shell pm list packages org.edgeslm.gemma4 2>&1 | Out-Host
Write-Host "完了。ランチャーに「Gemma4 層削減お試し」が増えます（既存のアプリはそのまま）。この版では分類の画面は使わない（Runner だけ使う）。" -ForegroundColor Green
Write-Host "次は: .\scripts\run_p2.ps1 -Stage device" -ForegroundColor Green
