# P39: 判定デモ v3・測定「Gemma4 判定デモ v3 測定」（org.edgeslm.gemma4router3、android_app_router3/）をビルドして端末に入れる。
# 設計: DoC/設計書_P39_時間の内訳を測る.md
# scripts/p36_build_app.ps1 の複製（フォルダとアプリ名だけ替えた）。既存のアプリ（判定デモ v2 を含む）とは別に入る（上書きしない）。
#
#   .\scripts\p39_build_app.ps1
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

Write-Host "=== ビルド（お試しアプリと同じ版の Gradle・AGP。新しい依存は無い）===" -ForegroundColor Cyan
Push-Location android_app_router3
$gargs = @('assembleDebug')
if (-not $Online) { $gargs = @('--offline') + $gargs }
& .\gradlew.bat @gargs
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "ビルドに失敗しました（exit=$code）" }
$apk = 'android_app_router3\app\build\outputs\apk\debug\app-debug.apk'
Get-Item $apk | Format-Table Name, Length, LastWriteTime

Write-Host "=== インストール（org.edgeslm.gemma4router3）===" -ForegroundColor Cyan
& $Adb start-server 2>&1 | Out-Null
& $Adb install -r $apk
if ($LASTEXITCODE -ne 0) { throw "インストールに失敗しました（端末が USB でつながっているか確認）" }
# 通知の権限（無くても動く。元のアプリと同じ）
& $Adb shell pm grant org.edgeslm.gemma4router3 android.permission.POST_NOTIFICATIONS 2>&1 | Out-Null
& $Adb shell pm list packages org.edgeslm.gemma4 2>&1 | Out-Host
Write-Host "完了。ランチャーに「Gemma4 判定デモ v3 測定」が増えます（判定デモ v2・既存のアプリはそのまま）。" -ForegroundColor Green
Write-Host "次は: .\scripts\run_p39.ps1 -Stage deploy" -ForegroundColor Green
