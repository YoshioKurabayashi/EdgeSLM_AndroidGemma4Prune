# -*- coding: utf-8 -*-
"""P3 手順 2: お試しアプリ「Gemma4 層削減お試し」（org.edgeslm.gemma4prune）に資材を送り、分類の画面を実機で動かして録画する。

設計: DoC/設計書_P3_層17の分類デモ.md の 4 節。EdgeSLM_AndroidGemma4Head の `app/b4_record_demo.py`（md5 838d962d）を複製し、
アプリ・ログのタグ・保存先・動画の名前を変え、最初に資材（`work/p3/L<n>/head/`）を送る段を足した（録画の手順は同じ。元のファイルは変更しない）:

0. 資材（head.json・head.bin・samples.json）をお試しアプリの置き場へ送る（md5 を確かめる。`--no-push` で飛ばす）

1. アプリを止め、画面を点けてロック画面を閉じる（端末の設定は変えない）
2. 画面録画を始める（`screenrecord --size 720x1680 --bit-rate 6000000`）
3. 分類の画面をテスト用のインテント引数 `--es prompts "発話1|発話2|…"` で開く。
   画面は発話を順に入力欄へ入れて分類する（手で押したときと同じ経路）
4. ログの `auto_classify done` を待って録画を止め、動画・最後の画面・ログ・結果（jsonl）を取り出す
5. 共有用の版（アプリが映る前を切り、最後の画面を 3 秒止める）を ffmpeg で作る（ffmpeg が無ければ飛ばす）

    python -m app.p3_demo                            # 層 17・既定の 6 文（改善版 B・案 C のデモと同じ文）
    python -m app.p3_demo --layers 26 --name demo_L26
    python -m app.p3_demo --open-only                # 資材を送って画面を開くだけ（録画しない）
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADB = r"D:\Android\sdk\platform-tools\adb.exe"
PKG = "org.edgeslm.gemma4prune"
ACT = f"{PKG}/org.edgeslm.demo.ClassifyActivity"
TAG = "EdgeSLMG4Prune"
DEV = f"/sdcard/Android/data/{PKG}/files"
REMOTE = "/data/local/tmp/edgeslm_g4prune_demo.mp4"
REC_OPTS = "--size 720x1680 --bit-rate 6000000 --time-limit 178"   # screenrecord の上限は 180 秒
OUT = ROOT / "results" / "P3_20261002_demo"
# 行き先の違う例文（samples.json にある文なら画面に PC の判定も並ぶ）＋ 意図が不明な例
DEFAULT_PROMPTS = ["エアコンを22度にして", "日本の首都はどこ", "前の曲に戻して", "近くのEV充電器を探して",
                   "車の状態を確認して", "それお願い"]
LEAD_S = 2.5          # 録画を始めてから画面を開くまで（共有用の版ではここまでを切る）


def adb(*a: str, timeout: int = 60, check: bool = True) -> str:
    r = subprocess.run([ADB, *a], capture_output=True, timeout=timeout)
    out = r.stdout.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise SystemExit(f"adb {' '.join(a)} に失敗: {r.stderr.decode('utf-8', 'replace')}")
    return out


def duration(video: Path) -> float:
    if not shutil.which("ffprobe"):
        return float("nan")
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or "nan")


def make_share(video: Path, share: Path, start: float) -> bool:
    if not shutil.which("ffmpeg"):
        print("   ffmpeg が無いので共有用の版は作らない（録画そのままは保存済み）", flush=True)
        return False
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(video), "-map", "0:v:0",
                    "-vf", "tpad=stop_mode=clone:stop_duration=3,fps=30", "-c:v", "libx264", "-crf", "20",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(share)],
                   check=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="|".join(DEFAULT_PROMPTS))
    ap.add_argument("--name", default="demo")
    ap.add_argument("--timeout", type=int, default=160)
    ap.add_argument("--layers", type=int, default=17, help="載せるグラフの層の数（work/p3/L<n>/head の資材）")
    ap.add_argument("--no-push", action="store_true", help="資材を送らない（送り済みのとき）")
    ap.add_argument("--open-only", action="store_true", help="資材を送って画面を開くだけ")
    a = ap.parse_args()
    prompts = [p for p in a.prompts.split("|") if p.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"gemma4prune_L{a.layers}_classify_{a.name}.mp4"
    share = OUT / f"gemma4prune_L{a.layers}_classify_{a.name}_share.mp4"

    others = ("org.edgeslm.gemma4headgen", "org.edgeslm.gemma4headfast", "org.edgeslm.gemma4head", "org.edgeslm.gemma4",
              "org.edgeslm.head", "org.edgeslm.demo")
    if not a.no_push:
        print(f"0) 資材を送る（層 {a.layers} 個）", flush=True)
        from app import p2_device as D
        src = ROOT / "work" / "p3" / f"L{a.layers:02d}" / "head"
        files = ["head.json", "head.bin", "samples.json"]
        if any(not (src / f).exists() for f in files):
            raise SystemExit(f"{src} に資材が無い。先に python -m app.p3_assets --layers {a.layers}")
        D.ensure_app()
        adb("shell", f"am force-stop {PKG}")
        adb("shell", f"am start -n {PKG}/org.edgeslm.demo.RunnerActivity")    # フォルダはアプリに作らせる
        time.sleep(2.0)
        for f in files:
            D.push(src / f, f"head/{f}")
    if a.open_only:
        for pkg in (PKG, *others):
            adb("shell", f"am force-stop {pkg}")
        adb("shell", "input keyevent KEYCODE_WAKEUP")
        adb("shell", "wm dismiss-keyguard")
        time.sleep(1.0)
        adb("shell", f"am start -n {ACT}")
        print("分類の画面を開きました")
        return

    print("1) アプリを止め、画面を点けてロック画面を閉じる", flush=True)
    # ほかのアプリがモデルを開いたままだとメモリが足りない（改善前の分類アプリ・Gemma4 のアプリ・前段のアプリ）
    for pkg in (PKG, *others):
        adb("shell", f"am force-stop {pkg}")
    adb("shell", "input keyevent KEYCODE_WAKEUP")
    adb("shell", "wm dismiss-keyguard")
    time.sleep(2.0)
    adb("shell", f"rm -f {REMOTE}")
    adb("logcat", "-c")

    print("2) 画面録画を始める", flush=True)
    rec = subprocess.Popen([ADB, "shell", f"screenrecord {REC_OPTS} {REMOTE}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(LEAD_S)
    if rec.poll() is not None:
        raise SystemExit(f"screenrecord が始まらなかった: {(rec.stderr.read() or b'').decode('utf-8', 'replace')}")

    print(f"3) 分類の画面を開く（{len(prompts)} 文）", flush=True)
    t0 = time.time()
    adb("shell", f"am start -n {ACT} --es prompts {shlex.quote('|'.join(prompts))}")
    status, seen = "timeout", 0
    while time.time() - t0 < a.timeout:
        time.sleep(1.0)
        log = adb("logcat", "-d", "-s", f"{TAG}:*", check=False)
        lines = [ln for ln in log.splitlines() if any(k in ln for k in ("classify ready", "auto_classify item",
                                                                           "open failed", "推論に失敗"))]
        for ln in lines[seen:]:
            print(f"   {time.time() - t0:5.1f}s  {ln.split(': ', 1)[-1][:200]}", flush=True)
        seen = len(lines)
        if "auto_classify done" in log:
            status = "done"
            break
        if "open failed" in log:
            status = "open_failed"
            break
    print(f"   → {status}（開いてから {time.time() - t0:.1f} 秒）", flush=True)

    time.sleep(4.0)                                     # 最後の結果を映しておく
    print("4) 録画を止めて取り出す", flush=True)
    adb("shell", "pkill -INT screenrecord", check=False)
    try:
        rec.wait(timeout=30)
    except subprocess.TimeoutExpired:
        rec.kill()
    time.sleep(2.0)
    adb("pull", REMOTE, str(video), timeout=180)
    (OUT / f"logcat_{a.name}.txt").write_text(adb("logcat", "-d", "-s", f"{TAG}:*", "EdgeSLMRunner:*", check=False),
                                              encoding="utf-8")
    # adb shell cat は改行を \r\r\n に変えるので pull で取る（2026-09-27 の 1 回目で jsonl の行が割れた）
    adb("pull", f"{DEV}/out/classify_auto.jsonl", str(OUT / f"classify_{a.name}.jsonl"), check=False)
    jp = OUT / f"classify_{a.name}.jsonl"
    auto = jp.read_text(encoding="utf-8") if jp.exists() else ""
    with open(OUT / f"final_screen_{a.name}.png", "wb") as f:
        subprocess.run([ADB, "exec-out", "screencap -p"], stdout=f, timeout=60)

    print("5) 共有用の版を作る", flush=True)
    made = make_share(video, share, LEAD_S)
    print(f"   録画そのまま {video.name}（{duration(video):.1f} 秒・{video.stat().st_size / 2**20:.1f}MiB）", flush=True)
    if made:
        print(f"   共有用      {share.name}（{duration(share):.1f} 秒・{share.stat().st_size / 2**20:.1f}MiB）", flush=True)

    print("\n===== 結果 =====", flush=True)
    for ln in auto.splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        pc = r.get("pc_final")
        mark = "" if pc is None else ("  一括収集と一致" if pc == r["final"] else f"  ★一括収集は {pc}")
        print(f"  {r['utterance']:<16} → {r['final']:<28} 確信度 {r['conf']:.2f} 棄却 {r['reject']:.3f} "
              f"NPU {r['npu_total_ms']:.0f}ms（{r['n_tokens']} トークン）{mark}", flush=True)
    print(f"保存しました: {OUT}", flush=True)
    sys.exit(0 if status == "done" else 1)


if __name__ == "__main__":
    main()
