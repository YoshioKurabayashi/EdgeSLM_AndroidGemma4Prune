# -*- coding: utf-8 -*-
r"""P38: 判定デモアプリ v2（Gemma4 判定デモ v2・`org.edgeslm.gemma4router2`）を**前半 17 層**で動かし、判定の様子を録画する。

設計: DoC/設計書_P38_層17の判定デモ動画.md。録画の手順は `app/p3_demo.py`（P3 の分類デモ）と同じ。送り込み・アプリの止め方は `app/p36_device.py` を使う（どちらも変更しない）。

1. ほかの Gemma のアプリを止め、画面を点けてロック画面を閉じる（端末の設定は変えない）
2. 画面録画を始める（`screenrecord --size 720x1680 --bit-rate 6000000`。上限 178 秒）
3. アプリ v2 を `--es prompts "17:|発話1|発話2|…"` で開く。アプリは既定の層 26 を開いたあと、「17:」で前半 17 層に切り替え、発話を順に判定する（手で押したときと同じ経路。1 件ごとに結果を 2.5 秒見せる）
4. ログの `router_auto done` を待って録画を止め、動画・最後の画面・ログ・結果（`out/router_auto.jsonl`）を取り出す
5. 共有用の版（層 17 に切り替える直前から。最後の画面を 3 秒止める）を ffmpeg で作る（ffmpeg が無ければ飛ばす）
6. 端末の判定を、PC の判定（評価セット v10・v11 の確かめ用の行にある層 17 の判定）と突き合わせ、`README.md` に書く

既定の 15 文は、評価セット v10・v11 のうち、**端末の層 17 と層 26 が同じ判定で、PC とも一致し、数え方で正解**になった行から、行き先ごとに選んだもの（実行 9・聞き返し 2・断る 2・雑談 2）。

    python -m app.p38_record                    # 既定の 15 文を層 17 で録画
    python -m app.p38_record --open-only        # 画面を開いて 15 文を流すだけ（録画しない。下見）
    python -m app.p38_record --layer 26 --name demo_L26
    python -m app.p38_record --prompts "車内を21度に設定してください|次の曲にしてください" --name short
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

from .common import PROJECT_ROOT
from . import p36_device as D

ROOT = PROJECT_ROOT
OUT = ROOT / "results" / "P38_20261015_L17_demo"
REMOTE = "/data/local/tmp/edgeslm_g4router2_demo.mp4"
REC_OPTS = "--size 720x1680 --bit-rate 6000000 --time-limit 178"   # screenrecord の上限は 180 秒
LEAD_S = 2.5          # 録画を始めてから画面を開くまで
CFG37 = ROOT / "work" / "p37" / "device" / "router" / "router.json"   # P37 で送った設定（確かめ用の行が v11）
EVALS = [ROOT / "work" / "p36" / "device" / "router" / "eval_v10.jsonl",
         ROOT / "work" / "p37" / "device" / "router" / "eval_v11.jsonl"]   # PC の判定（層ごと）
# （行き先, 発話）。評価セット v10・v11 で、端末の層 17・26 が同じ判定・PC と一致・正解だった行
DEFAULT = [
    ("エアコン（温度）", "車内を21度に設定してください"),
    ("エアコン（モード）", "暖房をつけてほしい"),
    ("音楽", "次の曲にしてください"),
    ("ナビ", "大阪駅へのルートを案内して"),
    ("車両の状態", "ガソリンはあとどれだけ残っている"),
    ("再生中の曲", "この曲の名前は何"),
    ("取扱説明書", "スペアタイヤの場所はどこ"),
    ("周辺の施設", "近くにある喫茶店を教えて"),
    ("カメラ", "後ろの様子を確認したい"),
    ("聞き返す（言いかけ）", "行き先の場所は、あのー"),
    ("聞き返す（言い直し）", "冷房、あ、暖房で"),
    ("断る（返さないものを聞く）", "近くのホテルの宿泊料金を調べて"),
    ("断る（機能にない）", "空気圧モニターの警告表示をクリアして"),
    ("雑談（知識を聞く）", "東京タワーができた年を教えて"),
    ("雑談", "明日は休みなんだ"),
]


def duration(video: Path) -> float:
    if not shutil.which("ffprobe") or not video.exists():
        return float("nan")
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return float("nan")


def make_share(video: Path, share: Path, start: float) -> bool:
    if not shutil.which("ffmpeg"):
        print("   ffmpeg が無いので共有用の版は作らない（録画そのままは保存済み）", flush=True)
        return False
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(video), "-map", "0:v:0",
                    "-vf", "tpad=stop_mode=clone:stop_duration=3,fps=30", "-c:v", "libx264", "-crf", "20",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(share)],
                   check=True)
    return True


def pc_table(layer: int) -> dict[str, dict]:
    """発話 → PC の判定（評価セット v10・v11 の確かめ用の行。後の版を優先）"""
    t: dict[str, dict] = {}
    for p in EVALS:
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                pc = (r.get("pc") or {}).get(str(layer))
                if pc:
                    t[r["utterance"]] = {"id": r["id"], **pc}
    return t


def check_config() -> None:
    """端末の設定（router.json）が P37 で送ったものか確かめる。違えば P37 のものを送る（P37 の deploy と同じ）"""
    if not CFG37.exists():
        raise SystemExit(f"{CFG37} が無い。先に .\\scripts\\run_p37.ps1 -AppOnly")
    if D.dev_md5(f"{D.DEV}/router/router.json") != D.md5(CFG37):
        print("  端末の router.json が P37 のものと違うので、P37 のものを送る", flush=True)
        D.push(CFG37, "router/router.json")
    for rel in ("router/L17/head.json", "router/bert/model.onnx", "models/ops/p2_L17_epctx.onnx"):
        if not D.dev_md5(f"{D.DEV}/{rel}"):
            raise SystemExit(f"端末に {rel} が無い。先に .\\scripts\\run_p36.ps1 -Stage deploy")


def same(dev: dict, pc: dict | None) -> bool | None:
    if not pc:
        return None
    return (dev.get("outcome"), dev.get("stage"), dev.get("tool"), dev.get("args")) == \
           (pc.get("outcome"), pc.get("stage"), pc.get("tool"), pc.get("args"))


def write_readme(layer: int, kinds: dict[str, str], rows: list[dict], pc: dict[str, dict], video: Path,
                 share: Path | None, status: str, t_open: float, t_ready: float | None) -> int:
    n_same = 0
    lines = []
    for i, r in enumerate(rows, 1):
        p = pc.get(r["utterance"])
        s = same(r, p)
        n_same += bool(s)
        got = r["outcome"] + (f"（{r['stage']}）" if r.get("stage") else "") + (f" {r['tool']}" if r.get("tool") else "") + \
            (f" `{r['args']}`" if r.get("args") else "")
        want = "—" if not p else p["outcome"] + (f"（{p['stage']}）" if p.get("stage") else "")
        if p and s is False:              # 違うときだけ、PC の機能・引数も出す
            want += (f" {p['tool']}" if p.get("tool") else "") + (f" `{p['args']}`" if p.get("args") else "")
        lines.append(f"|{i}|{kinds.get(r['utterance'], '')}|{r['utterance']}|{got}|{want}|"
                     f"{'一致' if s else ('—' if s is None else '**不一致**')}|{r.get('npu_total_ms', float('nan')):.0f}|"
                     f"{r.get('wall_ms', float('nan')):.0f}|")
    npu = sorted(r.get("npu_total_ms", 0) for r in rows)
    wall = sorted(r.get("wall_ms", 0) for r in rows)
    med = lambda xs: xs[len(xs) // 2] if xs else float("nan")
    md = [
        f"# P38: 判定デモアプリ v2 を前半 {layer} 層で動かした動画",
        "",
        f"録画: {time.strftime('%Y-%m-%d %H:%M')}（`python -m app.p38_record`・設計: [DoC/設計書_P38_層17の判定デモ動画.md](../../DoC/設計書_P38_層17の判定デモ動画.md)）",
        "",
        f"- アプリ: 「Gemma4 判定デモ v2」（`{D.PKG}`・P36）。Gemma 4 E2B の前半 {layer} 層（NPU）＋分類ヘッド・日本語 BERT（CPU）・規則・引数の取り出し・機能の定義での絞り込み・検査。**通信は使わない**。操作は模擬",
        f"- 結果: {status}。{len(rows)} 文のうち **PC と同じ判定 {n_same}**（PC の判定は評価セット v10・v11 の確かめ用の行の、層 {layer} の判定）",
        f"- 時間（中央値）: NPU {med(npu):.0f}ms・1 文の判定全体 {med(wall):.0f}ms（Gemma のトークン化・BERT・規則・引数を含む）",
        f"- 画面を開いてから層 {layer} の準備ができるまで: " + ("—" if t_ready is None else f"{t_ready - t_open:.1f} 秒（既定の層 26 を開いてから切り替えるため）"),
        "",
        "|ファイル|中身|",
        "|---|---|",
        f"|`{video.name}`|録画そのまま（{duration(video):.1f} 秒）|",
    ]
    if share is not None:
        md.append(f"|`{share.name}`|共有用（層 {layer} に切り替える直前から。最後の画面を 3 秒止める。{duration(share):.1f} 秒）|")
    md += [
        "|`final_screen_*.png`|最後の画面|",
        "|`router_auto_*.jsonl`|端末の判定（1 文 1 行）|",
        "|`logcat_*.txt`|ログ（`EdgeSLMRouter`）|",
        "",
        "|#|行き先|発話|端末の判定|PC の判定|PC と|NPU ms|全体 ms|",
        "|---|---|---|---|---|---|---|---|",
        *lines,
        "",
    ]
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
    return n_same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="", help="発話を | でつなぐ（既定は 15 文）")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--name", default="demo")
    ap.add_argument("--timeout", type=int, default=165)
    ap.add_argument("--open-only", action="store_true", help="画面を開いて流すだけ（録画しない）")
    a = ap.parse_args()
    prompts = [p.strip() for p in a.prompts.split("|") if p.strip()] if a.prompts else [u for _, u in DEFAULT]
    kinds = {u: k for k, u in DEFAULT}
    arg = "|".join([f"{a.layer}:", *prompts])
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"gemma4router2_L{a.layer}_judge_{a.name}.mp4"
    share = OUT / f"gemma4router2_L{a.layer}_judge_{a.name}_share.mp4"

    print("1) ほかの Gemma のアプリを止め、資材を確かめる", flush=True)
    D.ensure_app()                       # アプリ v2 が入っているか確かめ、ほかのモデルのアプリを止める
    check_config()
    D.adb("shell", f"am force-stop {D.PKG}")
    D.adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    D.adb("shell", "wm dismiss-keyguard", check=False)
    time.sleep(2.0)
    if a.open_only:
        D.adb("shell", f"am start -n {D.MAIN} --es prompts {shlex.quote(arg)}")
        print(f"画面を開きました（層 {a.layer} に切り替えてから {len(prompts)} 文を流します）")
        return
    D.adb("shell", f"rm -f {REMOTE}", check=False)
    D.adb("logcat", "-c", check=False)

    print("2) 画面録画を始める", flush=True)
    rec = subprocess.Popen([D.ADB, "shell", f"screenrecord {REC_OPTS} {REMOTE}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t_rec = time.time()
    time.sleep(LEAD_S)
    if rec.poll() is not None:
        raise SystemExit(f"screenrecord が始まらなかった: {(rec.stderr.read() or b'').decode('utf-8', 'replace')}")

    print(f"3) アプリ v2 を開く（層 {a.layer}・{len(prompts)} 文）", flush=True)
    t_open = time.time()
    D.adb("shell", f"am start -n {D.MAIN} --es prompts {shlex.quote(arg)}")
    status, seen, t_ready = "timeout", 0, None
    keys = ("router ready", " ready load_ms", "router_auto item", "router_fatal", "open failed")
    while time.time() - t_open < a.timeout:
        time.sleep(0.5)
        log = D.adb("logcat", "-d", "-s", f"{D.TAG}:*", check=False)
        lines = [ln for ln in log.splitlines() if any(k in ln for k in keys)]
        for ln in lines[seen:]:
            print(f"   {time.time() - t_open:5.1f}s  {ln.split(': ', 1)[-1][:180]}", flush=True)
        seen = len(lines)
        if t_ready is None and f"layer {a.layer} ready" in log:
            t_ready = time.time()
        if "router_auto done" in log:
            status = "done"
            break
        if "router_fatal" in log:
            status = "fatal"
            break
    print(f"   → {status}（開いてから {time.time() - t_open:.1f} 秒）", flush=True)

    time.sleep(4.0)                                     # 最後の結果を映しておく
    print("4) 録画を止めて取り出す", flush=True)
    D.adb("shell", "pkill -INT screenrecord", check=False)
    try:
        rec.wait(timeout=30)
    except subprocess.TimeoutExpired:
        rec.kill()
    time.sleep(2.0)
    D.adb("pull", REMOTE, str(video), timeout=300)
    (OUT / f"logcat_{a.name}.txt").write_text(D.adb("logcat", "-d", "-s", f"{D.TAG}:*", check=False), encoding="utf-8")
    jp = OUT / f"router_auto_{a.name}.jsonl"
    D.adb("pull", f"{D.DEV}/out/router_auto.jsonl", str(jp), check=False)   # adb shell cat は改行が割れるので pull
    with open(OUT / f"final_screen_{a.name}.png", "wb") as f:
        subprocess.run([D.ADB, "exec-out", "screencap -p"], stdout=f, timeout=60)

    print("5) 共有用の版を作る", flush=True)
    start = LEAD_S if t_ready is None else max(LEAD_S, t_ready - t_rec - 4.0)   # 「前半 17 層のグラフを開いています」から
    made = make_share(video, share, start)
    print(f"   録画そのまま {video.name}（{duration(video):.1f} 秒・{video.stat().st_size / 2**20:.1f}MiB）", flush=True)
    if made:
        print(f"   共有用      {share.name}（{start:.1f} 秒目から・{duration(share):.1f} 秒・{share.stat().st_size / 2**20:.1f}MiB）", flush=True)

    print("6) PC の判定と突き合わせる", flush=True)
    rows = []
    if jp.exists():
        for ln in jp.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    rows = [r for r in rows if int(r.get("layer", a.layer)) == a.layer]
    pc = pc_table(a.layer)
    n_same = write_readme(a.layer, kinds, rows, pc, video, share if made else None, status, t_open, t_ready)
    print("\n===== 結果 =====", flush=True)
    for r in rows:
        s = same(r, pc.get(r["utterance"]))
        mark = "" if s is None else ("  PC と一致" if s else "  ★PC と違う")
        print(f"  {r['utterance']:<22} → {r['outcome']}{'（' + r['stage'] + '）' if r.get('stage') else ''} "
              f"{r.get('tool') or ''} {r.get('args') or ''}  NPU {r.get('npu_total_ms', 0):.0f}ms{mark}", flush=True)
    print(f"{len(rows)} 文・PC と同じ判定 {n_same}。保存しました: {OUT}", flush=True)
    sys.exit(0 if status == "done" and len(rows) == len(prompts) else 1)


if __name__ == "__main__":
    main()
