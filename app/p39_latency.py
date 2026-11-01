# -*- coding: utf-8 -*-
r"""P39: 判定 1 回の時間を、部品ごとに分けて測る（判定デモ v3・測定 `org.edgeslm.gemma4router3`・`android_app_router3/`）。

設計: DoC/設計書_P39_時間の内訳を測る.md。アプリ v2（`android_app_router2/`）の複製に時計を足したアプリを使う（v2 とその置き場は変えない）。
端末を USB で接続して、利用者が実行する。VLM の作業と同時に動かさない。

|段|何をするか|
|---|---|
|`prep`|v3 の設定（`work/p39/device/router/router.json`）を作る。P37 の設定（v2・確かめ用の行は v11）を写し、層に **35 層（改善版 B のグラフ `g4cls64_full`）** を足す。35 層のヘッドは無いので層 26 のヘッドを使う（**35 層は時間だけ見る。判定は見ない**）。既定の層は 17|
|`deploy`|v3 の置き場へ、表・トークナイザ・グラフ 3 つ・BERT と辞書・ヘッド・設定・確かめ用の行を送る（md5 を確かめる）。端末の v2・お試しアプリの置き場に同じものがあれば端末の中で写す（読むだけ）|
|`bench`|v3 を `--es bench "17,26,35"` で開き、確かめ用の行（v11）の先頭 N 行を、層ごとに par（v2 と同じ。BERT を並行）・serial（BERT を先に単独で）の順に流す。最後に QNN の計測（`profiling_level`）を付けて開き直し、30 行流す。結果を `work/p39/device_io/` へ取り出す|
|`report`|部品ごとの時間（中央値・p90）・足し合わせの確かめ・PC と同じ判定か（層 17・26）・QNN の計測のまとめ → `results/P39/latency.md`|

    python -m app.p39_latency prep
    python -m app.p39_latency deploy
    python -m app.p39_latency bench --layers 17,26,35 --rows 300
    python -m app.p39_latency report
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

from .common import HEAD_PRJ, PROJECT_ROOT
from .p2_device import ADB, GRAPHS, OTHERS, TABLES, TOKENIZER, TOKENIZER_FILES, ctx_of

PKG = "org.edgeslm.gemma4router3"
MAIN = f"{PKG}/org.edgeslm.demo.RouterActivity"
TAG = "EdgeSLMRouter"
DEV = f"/sdcard/Android/data/{PKG}/files"
FROM = ["/sdcard/Android/data/org.edgeslm.gemma4router2/files",      # 読むだけ（同じものがあれば端末の中で写す）
        "/sdcard/Android/data/org.edgeslm.gemma4prune/files"]
STOP = [*OTHERS, "org.edgeslm.gemma4prune", "org.edgeslm.gemma4router", "org.edgeslm.gemma4router2"]   # 同時にモデルを開くと RAM が足りない
WORK = PROJECT_ROOT / "work" / "p39"
OUT = WORK / "device" / "router"
IO = WORK / "device_io"
RES = PROJECT_ROOT / "results" / "P39"
CFG37 = PROJECT_ROOT / "work" / "p37" / "device" / "router" / "router.json"
EVAL11 = PROJECT_ROOT / "work" / "p37" / "device" / "router" / "eval_v11.jsonl"
HEADS36 = PROJECT_ROOT / "work" / "p36" / "device" / "router"
BERT_SRC = PROJECT_ROOT / "work" / "p24" / "device" / "router" / "bert"
LAYERS = (17, 26, 35)


# ---------------------------------------------------------------- adb

def adb(*a: str, timeout: int = 900, check: bool = True) -> str:
    r = subprocess.run([ADB, *a], capture_output=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"adb {' '.join(a)}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8", "replace")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dev_md5(path: str) -> str:
    o = adb("shell", f"md5sum {path} 2>/dev/null", check=False).split()
    return o[0] if o and len(o[0]) == 32 else ""


def push(src: Path, rel: str) -> None:
    """md5 が同じなら飛ばす。v2・お試しアプリの置き場に同じもの（同じ相対の場所）があれば端末の中で写す。無ければ送る"""
    want = md5(src)
    if dev_md5(f"{DEV}/{rel}") == want:
        print(f"  skip {rel}（端末に同じものがある）", flush=True)
        return
    # フォルダはアプリが作る（adb で作ると持ち主が shell になり、アプリから読めない）
    for base in FROM:
        if dev_md5(f"{base}/{rel}") == want:
            adb("shell", f"cp {base}/{rel} {DEV}/{rel}", timeout=1800, check=False)
            if dev_md5(f"{DEV}/{rel}") == want:
                print(f"  copy {rel}（{base.split('/')[4]} の置き場から端末の中で写した・{src.stat().st_size / 2**20:.0f}MiB）md5 一致", flush=True)
                return
    adb("push", str(src), f"{DEV}/{rel}", timeout=1800)
    if dev_md5(f"{DEV}/{rel}") != want:
        raise SystemExit(f"md5 が合わない: {rel}")
    print(f"  push {rel}（{src.stat().st_size / 2**20:.0f}MiB）md5 一致", flush=True)


def ensure_app() -> None:
    if PKG not in adb("shell", f"pm list packages {PKG}"):
        raise SystemExit(f"端末に {PKG} が入っていない。先に .\\scripts\\run_p39.ps1 -Stage build")
    for p in STOP:
        adb("shell", f"am force-stop {p}", check=False)


def alive() -> bool:
    return bool(adb("shell", f"pidof {PKG}", check=False).strip())


# ---------------------------------------------------------------- prep・deploy

def prep() -> None:
    cfg = json.loads(CFG37.read_text(encoding="utf-8"))
    if not any(int(l["n"]) == 35 for l in cfg["layers"]):
        cfg["layers"].append({"n": 35, "label": "35 層（全部・時間だけ）", "model": "models/ops/g4cls64_full_epctx.onnx",
                              "head": "router/L26", "context_mib": 1273,
                              "detail": "Gemma 4 E2B の 35 層すべて（改善版 B のグラフ g4cls64_full・1,273MiB）。ヘッドは層 26 のものを借りる（判定は見ない。時間を測るためだけ）"})
    cfg["default_layer"] = 17
    cfg["eval"] = "router/eval_v11.jsonl"
    cfg["title"] = "車載ルーター 判定デモ v3・測定（時間の内訳）"
    cfg["version"] = "v3 測定（P39・2026-10-15）"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "router.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書きました: {OUT / 'router.json'}（層 {[l['n'] for l in cfg['layers']]}・既定 17・確かめ用の行 v11）", flush=True)


def free_mib() -> float:
    o = adb("shell", "df -k /sdcard", check=False).splitlines()
    try:
        return int(o[-1].split()[3]) / 1024
    except (IndexError, ValueError):
        return float("nan")


def deploy() -> None:
    if not (OUT / "router.json").exists():
        raise SystemExit("先に python -m app.p39_latency prep")
    ensure_app()
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", f"am force-stop {PKG}")
    adb("shell", f"am start -n {MAIN}")           # フォルダはアプリに作らせる（adb で作ると持ち主が shell になり、アプリから読めない）
    time.sleep(3.0)
    adb("shell", f"am force-stop {PKG}")
    if "yes" not in adb("shell", f"[ -d {DEV} ] && echo yes", check=False):
        raise SystemExit(f"アプリが置き場（{DEV}）を作っていない。画面が開けたか確認")
    print(f"== {PKG} の置き場へ（端末の空き {free_mib():,.0f}MiB。全部写すと約 4,300MiB 使う）", flush=True)
    t = json.loads((TABLES / "tables.json").read_text(encoding="utf-8"))
    for f in [t["embed"]["file"], t["embed"]["scale_file"], t["ple"]["file"], t["ple"]["scale_file"], *t["rope"]["files"].values()]:
        push(TABLES / f, f"models/ops/{f}")
    for f in TOKENIZER_FILES:
        push(TOKENIZER / f, f"tokenizer/{f}")
    for n in LAYERS:
        e, s = GRAPHS[n]
        b = ctx_of(e, s)
        push(b, f"models/ops/{b.name}")
        push(e, f"models/ops/{e.name}")
    for p in sorted(BERT_SRC.iterdir()):
        if p.is_file():
            push(p, f"router/bert/{p.name}")
    for n in (17, 26):
        for p in sorted((HEADS36 / f"L{n}").iterdir()):
            if p.is_file():
                push(p, f"router/L{n}/{p.name}")
    push(EVAL11, "router/eval_v11.jsonl")
    push(OUT / "router.json", "router/router.json")
    print(f"送り終わりました（端末の空き {free_mib():,.0f}MiB）", flush=True)


# ---------------------------------------------------------------- bench

def bench(layers: str, rows: int, modes: str, qnnprof: str, prof_rows: int, timeout_s: int) -> None:
    ensure_app()
    IO.mkdir(parents=True, exist_ok=True)
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", f"am force-stop {PKG}")
    time.sleep(8.0)
    adb("shell", f"rm -f {DEV}/out/bench_*.jsonl {DEV}/out/qnnprof_*", check=False)
    adb("logcat", "-c", check=False)
    extra = f"--es bench {shlex.quote(layers)} --es bench_rows {rows} --es bench_modes {shlex.quote(modes)}"
    if qnnprof and qnnprof != "off":
        extra += f" --es qnnprof {qnnprof} --es qnnprof_rows {prof_rows}"
    adb("shell", f"am start -n {MAIN} {extra}")
    t0, seen = time.time(), 0
    while True:
        time.sleep(10)
        log = adb("logcat", "-d", "-s", f"{TAG}:*", check=False)
        lines = [x for x in log.splitlines() if "router_bench" in x or " ready" in x or "router_fatal" in x or "open failed" in x]
        for x in lines[seen:]:
            print(f"  {time.time() - t0:6.0f}s  " + x.split(f"{TAG}:")[-1].strip()[:170], flush=True)
        seen = len(lines)
        if "router_bench done" in log:
            break
        if "router_fatal" in log:
            raise SystemExit("端末の測定が失敗した（logcat の EdgeSLMRouter）:\n" + "\n".join(log.splitlines()[-15:]))
        if time.time() - t0 > 60 and not alive():
            raise SystemExit("アプリが結果を出さずに終わった（35 層でメモリが足りなかった可能性。logcat を確認）:\n"
                             + adb("logcat", "-d", "-t", "80", check=False)[-4000:])
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"{timeout_s} 秒たっても終わらない")
    (IO / "logcat_bench.txt").write_text(adb("logcat", "-d", "-s", f"{TAG}:*", "EdgeSLMRunner:*", check=False), encoding="utf-8")
    names = adb("shell", f"ls {DEV}/out/", check=False).split()
    got = []
    for nm in names:
        if nm.startswith("bench_") or nm.startswith("qnnprof_"):
            adb("pull", f"{DEV}/out/{nm}", str(IO / nm))
            got.append(nm)
    print(f"取り出しました（{IO}）: {', '.join(sorted(got))}", flush=True)


# ---------------------------------------------------------------- report

def q(xs: list[float], p: float) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    i = p * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def parts(r: dict) -> dict[str, float]:
    """1 行の時間を、部品に分ける（ms）。名前は report の表の行"""
    tm = r.get("timing") or {}
    jt = r.get("jni_timing") or {}
    g = lambda k, d=tm: float(d.get(k, 0.0) or 0.0)
    native_in_job = sum(g(k) for k in ("pre_describe_versions", "describe", "embed_open", "ple_open", "alloc", "rope", "mask",
                                       "fill_embed", "fill_ple", "run_wall", "h_mean", "h_json"))
    jni_inner = g("get_string", jt) + g("parse_job", jt) + g("run_job", jt)
    p = {
        "Gemma のトークン化（Java）": r.get("gemma_tok_ms", 0.0),
        "job の JSON を作る（Java）": r.get("job_build_ms", 0.0),
        "JNI: 文字列の受け取り＋job の JSON を読む（C++）": g("get_string", jt) + g("parse_job", jt),
        "毎回の describe・versions（C++）": g("pre_describe_versions") + g("describe"),
        "埋め込みの表を開く＋スケールを読む（C++）": g("embed_open"),
        "PLE の表を開く＋スケールを読む（C++・35 層ぶん）": g("ple_open"),
        "入力の器を作る（C++）": g("alloc"),
        "RoPE の表を読んで写す（C++）": g("rope"),
        "マスクを作る（C++）": g("mask"),
        "埋め込みの表引き（C++・T=64 行）": g("fill_embed"),
        "PLE の表引き（C++・T=64 行 × 35 層）": g("fill_ple"),
        "NPU（ORT の Run）": g("run_ort"),
        "Run の前後（入力の包み・出力の写し）": g("run_wall") - g("run_ort"),
        "h の平均（C++）": g("h_mean"),
        "h を JSON の配列にする（C++）": g("h_json"),
        "job の残り（結果の組み立てなど・C++）": g("run_job", jt) - native_in_job,
        "結果を文字列にして Java へ返す（dump・JNI）": r.get("jni_ms", 0.0) - jni_inner,
        "結果の JSON を読む（Java）": r.get("parse_ms", 0.0),
        "h を float にする（Java）": r.get("floats_ms", 0.0),
        "ヘッド（Java）": r.get("head_ms", 0.0),
        "BERT を待った時間（並行のとき）": r.get("bert_wait_ms", 0.0),
        "BERT（単独・先に動かしたとき）": r.get("bert_path_ms", 0.0) if r.get("serial") else 0.0,
        "規則・引数・検査（Java）": r.get("rule_ms", 0.0),
    }
    p["測っていない残り"] = r.get("wall_ms", 0.0) - sum(p.values())
    return p


GROUPS = [
    ("Gemma の層（NPU）", ["NPU（ORT の Run）"]),
    ("埋め込み・PLE の表引き（CPU・毎回）", ["埋め込みの表引き（C++・T=64 行）", "PLE の表引き（C++・T=64 行 × 35 層）"]),
    ("RoPE・マスク（毎回作り直し）", ["RoPE の表を読んで写す（C++）", "マスクを作る（C++）"]),
    ("表を開く・スケールを読む・器（毎回）", ["埋め込みの表を開く＋スケールを読む（C++）", "PLE の表を開く＋スケールを読む（C++・35 層ぶん）", "入力の器を作る（C++）",
                                         "毎回の describe・versions（C++）"]),
    ("受け渡し（JSON・JNI）", ["job の JSON を作る（Java）", "JNI: 文字列の受け取り＋job の JSON を読む（C++）", "h を JSON の配列にする（C++）",
                          "job の残り（結果の組み立てなど・C++）", "結果を文字列にして Java へ返す（dump・JNI）", "結果の JSON を読む（Java）", "h を float にする（Java）",
                          "Run の前後（入力の包み・出力の写し）"]),
    ("トークン化・ヘッド・h の平均・規則", ["Gemma のトークン化（Java）", "ヘッド（Java）", "h の平均（C++）", "規則・引数・検査（Java）"]),
    ("BERT（並行: 待った分／単独: 全部）", ["BERT を待った時間（並行のとき）", "BERT（単独・先に動かしたとき）"]),
    ("測っていない残り", ["測っていない残り"]),
]


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def qnn_summary(p: Path) -> list[str]:
    """QNN の計測の CSV を、ざっくりまとめる（形が版で違うので、列の名前を見て合わせる。読めなければ先頭を出す）"""
    if not p.exists() or p.stat().st_size == 0:
        return [f"- `{p.name}`: 無い（計測を付けて開けなかったか、何も書かれなかった）"]
    raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out = [f"- `{p.name}`: {len(raw):,} 行・{p.stat().st_size / 1024:.0f}KiB"]
    try:
        rd = list(csv.DictReader(raw))
    except csv.Error:
        rd = []
    if not rd:
        return out + ["", "```", *raw[:20], "```"]
    cols = list(rd[0].keys())
    name_c = next((c for c in cols if c and "Identifier" in c), None) or \
        next((c for c in cols if c and c.strip().lower() in ("message", "event", "name")), None)
    time_c = next((c for c in cols if c and c.strip().lower() in ("time", "value", "duration")), None)
    unit_c = next((c for c in cols if c and "Unit" in c), None)
    out.append(f"- 列: {', '.join(c for c in cols if c)}")
    if not name_c or not time_c:
        return out + ["", "```", *raw[:20], "```"]
    agg: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rd:
        try:
            v = float(r[time_c])
        except (TypeError, ValueError):
            continue
        agg[(r.get(name_c, ""), r.get(unit_c, "") if unit_c else "")].append(v)
    out += ["", f"|{name_c}|単位|回数|中央値|合計|", "|---|---|---|---|---|"]
    for (nm, u), vs in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:40]:
        out.append(f"|{nm[:80]}|{u}|{len(vs)}|{q(vs, 0.5):,.1f}|{sum(vs):,.0f}|")
    return out


CHUNKS = {17: {"A": "入口＋層 0〜8", "c1": "層 9〜16＋最後の正規化"},
          26: {"A": "入口＋層 0〜8", "c1": "層 9〜17", "c2": "層 18〜25＋最後の正規化"},
          35: {"A": "入口＋層 0〜8", "c1": "層 9〜17", "c2": "かたまり 3", "c3": "かたまり 4", "c4": "かたまり 5（最後の正規化まで）"}}
ROOT_US = ["Accelerator (execute) time", "QNN accelerator (execute) time", "RPC (execute) time", "QNN (execute) time"]


def qnn_chunks(p: Path, n: int) -> list[str]:
    """QNN の計測（detailed）を、1 回の実行ごとに、かたまり（ノード名の頭 c1_・c2_ …。無いものは入口のかたまり A）と入口の数ノードに分ける。
    計測を付けると実行が遅くなる（詳しく記録するため）ので、時間ではなく**サイクルの割合**で見る"""
    import re
    if not p.exists():
        return []
    runs, cur, root = [], None, collections.defaultdict(list)
    for r in csv.DictReader(p.read_text(encoding="utf-8", errors="replace").splitlines()):
        eid, lvl = r.get("Event Identifier", ""), r.get("Event Level", "")
        try:
            v = float(r.get("Time", "nan"))
        except ValueError:
            continue
        if lvl == "ROOT":
            if "load binary" in eid:
                continue
            root[eid].append(v)
            if eid == "Accelerator (execute) time (cycles)":
                cur = collections.Counter()
                runs.append(cur)
        elif cur is not None:
            name = eid.replace(" (cycles)", "")
            m = re.match(r"(c\d+)_", name)
            cur[m.group(1) if m else "A"] += v
            op = re.search(r"OpId_(\d+)", name)
            if op and int(op.group(1)) <= 97 and not m:       # 入口（埋め込みの変換・PLE の投影と正規化・ple_tok の変換。OpId 97 まで）
                cur["entry"] += v
    if not runs:
        return []
    acc = q(root["Accelerator (execute) time (cycles)"], .5)
    out = ["", f"かたまりごとのサイクル（{len(runs)} 回の中央値・全体 {acc / 1e6:,.1f}M サイクルに対する割合）:", "", "|かたまり|中身|M サイクル|割合|", "|---|---|---|---|"]
    for k in sorted({k for r in runs for k in r if k != "entry"}, key=lambda x: (x != "A", x)):
        c = q([r[k] for r in runs], .5)
        out.append(f"|{k}|{CHUNKS.get(n, {}).get(k, '')}|{c / 1e6:,.1f}|{c / acc * 100:.1f}%|")
    e = q([r["entry"] for r in runs], .5)
    out.append(f"|（A のうち入口）|埋め込みの変換・PLE の投影（35 層ぶん）と正規化・ple_tok の変換（OpId ≤ 97）|{e / 1e6:,.2f}|{e / acc * 100:.1f}%|")
    out += ["", "計測を付けたときの時間（中央値・ms。計測の分だけ遅い）: " + "・".join(f"{k} {q(root[k], .5) / 1000:.1f}" for k in ROOT_US if root.get(k))]
    return out


def report() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    files = sorted(IO.glob("bench_L*_*.jsonl"))
    if not files:
        raise SystemExit(f"{IO} に結果が無い。先に bench")
    md = ["# P39: 判定 1 回の時間の内訳（判定デモ v3・測定）", "",
          f"作成: {time.strftime('%Y-%m-%d %H:%M')}（`python -m app.p39_latency report`）｜ 設計: [DoC/設計書_P39_時間の内訳を測る.md](../../DoC/設計書_P39_時間の内訳を測る.md）",
          "", "数字は ms・中央値（括弧は p90）。par = アプリ v2 と同じ（BERT を NPU と並行）、serial = BERT を先に単独で動かしてから Gemma。35 層は層 26 のヘッドを借りたので**判定は見ない（時間だけ）**。", ""]
    data = {}
    for f in files:
        _, ln, mode = f.stem.split("_", 2)
        rows = [r for r in load(f) if not r.get("error")]
        data[(int(ln[1:]), mode)] = rows
    keys = sorted(data)
    # 概要
    md += ["## 1. 概要（グループごと）", "", "|グループ|" + "|".join(f"L{n} {m}" for n, m in keys) + "|", "|---|" + "---|" * len(keys)]
    P = {k: [parts(r) for r in data[k]] for k in keys}
    for g, members in GROUPS:
        cells = []
        for k in keys:
            vs = [sum(p[m] for m in members) for p in P[k]]
            cells.append(f"{q(vs, .5):.1f}（{q(vs, .9):.1f}）")
        md.append(f"|{g}|" + "|".join(cells) + "|")
    cells = [f"**{q([r['wall_ms'] for r in data[k]], .5):.1f}**（{q([r['wall_ms'] for r in data[k]], .9):.1f}）" for k in keys]
    md.append("|**判定 1 回の全体（wall）**|" + "|".join(cells) + "|")
    md.append("|行の数|" + "|".join(str(len(data[k])) for k in keys) + "|")
    # 詳細
    md += ["", "## 2. 部品ごと", "", "|部品|" + "|".join(f"L{n} {m}" for n, m in keys) + "|", "|---|" + "---|" * len(keys)]
    for name in P[keys[0]][0].keys():
        md.append(f"|{name}|" + "|".join(f"{q([p[name] for p in P[k]], .5):.2f}（{q([p[name] for p in P[k]], .9):.1f}）" for k in keys) + "|")
    # 参考の内訳
    md += ["", "## 3. 参考", ""]
    for k in keys:
        rs = data[k]
        bert = [r.get("bert_ms", 0) + r.get("bert_tok_ms", 0) for r in rs]
        gem = [r.get("gemma_path_ms", 0) for r in rs]
        same = [r.get("pc_same") for r in rs if r.get("pc_same") is not None]
        tok = [((r.get("timing") or {}).get("real_tokens", 0)) for r in rs]
        md.append(f"- L{k[0]} {k[1]}: BERT（トークン化＋推論）{q(bert, .5):.1f}ms・Gemma 側（トークン化〜ヘッド）{q(gem, .5):.1f}ms・本物のトークン {q(tok, .5):.0f} 個（T=64）"
                  + ("" if k[0] == 35 else f"・PC と同じ判定 {sum(1 for x in same if x)}/{len(same)}"))
    md += ["", "## 4. QNN の計測（NPU の中）", ""]
    for n in LAYERS:
        md += [f"### 層 {n}", "", *qnn_chunks(IO / f"qnnprof_L{n}.csv", n), "", "ノードごと（サイクルの合計が多い順）:", "", *qnn_summary(IO / f"qnnprof_L{n}.csv"), ""]
        pr = load(IO / f"qnnprof_L{n}.jsonl")
        if pr:
            md.append(f"- 計測を付けたときの NPU（ORT の Run）中央値 {q([float((r.get('timing') or {}).get('run_ort', 0)) for r in pr], .5):.1f}ms（{len(pr)} 行）")
            md.append("")
    (RES / "latency.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    summ = {f"L{n}_{m}": {name: q([p[name] for p in P[(n, m)]], .5) for name in P[(n, m)][0]} | {"wall": q([r["wall_ms"] for r in data[(n, m)]], .5), "rows": len(data[(n, m)])}
            for n, m in keys}
    (RES / "latency.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md[:8 + len(GROUPS) + 6]), flush=True)
    print(f"\n書きました: {RES / 'latency.md'}・latency.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "deploy", "bench", "report"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--modes", default="par,serial")
    ap.add_argument("--qnnprof", default="detailed", help="off・basic・detailed")
    ap.add_argument("--prof-rows", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    if a.stage == "prep":
        prep()
    elif a.stage == "deploy":
        deploy()
    elif a.stage == "bench":
        bench(a.layers, a.rows, a.modes, a.qnnprof, a.prof_rows, a.timeout)
    else:
        report()


if __name__ == "__main__":
    main()
