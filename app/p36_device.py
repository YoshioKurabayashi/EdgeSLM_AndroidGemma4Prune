# -*- coding: utf-8 -*-
r"""P36: 判定デモアプリ v2（Gemma4 判定デモ v2・`org.edgeslm.gemma4router2`、`android_app_router2/`）へ資材を送り、端末の判定を PC の判定と突き合わせる。

設計: DoC/設計書_P36_判定デモアプリv2.md の 7 節。`app/p24_device.py` を写して、アプリ・資材・確かめ用の行（評価セット v10）・数え方（P33 の改定後）を替えたもの。端末を USB で接続して、利用者が実行する。

|段|何をするか|
|---|---|
|`deploy`|v2 の置き場へ、表・Gemma 4 のトークナイザ・層 17 と 26 のグラフ・`work/p36/device/router/`（ヘッド・設定・確かめ用の行）・BERT と辞書（P24 で作った `work/p24/device/router/bert/`）を送る（md5 を確かめる）。端末のお試しアプリ（gemma4prune）・判定デモ（gemma4router）の置き場に同じものがあれば、端末の中で写す|
|`open`|ほかの Gemma のアプリを止めて、v2 の画面を開く（`--prompts "発話1\|発話2"` を付けると、順に判定する）|
|`check`|確かめ用の行（評価セット v10 の 600 行）を、端末の中で層 17・26 の順に全部判定させ、PC の判定（`p36_decide`）と突き合わせる → `results/P36/device_check.md`|

**既存のアプリとその置き場には書き込まない**（gemma4prune・gemma4router の置き場は、写す元として読むだけ）。

    python -m app.p36_device deploy
    python -m app.p36_device open
    python -m app.p36_device check
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import PROJECT_ROOT
from .p2_device import ADB, GRAPHS, OTHERS, TABLES, TOKENIZER, TOKENIZER_FILES, ctx_of
from .p7_data import read_jsonl
from .p33_check import grade_new as grade
from .p34_data import CH10

PKG = "org.edgeslm.gemma4router2"
MAIN = f"{PKG}/org.edgeslm.demo.RouterActivity"
TAG = "EdgeSLMRouter"
DEV = f"/sdcard/Android/data/{PKG}/files"
PRUNE_DEV = "/sdcard/Android/data/org.edgeslm.gemma4prune/files"      # 読むだけ（同じグラフがあれば、端末の中で写す）
ROUTER1_DEV = "/sdcard/Android/data/org.edgeslm.gemma4router/files"   # 読むだけ（P24 の判定デモ。同じ BERT・辞書があれば、端末の中で写す）
STOP = [*OTHERS, "org.edgeslm.gemma4prune", "org.edgeslm.gemma4router"]  # 同時にモデルを開くと RAM が足りない
BERT_SRC = PROJECT_ROOT / "work" / "p24" / "device" / "router" / "bert"   # P24 で作った BERT（ONNX）と辞書（読むだけ）
WORK = PROJECT_ROOT / "work" / "p36"
OUT = WORK / "device" / "router"
IO = WORK / "device_io"
RES = PROJECT_ROOT / "results" / "P36"
LAYERS = (17, 26)
WRITE = {"control_hvac", "control_media", "set_navigation"}


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


def push(src: Path, rel: str, copy_from: str | None = None) -> None:
    want = md5(src)
    if dev_md5(f"{DEV}/{rel}") == want:
        print(f"  skip {rel}（端末に同じものがある）", flush=True)
        return
    if copy_from and dev_md5(copy_from) == want:
        adb("shell", f"cp {copy_from} {DEV}/{rel}", timeout=1800, check=False)
        if dev_md5(f"{DEV}/{rel}") == want:
            print(f"  copy {rel}（{copy_from.split('/')[4]} の置き場にある同じものを、端末の中で写した・{src.stat().st_size / 2**20:.0f}MiB）md5 一致", flush=True)
            return
    adb("push", str(src), f"{DEV}/{rel}", timeout=1800)
    if dev_md5(f"{DEV}/{rel}") != want:
        raise SystemExit(f"md5 が合わない: {rel}")
    print(f"  push {rel}（{src.stat().st_size / 2**20:.0f}MiB）md5 一致", flush=True)


def ensure_app() -> None:
    if PKG not in adb("shell", f"pm list packages {PKG}"):
        raise SystemExit(f"端末に {PKG} が入っていない。先に .\\scripts\\run_p36.ps1 -Stage build")
    for p in STOP:
        adb("shell", f"am force-stop {p}", check=False)


def alive() -> bool:
    return bool(adb("shell", f"pidof {PKG}", check=False).strip())


# ---------------------------------------------------------------- deploy・open

def deploy() -> None:
    model = BERT_SRC / "model.onnx"
    if not (OUT / "router.json").exists() or not model.exists():
        raise SystemExit(f"資材が無い（{OUT}・{BERT_SRC}）。先に .\\scripts\\run_p36.ps1 -Stage prep（BERT は P24 の prep で作ったもの）")
    ensure_app()
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", f"am force-stop {PKG}")
    adb("shell", f"am start -n {MAIN}")           # フォルダはアプリに作らせる（adb で作ると持ち主が shell になり、アプリから読めない）
    time.sleep(3.0)
    adb("shell", f"am force-stop {PKG}")
    if "router" not in adb("shell", f"ls {DEV}/", check=False):
        raise SystemExit(f"アプリが置き場（{DEV}/router）を作っていない。画面が開けたか確認")
    print(f"== {PKG} の置き場へ（PC の原本は読むだけ。既存のアプリの置き場には書き込まない）", flush=True)
    t = json.loads((TABLES / "tables.json").read_text(encoding="utf-8"))
    for f in [t["embed"]["file"], t["embed"]["scale_file"], t["ple"]["file"], t["ple"]["scale_file"], *t["rope"]["files"].values()]:
        push(TABLES / f, f"models/ops/{f}", f"{PRUNE_DEV}/models/ops/{f}")
    for f in TOKENIZER_FILES:
        push(TOKENIZER / f, f"tokenizer/{f}", f"{PRUNE_DEV}/tokenizer/{f}")
    for n in LAYERS:
        e, s = GRAPHS[n]
        b = ctx_of(e, s)
        push(b, f"models/ops/{b.name}", f"{PRUNE_DEV}/models/ops/{b.name}")
        push(e, f"models/ops/{e.name}", f"{PRUNE_DEV}/models/ops/{e.name}")
    for p in sorted(BERT_SRC.iterdir()):
        if p.is_file():
            push(p, f"router/bert/{p.name}", f"{ROUTER1_DEV}/router/bert/{p.name}")
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            push(p, "router/" + str(p.relative_to(OUT)).replace("\\", "/"))
    print("送り終わりました", flush=True)


def open_app(prompts: str | None) -> None:
    ensure_app()
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", f"am force-stop {PKG}")
    time.sleep(2.0)
    adb("shell", f"am start -n {MAIN}" + (f" --es prompts {shlex.quote(prompts)}" if prompts else ""))
    print(f"画面を開きました（{PKG}）。グラフと BERT の読み込みに 10 秒ほどかかります", flush=True)


# ---------------------------------------------------------------- check

def run_eval(layers: str, timeout_s: int) -> None:
    ensure_app()
    IO.mkdir(parents=True, exist_ok=True)
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", f"am force-stop {PKG}")
    time.sleep(8.0)
    adb("logcat", "-c", check=False)
    adb("shell", f"am start -n {MAIN} --es eval {shlex.quote(layers)}")
    t0, last = time.time(), ""
    while True:
        time.sleep(10)
        log = adb("logcat", "-d", "-s", f"{TAG}:*", check=False)
        lines = [x for x in log.splitlines() if "router_" in x or "ready" in x]
        if lines and lines[-1] != last:
            last = lines[-1]
            print("  " + last.split(f"{TAG}:")[-1].strip()[:160], flush=True)
        if "router_eval done" in log:
            break
        if "router_fatal" in log:           # 画面が、準備か確かめの途中で止まった（資材が無い・グラフが開けない など）
            raise SystemExit("端末の確かめが失敗した（logcat の EdgeSLMRouter）:\n" + "\n".join(log.splitlines()[-15:]))
        if time.time() - t0 > 40 and not alive():
            raise SystemExit("アプリが結果を出さずに終わった（logcat を確認）:\n" + adb("logcat", "-d", "-t", "60", check=False)[-3000:])
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"{timeout_s} 秒たっても終わらない")
    for n in [x.strip() for x in layers.split(",")]:
        adb("pull", f"{DEV}/out/router_eval_L{n}.jsonl", str(IO / f"router_eval_L{n}.jsonl"))


def med(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def compare(layers: str) -> None:
    rows = read_jsonl(CH10 / "utterances.jsonl")
    byid = {r["id"]: r for r in rows}
    pcs = {json.loads(x)["id"]: json.loads(x) for x in (OUT / "eval_v10.jsonl").read_text(encoding="utf-8").split("\n") if x}
    truth = {}
    tt = WORK / "tok_truth.tsv"
    if tt.exists():
        for line in tt.read_text(encoding="utf-8").split("\n"):
            c = line.split("\t")
            if len(c) >= 2 and c[1]:
                truth[c[0]] = [int(x) for x in c[1].split(" ")]
    RES.mkdir(parents=True, exist_ok=True)
    md = ["判定デモアプリ v2（端末の中だけで判定）と、PC の判定（`p36_decide`。評価の流れと同じ・P35 の構成）の突き合わせ。行は評価セット v10 の 600 行。"
          "PC は「端末で集めた h（P34）→ ヘッド（1,656 行・S1 の seed）」「PC の BERT」、端末は「その場で NPU に流した h → ヘッド」「端末の BERT（ONNX・CPU）」\n",
          "## 判定が同じか\n", "|層|行|端末で失敗した行|判定（結果・止めた段・機能・引数）が PC と同じ|同じ割合|Gemma の入力が PC と同じ|BERT の入力が PC と同じ|BERT の確率の差の最大|ヘッドの確信度の差の最大|棄却の得点の差の最大|",
          "|---|---|---|---|---|---|---|---|---|---|"]
    kpi_md = ["\n## KPI（載せる seed 1 つ・評価セット v10・改定後の数え方。値は見ていない KPI。値まで見た KPI は results/P34・P35）\n",
              "|層|どこの判定か|EtoE の成功|重大な誤り（件）|不要な聞き返し|通常の誤り|1 回目で正しく処理|", "|---|---|---|---|---|---|---|"]
    time_md = ["\n## 1 回の判定にかかる時間（中央値・ms）\n",
               "|層|合計（待ち時間）|NPU|入力の準備|ヘッド|BERT|BERT の入力作り|Gemma の入力作り|規則・引数・検査|", "|---|---|---|---|---|---|---|---|---|"]
    diff_md, info = [], {}
    for n in [x.strip() for x in layers.split(",")]:
        f = IO / f"router_eval_L{n}.jsonl"
        got = [json.loads(x) for x in f.read_text(encoding="utf-8").split("\n") if x]
        err = [g for g in got if g.get("error")]
        ok = [g for g in got if not g.get("error")]
        same = [g for g in ok if g.get("pc_same")]
        gem = sum(1 for g in ok if g.get("gemma_ids_match"))
        btok = sum(1 for g in ok if truth.get(g["utterance"]) == g.get("bert_ids")) if truth else None
        d_b = max((abs(g["bert"] - pcs[g["id"]]["pc"][n]["bert"]) for g in ok), default=float("nan"))
        with_head = [g for g in ok if "conf" in g]
        d_c = max((abs(g["conf"] - pcs[g["id"]]["pc"][n]["conf"]) for g in with_head), default=float("nan"))
        d_r = max((abs(g["reject"] - pcs[g["id"]]["pc"][n]["reject"]) for g in with_head), default=float("nan"))
        md.append(f"|{n}|{len(got)}|{len(err)}|{len(same)}|{len(same) / max(len(got), 1):.3f}|{gem}|{'—' if btok is None else btok}|{d_b:.2e}|{d_c:.2e}|{d_r:.2e}|")
        print(f"層 {n}: {len(got)} 行・失敗 {len(err)}・判定が PC と同じ {len(same)}（{len(same) / max(len(got), 1):.3f}）・Gemma の入力が同じ {gem}・BERT の入力が同じ {btok}・"
              f"差の最大 BERT {d_b:.1e}・確信度 {d_c:.1e}・棄却 {d_r:.1e}", flush=True)

        def kpis(outs: list[tuple]) -> dict:
            g = [grade(byid[i], o, t if o != "雑談" else None, WRITE, True) for i, o, t in outs]
            known = [byid[i]["action"] != "UNKNOWN" for i, _, _ in outs]
            return {"e2e": float(np.mean([x in ("正解", "回復可能") for x in g])), "severe_n": int(sum(x == "重大" for x in g)), "normal": float(np.mean([x == "通常" for x in g])),
                    "first": float(np.mean([x == "正解" for x in g])), "over": float(np.mean([o == "聞き返し" for (_, o, _), k in zip(outs, known) if k]))}

        k_dev = kpis([(g["id"], g["outcome"], g.get("tool")) for g in ok]) if ok else {}
        k_pc = kpis([(i, p["pc"][n]["outcome"], p["pc"][n]["tool"]) for i, p in pcs.items()])
        for name, k in (("端末（アプリ）", k_dev), ("PC（評価の流れ）", k_pc)):
            if k:
                kpi_md.append(f"|{n}|{name}|{k['e2e']:.3f}|{k['severe_n']}|{k['over']:.3f}|{k['normal']:.3f}|{k['first']:.3f}|")
                print(f"  {name}: EtoE {k['e2e']:.3f}・重大 {k['severe_n']} 件・不要な聞き返し {k['over']:.3f}・通常 {k['normal']:.3f}・1 回目 {k['first']:.3f}", flush=True)
        t = {k: med([g.get(k) for g in ok]) for k in ("wall_ms", "npu_run_ms", "npu_prep_ms", "head_ms", "bert_ms", "bert_tok_ms", "gemma_tok_ms", "rule_ms")}
        time_md.append(f"|{n}|{t['wall_ms']:.0f}|{t['npu_run_ms']:.0f}|{t['npu_prep_ms']:.0f}|{t['head_ms']:.1f}|{t['bert_ms']:.0f}|{t['bert_tok_ms']:.1f}|{t['gemma_tok_ms']:.1f}|{t['rule_ms']:.1f}|")
        print(f"  時間（中央値 ms）: 合計 {t['wall_ms']:.0f}・NPU {t['npu_run_ms']:.0f}・準備 {t['npu_prep_ms']:.0f}・ヘッド {t['head_ms']:.1f}・BERT {t['bert_ms']:.0f}・"
              f"BERT の入力 {t['bert_tok_ms']:.1f}・Gemma の入力 {t['gemma_tok_ms']:.1f}・規則ほか {t['rule_ms']:.1f}", flush=True)
        bad = [g for g in ok if not g.get("pc_same")]
        stages = collections.Counter(f"{pcs[g['id']]['pc'][n]['outcome']}（{pcs[g['id']]['pc'][n]['stage'] or '—'}）→ {g['outcome']}（{g.get('stage') or '—'}）" for g in bad)
        diff_md += [f"\n## 層 {n}: PC と判定が違った行（{len(bad)}）\n"]
        if bad:
            diff_md += ["|発話|PC|端末|確信度 PC→端末|棄却 PC→端末|BERT PC→端末|", "|---|---|---|---|---|---|"]
            for g in bad[:40]:
                p = pcs[g["id"]]["pc"][n]
                diff_md.append(f"|{g['utterance']}|{p['outcome']}（{p['stage'] or '—'}）{p['tool'] or ''} {p['args'] or ''}|{g['outcome']}（{g.get('stage') or '—'}）{g.get('tool') or ''} {g.get('args') or ''}|"
                               f"{p['conf']:.3f}→{g.get('conf', float('nan')):.3f}|{p['reject']:.4f}→{g.get('reject', float('nan')):.4f}|{p['bert']:.4f}→{g['bert']:.4f}|")
            diff_md.append("\n変わり方: " + "・".join(f"{k} {v} 行" for k, v in stages.most_common()))
        for g in err[:5]:
            diff_md.append(f"\n失敗: {g['utterance']} … {g['error']}")
        info[n] = {"rows": len(got), "errors": len(err), "same": len(same), "gemma_ids_same": gem, "bert_ids_same": btok, "bert_absdiff_max": d_b, "conf_absdiff_max": d_c,
                   "reject_absdiff_max": d_r, "kpi_device": k_dev, "kpi_pc": k_pc, "time_ms_median": t, "changes": dict(stages)}
    (RES / "device_check.md").write_text("\n".join(md + kpi_md + time_md + diff_md) + "\n", encoding="utf-8")
    (RES / "device_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存しました: {RES / 'device_check.md'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["deploy", "open", "check", "compare"])
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--prompts", default="")
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()
    print(f"\n===== P36 {a.step} =====", flush=True)
    if a.step == "deploy":
        deploy()
    elif a.step == "open":
        open_app(a.prompts or None)
    elif a.step == "check":
        run_eval(a.layers, a.timeout)
        compare(a.layers)
    else:
        compare(a.layers)        # 端末を使わず、取ってある結果を比べ直す


if __name__ == "__main__":
    main()
