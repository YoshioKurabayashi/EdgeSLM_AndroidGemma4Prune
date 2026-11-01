# -*- coding: utf-8 -*-
"""P36: 判定デモアプリ v2（`org.edgeslm.gemma4router2`・`android_app_router2/`）に載せる資材を作る。

設計: DoC/設計書_P36_判定デモアプリv2.md の 6 節。`app/p24_assets.py` の heads・truth を写して、ヘッド（1,656 行・S1 の seed）・確かめ用の行（評価セット v10）に替えたもの。

|段|中身|venv|
|---|---|---|
|`heads`|ヘッド（1,656 行・mlp・層 17 = seed 17・層 26 = seed 29。棄却の閾値は validation の 2%）を端末の形に書き出す・画面の設定（router.json）・確かめ用の行（eval_v10.jsonl）|どちらでも（numpy だけ。VM の python3 でも動く）|
|`truth`|確かめ用の行の、PC の BertJapaneseTokenizer の ID 列（端末の BERT の入力が同じかを見る）|EdgeSLM_Android（transformers）|

BERT（ONNX・辞書）は P24 で作ったもの（`work/p24/device/router/bert/`）をそのまま使う（`p36_device deploy` が送る。端末に P24 のアプリの同じものがあれば、端末の中で写す）。

    python -m app.p36_assets heads      # work/p36/device/router/・work/p36/fixtures/h_L<n>.bin
    python -m app.p36_assets truth      # work/p36/tok_truth.tsv
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT
from .g3_headlib import LABELS_JA, numpy_head
from .p6_seeds import head_probs, load_pt
from .p7_data import read_jsonl
from .p24_assets import INFO, MAX_LENGTH, TOOL_JA, md5, pick_samples
from .p28_data import TR4
from .p28_verdict import SRC1656, hook1656
from .p7_block import HEADS
from .p31_check import pick_seed
from .p34_data import CH10
from .p34_verdict import WORK as P34W, load_eval10

WORK = PROJECT_ROOT / "work" / "p36"
FIX = WORK / "fixtures"
OUT = WORK / "device" / "router"
RES = PROJECT_ROOT / "results" / "P36"
BERT_DIR = PROJECT_ROOT / "work" / "p20" / "bert_F" / "bert_s0"
TRUTH = WORK / "tok_truth.tsv"
LAYERS = (17, 26)


def read_flow(n: int) -> tuple[list[str], dict, list[dict]]:
    """`p36_fixtures` が作った flow_L<n>.tsv（評価セット v10 の行）"""
    lines = (FIX / f"flow_L{n}.tsv").read_text(encoding="utf-8").split("\n")
    classes = lines[0].split("\t")[1:]
    th = {k: float(v) for k, v in (x.split("=") for x in lines[1].split("\t")[1:])}
    rows = []
    for line in lines[2:]:
        if not line:
            continue
        c = line.split("\t")
        rows.append({"utterance": c[0], "probs": [float(x) for x in c[1].split(" ")], "reject": float(c[2]), "bert": float(c[3]),
                     "outcome": c[4], "stage": None if c[5] == "-" else c[5], "tool": None if c[6] == "-" else c[6], "args": None if c[7] == "-" else c[7], "rule": None if c[8] == "-" else c[8]})
    return classes, th, rows


def export_head(n: int, seed: int) -> dict:
    hd = HEADS[SRC1656](n)
    name = f"heads_mlp_seed{seed}.pt"
    blob = load_pt(hd / name)
    rep = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}[name]
    classes = list(blob["metadata"]["action_classes"])
    th = hook1656(n, SRC1656, [(seed, classes, {k: float(v) for k, v in rep["thresholds"].items()}, None)])[0][2]
    state = {k: np.ascontiguousarray(np.asarray(v), dtype="<f4") for k, v in blob["state"].items() if k != "temperature"}
    tensors, chunks, off = {}, [], 0
    for key in sorted(state):
        tensors[key] = {"offset": off, "shape": list(state[key].shape)}
        chunks.append(state[key].ravel())
        off += state[key].size
    flat = np.concatenate(chunks).astype("<f4")
    tools = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    meta = {"hidden": int(state["standardizer.mean"].shape[0]), "reject_hidden": int(state["reject_standardizer.mean"].shape[0]),
            "width": int(state["heads.action.0.weight"].shape[0]), "kind": blob["config"]["kind"], "temperature": float(rep["temperature"]),
            "action_classes": classes, "argmode_classes": list(blob["metadata"].get("argmode_classes", [])), "thresholds": {k: float(v) for k, v in th.items()},
            "tool_effect": tools, "argmode_tools": ["control_hvac", "control_media"], "labels_ja": LABELS_JA, "tensors": tensors, "total_floats": int(flat.size),
            "source": {"checkpoint": str(hd / name), "variant": f"router2_L{n}", "rows": "1,656 行（P28）", "seed": seed, "seed_rule": "S1（validation でヘッドが通す割合が一番大きい seed・P31）",
                       "reject_threshold": "validation の 2%"}}
    # ---- 確かめ: 端末と同じ式（numpy_head）↔ 評価で使った式（head_probs）↔ p36_fixtures の表
    rows, last, mean = load_eval10(n)
    mine = numpy_head(meta, flat, last, mean)
    ref = head_probs(blob, float(rep["temperature"]), last, mean)
    fcls, fth, frows = read_flow(n)
    if fcls != classes or len(frows) != len(rows) or any(a["utterance"] != b["utterance"] for a, b in zip(rows, frows)):
        raise SystemExit(f"層 {n}: flow_L{n}.tsv の並びが評価セットと違う（先に python -m app.p36_fixtures）")
    d1 = float(np.abs(mine["action"] - ref["action"]).max())
    d2 = float(np.abs(mine["reject"] - ref["reject"]).max())
    d3 = float(np.abs(mine["action"] - np.array([r["probs"] for r in frows])).max())
    d4 = float(np.abs(mine["reject"] - np.array([r["reject"] for r in frows])).max())
    dth = max(abs(fth[k] - meta["thresholds"][k]) for k in ("CHAT", "TOOL_read", "TOOL_write", "REJECT"))
    print(f"層 {n}・seed {seed}: 温度 {meta['temperature']:.2f}・閾値 {meta['thresholds']} / 式の差 確率 {d1:.1e}・棄却 {d2:.1e} / 表との差 確率 {d3:.1e}・棄却 {d4:.1e}・閾値 {dth:.1e}", flush=True)
    if max(d1, d2, d3, d4, dth) > 1e-9:
        raise SystemExit(f"層 {n}: 書き出したヘッドが、評価で使ったものと合わない")
    meta["self_check"] = {"rows": len(rows), "action_prob_absdiff_max": d3, "reject_absdiff_max": d4}
    out = OUT / f"L{n}"
    out.mkdir(parents=True, exist_ok=True)
    flat.tofile(out / "head.bin")
    (out / "head.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tsv = [f"hidden\t{meta['hidden']}", f"reject_hidden\t{meta['reject_hidden']}", f"temperature\t{meta['temperature']!r}", "classes\t" + "\t".join(classes)]
    tsv += [f"threshold\t{k}\t{meta['thresholds'][k]!r}" for k in ("CHAT", "TOOL_read", "TOOL_write", "REJECT")]
    tsv += [f"tensor\t{k}\t{v['offset']}\t" + "\t".join(str(x) for x in v["shape"]) for k, v in tensors.items()]
    tsv.append(f"total_floats\t{flat.size}")
    (out / "head.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8", newline="\n")
    np.hstack([last, mean]).astype("<f4").tofile(FIX / f"h_L{n}.bin")
    return meta


def heads() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    seeds = {n: pick_seed(n)[0] for n in LAYERS}
    metas = {n: export_head(n, seeds[n]) for n in LAYERS}
    rows = read_jsonl(CH10 / "utterances.jsonl")
    flows = {n: read_flow(n)[2] for n in LAYERS}
    bert_hi = read_flow(LAYERS[0])[1]["BERT_HI"]
    toks = json.loads((P34W / "token_ids_eval.json").read_text(encoding="utf-8"))
    p3 = json.loads((PROJECT_ROOT / "work" / "p3" / "L17" / "head" / "samples.json").read_text(encoding="utf-8"))
    job = {k: v for k, v in p3["job"].items() if k != "model"}
    with open(OUT / "eval_v10.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            if flows[17][i]["utterance"] != r["utterance"]:
                raise SystemExit("flow の並びが評価セット v10 と違う")
            pc = {str(n): {k: flows[n][i][k] for k in ("outcome", "stage", "tool", "args", "rule", "bert", "reject")} | {"conf": max(flows[n][i]["probs"])} for n in LAYERS}
            f.write(json.dumps({"id": r["id"], "utterance": r["utterance"], "motion": r["state"]["motion"], "gemma_ids": toks[r["id"]], "pc": pc}, ensure_ascii=False) + "\n")
    samples = pick_samples(rows, flows)
    byid = {r["id"]: i for i, r in enumerate(rows)}
    for s in samples:
        i = byid[s["id"]]
        s["pc"] = {str(n): {k: flows[n][i][k] for k in ("outcome", "stage", "tool", "args")} for n in LAYERS}
    cfg = {"title": "車載ルーター 判定デモ v2（スマホの中だけで判定）",
           "subtitle": "発話を、層を減らした Gemma 4 E2B（NPU）＋分類ヘッド・日本語 BERT・規則・引数の取り出し・機能の定義での絞り込み・検査に通し、実行／聞き返す／断る／雑談 を決めます。"
                       "操作は模擬です（実際の車は動かしません）。通信は使いません。",
           "note": "評価の流れ（P35 の構成）と同じ順番・同じ部品です。ヘッドは 1,656 行で学習したもの（層 17 = seed " + str(seeds[17]) + "・層 26 = seed " + str(seeds[26]) +
                   "。validation で選んだ seed）、BERT は案 F（seed 0）。機能にない依頼・機能が返さないものを聞く依頼は「その機能はありません」と断ります。聞き返した後の 2 回目のやり取りは入っていません。",
           "chat_prefix": p3["chat_prefix"], "chat_suffix": p3["chat_suffix"], "job": job, "default_layer": 26,
           "layers": [{"n": n, "label": INFO[n]["label"], "model": f"models/ops/p2_L{n}_epctx.onnx", "head": f"router/L{n}", "context_mib": INFO[n]["ctx"],
                       "detail": f"Gemma 4 E2B の 35 層のうち前半 {n} 層（{INFO[n]['pct']}）・グラフ {INFO[n]['ctx']}MiB・ヘッド seed {seeds[n]}"} for n in LAYERS],
           "bert": {"dir": "router/bert", "model": "router/bert/model.onnx", "max_length": MAX_LENGTH, "hi": bert_hi, "lo": 0.5, "threads": 4,
                    "name": "cl-tohoku/bert-base-japanese-v3 を学習し直したもの（案 F・seed 0）"},
           "labels_ja": LABELS_JA, "tool_ja": TOOL_JA, "samples": samples, "eval": "router/eval_v10.jsonl", "version": "v2（P36・2026-10-14）"}
    (OUT / "router.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = {str(p.relative_to(OUT)).replace("\\", "/"): {"bytes": p.stat().st_size, "md5": md5(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"例文 {len(samples)} 文: " + "／".join(s["utterance"] for s in samples))
    print(f"確かめ用の行 {len(rows)}（評価セット v10）/ BERT が断る閾値 {bert_hi:.4f}")
    print(f"保存しました: {OUT}（{len(manifest)} ファイル）")
    (RES / "assets.json").write_text(json.dumps({"heads": {str(n): {"seed": seeds[n], **{k: metas[n][k] for k in ("temperature", "thresholds", "self_check")}} for n in LAYERS},
                                                 "samples": samples, "bert_hi": bert_hi}, ensure_ascii=False, indent=1), encoding="utf-8")


def truth() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
    us = [r["utterance"] for r in read_jsonl(CH10 / "utterances.jsonl")]
    WORK.mkdir(parents=True, exist_ok=True)
    with open(TRUTH, "w", encoding="utf-8", newline="\n") as f:
        for u in us:
            ids = tok(u, truncation=True, max_length=MAX_LENGTH)["input_ids"]
            f.write(u + "\t" + " ".join(str(i) for i in ids) + "\n")
    print(f"PC の BertJapaneseTokenizer の ID 列: {len(us)} 行 → {TRUTH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["heads", "truth"])
    a = ap.parse_args()
    print(f"\n===== P36 assets {a.step} =====", flush=True)
    {"heads": heads, "truth": truth}[a.step]()


if __name__ == "__main__":
    main()
