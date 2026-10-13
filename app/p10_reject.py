# -*- coding: utf-8 -*-
"""P10: 棄却の閾値の決め方を変えて、不要な聞き返しと重大な誤りの釣り合いを見る。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P10_棄却の閾値の決め方.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

いまの決め方（EdgeSLM_MultiHead_Multi の `evaluate.choose_thresholds`）は、4 つの閾値の総当たりで合格基準からの外れが最小の組を選ぶ。
選ばれる棄却の閾値は seed ごとに 0.21〜0.99 と割れ（P6）、P9 の時点で不要な聞き返しの最大の原因になっていた。

ここでは、**棄却の閾値だけ**を次の決め方に差し替える（確信度の閾値は各ヘッドが validation で決めた値のまま）。どれも評価セットを見ない。

|決め方|中身|
|---|---|
|いまの決め方|総当たりで選ばれた値（比べる基準）|
|validation の x%|validation の行き先のある行のうち、x% だけを誤って確認へ回す値（棄却の得点の分位点）。x = 1・2・3・5|
|validation＋test の x%|分位点を決める行を、validation（156 行）から validation＋test（303 行）に増やす。x = 2・3|
|3 つの平均＋validation の x%|seed 3 つのヘッドの確率を平均して 1 つのヘッドとして使い、その得点で分位点を決める。x = 2・3|

**あらかじめ決めた本命は「validation の 2%」**: 不要な聞き返しの目標 5% を、規則 1%・引数 1%・BERT のフラグ 1%・棄却ヘッド 2% と割り振った値。
ほかの決め方は、感度を見るために並べる（評価セットの結果を見て本命を選び直さない）。

    python -m app.p10_reject        # results/P10/reject_rule.md・reject_rule.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import PROJECT_ROOT
from .p5_retrain import WORK as P5
from .p6_seeds import SEEDS, head_probs, load_pt
from .p7_block import HEADS
from .p8_gate import STEPS, run
from .p9_arguments import make_resolve

RES = PROJECT_ROOT / "results" / "P10"
SRC = "933 行（P5）"
DATA = PROJECT_ROOT / "data" / "train_add_v1" / "utterances.jsonl"
PRIMARY = "validation の 2%"
COLS = [("EtoE の成功", "e2e", 3), ("重大な誤り（200 行中の件数）", "severe_n", 1), ("不要な聞き返し", "over", 3),
        ("意図が不明を聞き返した", "amb_ok", 3), ("機能にない依頼を操作として実行（件）", "os_severe_n", 1), ("1 回目で正しく処理", "first", 3)]


def known_rows(n: int, splits: tuple) -> tuple[np.ndarray, np.ndarray]:
    """933 行のうち、閾値を決めるのに使う行き先のある行の `h`（P5 の特徴）"""
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = P5 / f"L{n:02d}" / "features"
    ids = json.loads((d / "features_F2.meta.json").read_text(encoding="utf-8"))["ids"]
    if ids != [r["id"] for r in rows]:
        raise SystemExit(f"層 {n}: 特徴の並びがデータと違う")
    pick = [i for i, r in enumerate(rows) if r["split"] in splits and r["route"] != "UNKNOWN"]
    z = np.load(d / "features_F2.npz")
    return z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)


def quantile(scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(scores, 1.0 - fpr, method="higher"))


def make_hook(kind: str, fpr: float, log: dict):
    """`p8_gate.run` に渡す差し替え。kind: current / val / valtest / ensemble"""
    def hook(n: int, src: str, heads: list) -> list:
        if kind == "current":
            log[n] = [h[2]["REJECT"] for h in heads]
            return heads
        d = HEADS[src](n)
        reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
        last, mean = known_rows(n, ("validation", "test") if kind == "valtest" else ("validation",))
        val = []
        for seed in SEEDS:
            name = f"heads_mlp_seed{seed}.pt"
            val.append(head_probs(load_pt(d / name), float(reps[name]["temperature"]), last, mean)["reject"])
        if kind == "ensemble":
            tau = quantile(np.mean(val, axis=0), fpr)
            th = {k: float(np.median([h[2][k] for h in heads])) for k in ("TOOL_write", "TOOL_read", "CHAT")}
            p = {"action": np.mean([h[3]["action"] for h in heads], axis=0), "reject": np.mean([h[3]["reject"] for h in heads], axis=0)}
            log[n] = [tau]
            return [("平均", heads[0][1], {**th, "REJECT": tau}, p)]
        taus = [quantile(v, fpr) for v in val]
        log[n] = taus
        return [(seed, classes, {**th, "REJECT": t}, p) for (seed, classes, th, p), t in zip(heads, taus)]
    return hook


RULES = [("いまの決め方（総当たり）", "current", 0.0)] + [(f"validation の {int(x * 100)}%", "val", x) for x in (0.01, 0.02, 0.03, 0.05)] + \
        [(f"validation＋test の {int(x * 100)}%", "valtest", x) for x in (0.02, 0.03)] + \
        [(f"3 つの平均＋validation の {int(x * 100)}%", "ensemble", x) for x in (0.02, 0.03)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    final = STEPS[-1]
    res = {}
    print(f"\n===== P10: 棄却の閾値の決め方（ヘッド {SRC}・引数は補った処理・{final[0]}）=====")
    print(f"あらかじめ決めた本命: {PRIMARY}")
    for name, kind, fpr in RULES:
        log = {}
        blob = run(ns, make_resolve=make_resolve, out=None, heads_hook=make_hook(kind, fpr, log), steps=[final], sources=[SRC], quiet=True)
        for n in ns:
            key = f"{n}|{SRC}|{final[0]}"
            if key in blob["kpi"]:
                res[f"{n}|{name}"] = {"taus": log.get(n, []), "kpi": blob["kpi"][key], "stages": blob["detail"][key]["stages"],
                                      "severe_rows": blob["detail"][key]["severe_rows"]}
    md = []
    for n in ns:
        print(f"\n########## 層 {n} 個 ##########")
        md += [f"\n### 層 {n} 個\n", "|棄却の閾値の決め方|決まった閾値|" + "|".join(c for c, _, _ in COLS) + "|", "|---|---|" + "---|" * len(COLS)]
        for name, _, _ in RULES:
            r = res.get(f"{n}|{name}")
            if not r:
                continue
            taus = "・".join(f"{t:.3f}" for t in r["taus"])
            cells = [f"{r['kpi'][k]['mean']:.{nd}f}（{r['kpi'][k]['min']:.{nd}f}〜{r['kpi'][k]['max']:.{nd}f}）" for _, k, nd in COLS]
            mark = "**" if name == PRIMARY else ""
            print(f"{name:<24} 閾値 {taus:<20} " + "  ".join(f"{c}: {x}" for (c, _, _), x in zip(COLS, cells)))
            md.append(f"|{mark}{name}{mark}|{taus}|" + "|".join(cells) + "|")
        r = res.get(f"{n}|{PRIMARY}")
        if r:
            known = {k.split("|")[1]: v for k, v in r["stages"].items() if k.startswith("対象の依頼・雑談") and "聞き返し" in k}
            line = "・".join(f"{k.replace('聞き返し（', '').replace('）', '')} {v:.1f}" for k, v in known.items())
            print(f"  本命のとき、対象の依頼・雑談を聞き返した段（1 回あたりの行数）: {line}")
            print("  本命のとき残る重大な誤り: " + " / ".join(f"[{x['category']}]{x['utterance']}→{x['executed']}（{x['share']:.2f}）" for x in r["severe_rows"][:10]))
            md += [f"\n本命（{PRIMARY}）のとき、対象の依頼・雑談を聞き返した段（1 回あたりの行数）: {line}\n",
                   "本命のとき残る重大な誤り:\n", "|種類|発話|実行したツール|出た割合|", "|---|---|---|---|"]
            md += [f"|{x['category']}|{x['utterance']}|{x['executed']}|{x['share']:.2f}|" for x in r["severe_rows"][:20]]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "reject_rule.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "reject_rule.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（reject_rule.md・reject_rule.json）")


if __name__ == "__main__":
    main()
