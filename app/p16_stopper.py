# -*- coding: utf-8 -*-
"""P16 追加: 「機能にない依頼を止める力」を、LM ヘッドと BERT で同じ条件にそろえて比べる。学習なし・端末なし（numpy だけ）。

設計: DoC/設計書_P16_部品ごとの分類精度.md の 7 節。

P16 の表では、ヘッドは棄却の閾値（validation の 2%）、BERT はフラグ（0.5）で数えていて、止める手間がそろっていない。
ここでは **対象の依頼と雑談を誤って止める割合を同じ（2%・5%）にそろえたとき**、機能にない依頼をどれだけ止められるかを出す。
あわせて、学習データの量の効き目を見る:

|部品|学習に使った機能にない依頼|
|---|---|
|ヘッド 933 行（P5）|60 行（20 機能）|
|ヘッド 1,101 行（P7）|180 行（40 機能）|
|BERT 案 C（P11）|144 行（40 機能。train・validation の分）|
|BERT 案 E（P13）|252 行（76 機能）|

閾値は評価セット v3 の対象の行から決める（そろえて比べるための値。実際の運用の閾値ではない）。

    python -m app.p16_stopper        # results/P16/stopper.md・stopper.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import PROJECT_ROOT
from .p6_seeds import SEEDS, auc, head_probs, load_pt
from .p7_block import HEADS
from .p7_data import read_jsonl
from .p13_data import CH3
from .p13_eval import WORK as P13, load_eval3

RES = PROJECT_ROOT / "results" / "P16"
FPRS = (0.02, 0.05)


def ms(xs, nd=3) -> str:
    xs = [float(x) for x in xs]
    return f"{np.mean(xs):.{nd}f}" if min(xs) == max(xs) else f"{np.mean(xs):.{nd}f}（{min(xs):.{nd}f}〜{max(xs):.{nd}f}）"


def at_fpr(score, known, sel, fpr: float) -> float:
    tau = float(np.quantile(score[known], 1.0 - fpr, method="higher"))
    return float(np.mean(score[sel] > tau))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    ch3 = read_jsonl(CH3 / "utterances.jsonl")
    ids = [r["id"] for r in ch3]
    cat = np.array([r["category"] for r in ch3])
    grp = np.array([r.get("os_group") or "" for r in ch3])
    known = np.array([r["action"] != "UNKNOWN" for r in ch3])
    isH, isG = cat == "H", cat == "G"
    groups = [("機能にない依頼・180 行", isH), ("　学習に同じ機能がある・60 行", isH & (grp == "seen")), ("　案 E で足した 20 機能・40 行", isH & (grp == "new_seen")),
              ("　どの学習にも無い機能・80 行", isH & (grp == "unseen")), ("意図が不明・140 行", isG)]
    parts = {}
    for n in ns:
        loaded = load_eval3(n)
        if loaded is None:
            continue
        _, last, mean = loaded
        for src in HEADS:
            d = HEADS[src](n)
            if not (d / "evaluate_report.json").exists():
                continue
            reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
            sc = []
            for seed in SEEDS:
                name = f"heads_mlp_seed{seed}.pt"
                p = head_probs(load_pt(d / name), float(reps[name]["temperature"]), last, mean)
                sc.append(p["reject"])
            parts[f"ヘッド {src.split('（')[0]}・層 {n}"] = sc
    for key, lab in (("C", "BERT 案 C"), ("E", "BERT 案 E")):
        f = P13 / f"text_scores_{key}.json"
        if f.exists():
            ts = json.loads(f.read_text(encoding="utf-8"))
            parts[lab] = [np.array([ts["models"][m]["pred"][i] for i in ids]) for m in sorted(ts["models"]) if m.startswith("bert_s")]
    print("\n===== P16 追加: 機能にない依頼を止める力（対象を誤って止める割合をそろえて比べる）=====")
    md = ["評価セット v3（800 行）。対象の依頼と雑談 480 行を誤って止める割合を 2%・5% にそろえたとき、止められた割合。数字は seed 3 つの平均（かっこは最小〜最大）。"
          "ヘッドは棄却の得点だけ（確信度の閾値は入れていない）\n", "|部品|AUC（機能にない依頼 ↔ 対象）|" +
          "|".join(f"{lab}（誤って止める {int(f * 100)}%）" for lab, _ in groups for f in FPRS) + "|", "|---|---|" + "---|" * (len(groups) * len(FPRS))]
    res = {}
    for name, scs in parts.items():
        aucs = [auc(s[isH], s[known]) for s in scs]
        cells, out = [], {"auc": aucs}
        for lab, sel in groups:
            for f in FPRS:
                v = [at_fpr(s, known, sel, f) for s in scs]
                out[f"{lab.strip()}|{f}"] = v
                cells.append(ms(v))
        res[name] = out
        md.append(f"|{name}|{ms(aucs)}|" + "|".join(cells) + "|")
        print(f"{name:<22} AUC {ms(aucs)} | 2%: 全体 {cells[0]}・seen {cells[2]}・足した {cells[4]}・unseen {cells[6]}・不明 {cells[8]} | 5%: 全体 {cells[1]}・unseen {cells[7]}・不明 {cells[9]}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "stopper.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "stopper.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（stopper.md・stopper.json）")


if __name__ == "__main__":
    main()
