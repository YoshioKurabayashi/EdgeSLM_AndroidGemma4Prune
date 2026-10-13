# -*- coding: utf-8 -*-
"""P18 手順 3: 結果を見た後の分析（合否の後で足した。学習なし・端末なし。numpy と PyYAML だけ）。

設計: DoC/設計書_P18_BERTとヘッドを学習し直す.md の 10 節。

**ここの数字は、評価セット v4 の結果を見てから出したもの。合否ではない**（合否は `p18_retrain judge` の本命の数字）。
評価セット v4 は、ここで「直す用」になる。

|分析|見るもの|
|---|---|
|1 BERT の seed ごと|全体で学習した BERT が、対象の依頼にフラグを立てる割合。学習側の交差検証・1,101 行の test・評価セット v1〜v4 で、seed ごとに|
|2 BERT の seed を分けた KPI|本命の構成（ヘッド 1,584 行）で、BERT を seed ごと・seed の平均にしたときの KPI|
|3 棄却の閾値の決め方|validation の x%（x = 1・2・3・5・7・10）で、ヘッド 933 行と 1,584 行を比べる（BERT は案 F）|
|4 棄却の閾値（BERT は seed の平均）|ヘッド 1,584 行・BERT は案 F の seed 3 つの確率の平均で、同じ x を動かす|

    python -m app.p18_analyze        # results/P18/analyze.md・analyze.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .p6_seeds import head_probs, load_pt
from .p7_block import HEADS
from .p7_data import CH1, CH2, TR2, read_jsonl
from .p8_gate import BERT_LO, STEPS, run
from .p10_reject import SRC as SRC933, make_hook as hook933, quantile
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import cell, hi_rule, load_eval3
from .p17_arguments import make_resolve
from .p17_rules import make_rules
from .p18_data import CH4, TR3
from .p18_retrain import RES, SRC1584, WORK, cached, has_heads, load_eval4

KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3)]
FPRS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)


def hook1584(fpr: float):
    """`p18_retrain.hook1584` と同じ決め方で、誤って回す割合だけ変える"""
    def hook(n: int, src: str, heads: list) -> list:
        rows = read_jsonl(TR3 / "utterances.jsonl")
        d = WORK / f"L{n:02d}" / "features"
        pick = [i for i, r in enumerate(rows) if r["split"] == "validation" and r["route"] != "UNKNOWN"]
        z = np.load(d / "features_F2.npz")
        last, mean = z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)
        hd = HEADS[src](n)
        reps = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
        out = []
        for (seed, classes, th, p) in heads:
            name = f"heads_mlp_seed{seed}.pt"
            val = head_probs(load_pt(hd / name), float(reps[name]["temperature"]), last, mean)["reject"]
            out.append((seed, classes, {**th, "REJECT": quantile(val, fpr)}, p))
        return out
    return hook


def measure(n: int, src: str, ts: dict, loader, fpr: float = 0.02):
    if not has_heads(n, src) or loader(n) is None:
        return None
    final = STEPS[-1]
    hook = hook933("val", fpr, {}) if src == SRC933 else hook1584(fpr)
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=hook, steps=[final], sources=[src], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=make_rules, amb_chat_ok=True)
    return blob["kpi"][f"{n}|{src}|{final[0]}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    raw = {k: json.loads((WORK / f"text_scores_{k}.json").read_text(encoding="utf-8")) for k in "EF"}
    names = sorted(raw["F"]["models"])
    print("\n===== P18 分析（結果を見た後。合否ではない）=====")
    md = ["**ここの数字は、評価セット v4 の結果を見てから出したもの。合否ではない**（合否は `retrain.md` の本命の数字）。"
          "数字はヘッドの seed 3 つ（× BERT の seed）の平均。かっこは最小〜最大\n"]
    res = {"bert_seed": {}, "kpi_by_bert": {}, "reject_sweep": {}, "reject_sweep_mean": {}}

    # ---- 1. BERT の seed ごと: 対象の依頼にフラグを立てる割合
    held = [("学習側の交差検証（5 つに分けて学習したモデル）", None),
            ("1,101 行の test（全体で学習したモデル。以下同じ）", [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] == "test"]),
            ("評価セット v1", read_jsonl(CH1)), ("評価セット v2", read_jsonl(CH2 / "utterances.jsonl")),
            ("評価セット v3", read_jsonl(CH3 / "utterances.jsonl")), ("評価セット v4", read_jsonl(CH4 / "utterances.jsonl"))]
    print("\n===== 1. BERT が対象の依頼と雑談にフラグ（0.5 以上）を立てる割合（seed ごと）=====")
    md += ["\n## 1. BERT が対象の依頼と雑談にフラグ（0.5 以上）を立てる割合（seed ごと）\n",
           "|測る行|行|" + "|".join(f"案 {k}・{nm}" for k in "EF" for nm in names) + "|", "|---|---|" + "---|" * (2 * len(names))]
    for title, rows in held:
        cells, n_rows = [], 0
        for k in "EF":
            fit = list(raw[k]["labels_fit"])
            y = np.array([raw[k]["labels_fit"][i] for i in fit])
            for nm in names:
                m = raw[k]["models"][nm]
                if rows is None:
                    v = np.array([m["cv"][i] for i in fit])[y == 0]
                    n_rows = f"{int((y == 0).sum())}（案 F）"
                else:
                    known = [r for r in rows if r["action"] not in ("UNKNOWN", "MULTI")]
                    v = np.array([m["pred"][r["id"]] for r in known])
                    n_rows = len(known)
                x = float(np.mean(v >= BERT_LO))
                res["bert_seed"].setdefault(title, {})[f"{k}|{nm}"] = x
                cells.append(f"{x:.3f}")
        print(f"{title:<30} " + "  ".join(cells))
        md.append(f"|{title}|{n_rows}|" + "|".join(cells) + "|")
    ch4 = read_jsonl(CH4 / "utterances.jsonl")
    isH = [r["category"] == "H" for r in ch4]
    rec = [f"{float(np.mean([raw[k]['models'][nm]['pred'][r['id']] >= BERT_LO for r, h in zip(ch4, isH) if h])):.3f}" for k in "EF" for nm in names]
    md.append("|（参考）評価セット v4 の機能にない依頼にフラグ|150|" + "|".join(rec) + "|")
    print(f"{'（参考）v4 の機能にない依頼にフラグ':<30} " + "  ".join(rec))
    worst = max(names, key=lambda nm: res["bert_seed"]["評価セット v4"][f"F|{nm}"])
    rest = [nm for nm in names if nm != worst]
    p = {nm: {r["id"]: raw["F"]["models"][nm]["pred"][r["id"]] for r in ch4} for nm in names}
    only = [r["utterance"] for r in ch4 if r["action"] != "UNKNOWN" and p[worst][r["id"]] >= BERT_LO and all(p[nm][r["id"]] < BERT_LO for nm in rest)]
    md.append(f"\n案 F の {worst} だけがフラグを立てた評価セット v4 の対象の行（{len(only)} 行）: " + " / ".join(only))
    res["bert_seed"]["only_worst"] = {"seed": worst, "rows": only}

    # ---- 2. 本命の構成で、BERT を seed ごと・seed の平均にしたときの KPI
    variants = [(f"案 F・{nm} だけ", by_mode(raw["F"], MODES[0], only=(nm,))) for nm in names] + \
               [("案 F・seed 3 つ（本命。1 つずつ）", by_mode(raw["F"], MODES[0])), ("案 F・seed 3 つの確率の平均", by_mode(raw["F"], MODES[1])),
                ("案 E・seed 3 つの確率の平均", by_mode(raw["E"], MODES[1]))]
    sets = [("v4", cached(load_eval4)), ("v3", cached(load_eval3))]
    md.append("\n## 2. 本命の構成（ヘッド 1,584 行）で、BERT を seed ごと・seed の平均にしたときの KPI\n")
    for tag, loader in sets:
        for n in ns:
            print(f"\n===== 2. BERT の seed を分けた KPI・{tag}・層 {n} 個（ヘッド 1,584 行）=====")
            md += [f"\n### {tag}・層 {n} 個\n", "|BERT|" + "|".join(lab for lab, _, _ in KPI) + "|", "|---|" + "---|" * len(KPI)]
            for title, ts in variants:
                k = measure(n, SRC1584, ts, loader)
                if k is None:
                    continue
                cells = [cell(k[key], nd) for _, key, nd in KPI]
                res["kpi_by_bert"][f"{tag}|{n}|{title}"] = {key: k[key] for _, key, _ in KPI}
                print(f"{title:<26} " + "  ".join(f"{lab}: {x}" for (lab, _, _), x in zip(KPI, cells)))
                md.append(f"|{title}|" + "|".join(cells) + "|")

    # ---- 3. 棄却の閾値（validation の x%）を動かす。BERT は案 F（seed 3 つ）
    tsF = by_mode(raw["F"], MODES[0])
    md.append("\n## 3. 棄却の閾値の決め方（validation の行き先のある行の x% を誤って回す値）を動かす。BERT は案 F・seed 3 つ\n")
    for tag, loader in sets:
        for n in ns:
            print(f"\n===== 3. 棄却の閾値・{tag}・層 {n} 個 =====")
            md += [f"\n### {tag}・層 {n} 個\n", "|ヘッド|x|" + "|".join(lab for lab, _, _ in KPI[:4]) + "|", "|---|---|" + "---|" * 4]
            for src in (SRC933, SRC1584):
                for fpr in FPRS:
                    k = measure(n, src, tsF, loader, fpr)
                    if k is None:
                        continue
                    cells = [cell(k[key], nd) for _, key, nd in KPI[:4]]
                    res["reject_sweep"][f"{tag}|{n}|{src}|{fpr}"] = {key: k[key] for _, key, _ in KPI[:4]}
                    mark = "**" if fpr == 0.02 else ""
                    print(f"{src:<14} {int(fpr * 100):>2}%  " + "  ".join(f"{lab}: {x}" for (lab, _, _), x in zip(KPI, cells)))
                    md.append(f"|{src}|{mark}{int(fpr * 100)}%{mark}|" + "|".join(cells) + "|")
    # ---- 4. 同じことを、BERT を案 F・seed 3 つの確率の平均にして（ヘッド 1,584 行）
    tsM = by_mode(raw["F"], MODES[1])
    md.append("\n## 4. 棄却の閾値を動かす（ヘッド 1,584 行・BERT は案 F の seed 3 つの確率の平均）\n")
    for tag, loader in sets:
        for n in ns:
            print(f"\n===== 4. 棄却の閾値（BERT は seed の平均）・{tag}・層 {n} 個 =====")
            md += [f"\n### {tag}・層 {n} 個\n", "|x|" + "|".join(lab for lab, _, _ in KPI) + "|", "|---|" + "---|" * len(KPI)]
            for fpr in FPRS:
                k = measure(n, SRC1584, tsM, loader, fpr)
                if k is None:
                    continue
                cells = [cell(k[key], nd) for _, key, nd in KPI]
                res["reject_sweep_mean"][f"{tag}|{n}|{fpr}"] = {key: k[key] for _, key, _ in KPI}
                mark = "**" if fpr == 0.02 else ""
                print(f"{int(fpr * 100):>2}%  " + "  ".join(f"{lab}: {x}" for (lab, _, _), x in zip(KPI, cells)))
                md.append(f"|{mark}{int(fpr * 100)}%{mark}|" + "|".join(cells) + "|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "analyze.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "analyze.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（analyze.md・analyze.json）")


if __name__ == "__main__":
    main()
