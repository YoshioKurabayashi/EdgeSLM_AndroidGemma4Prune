# -*- coding: utf-8 -*-
"""P12: BERT の断る閾値の決め方と、フラグを効かせる範囲を変えて、対象を断る件数と KPI を見る。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P12_BERTの断る閾値の決め方.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

P11 で、断る閾値（学習側で 1% を止める値）が seed で 0.58〜0.99 に割れ、低い seed は対象の依頼を断った。
ここでは次の 2 つを差し替える（どれも学習側の交差検証だけで決まる。評価セットを見ない）。

|断る閾値の決め方|中身|
|---|---|
|いまの決め方|学習側の 99% 点|
|99.5% 点|学習側の 0.5% を止める値|
|1 件も止めない|学習側の最大より上|
|下限つき|99% 点と 0.9 の大きい方|
|断らない|フラグだけ（参考）|

|フラグの範囲|中身|
|---|---|
|操作系だけ|いまのまま|
|すべての機能|読むだけの機能・雑談でも、確率 0.5 以上なら聞き返す|

**あらかじめ決めた本命は「99.5% 点 ＋ すべての機能」**（設計書 5 節）。BERT は P11 の本命（案 C）。参考に案 A。
ヘッドは 933 行（P5）・引数は P9 の補った処理・棄却の閾値は validation の 2%（P10）。

    python -m app.p12_block                 # results/P12/block_rule.md・block_rule.json
    python -m app.p12_block --berts C       # BERT の案をしぼる
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import PROJECT_ROOT
from .p7_data import CH1, CH2, read_jsonl
from .p8_gate import BERT_LO, STEPS, run
from .p9_arguments import make_resolve
from .p10_reject import PRIMARY as REJECT_RULE, SRC, make_hook
from .p11_bert import MODES, RES as P11_RES, VARIANTS, by_mode, groups, load_variants

RES = PROJECT_ROOT / "results" / "P12"
q = lambda neg, x: float(np.quantile(neg, x, method="higher"))
RULES = [("いまの決め方（99% 点）", lambda neg: q(neg, 0.99)),
         ("99.5% 点", lambda neg: q(neg, 0.995)),
         ("1 件も止めない", lambda neg: float(neg.max()) + 1e-4),
         ("下限つき（99% 点と 0.9 の大きい方）", lambda neg: max(q(neg, 0.99), 0.9)),
         ("断らない（フラグだけ）", lambda neg: 1.01)]
SCOPES = [("操作系だけ", False), ("すべての機能", True)]
PRIMARY = ("99.5% 点", "すべての機能")
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("通常の誤り", "normal", 3),
       ("1 回目で正しく処理", "first", 3), ("機能にない依頼を断った", "os_refuse", 3), ("同・聞き返した", "os_clarify", 3), ("対象を断った（件）", "known_refused_n", 1)]


def cell(v: dict, nd: int = 3) -> str:
    return f"{v['mean']:.{nd}f}" if v["min"] == v["max"] else f"{v['mean']:.{nd}f}（{v['min']:.{nd}f}〜{v['max']:.{nd}f}）"


def agg(xs: list) -> dict:
    return {"mean": float(np.mean(xs)), "min": float(min(xs)), "max": float(max(xs))}


def text_side(ts: dict, g: dict, rule) -> dict:
    """決め方ごとに、seed ごとの閾値と、断った行の数（学習側の交差検証・評価セット v1・v2）"""
    fit_ids = list(ts["labels_fit"])
    y = np.array([ts["labels_fit"][i] for i in fit_ids])
    out = {"hi": [], "cv_known": [], "cv_os": [], "ch1_known": [], "ch1_os": [], "ch2_known": [], "ch2_os": [], "test_known": [], "test_os": []}
    for name in sorted(ts["models"]):
        m = ts["models"][name]
        hi = rule(np.array([m["cv"][i] for i in fit_ids])[y == 0])
        n_over = lambda part, ids: int(sum(m[part][i] >= hi for i in ids))
        out["hi"].append(hi)
        out["cv_known"].append(n_over("cv", g["cv"]["known"]))
        out["cv_os"].append(n_over("cv", g["cv"]["os"]) / len(g["cv"]["os"]))
        for s in ("test", "ch1", "ch2"):
            out[f"{s}_known"].append(n_over("pred", g[s]["known"]))
            out[f"{s}_os"].append(n_over("pred", g[s]["os"]) / len(g[s]["os"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--berts", default="C,A")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    tss = load_variants()
    keys = [k for k in a.berts.split(",") if k in tss]
    if not keys:
        raise SystemExit("BERT の結果が無い（先に P11 の text）")
    title = {k: t for k, t, _ in VARIANTS}
    g = groups()
    final = STEPS[-1]
    print(f"\n===== P12: BERT の断る閾値の決め方 × フラグの範囲（ヘッド {SRC}・引数は補った処理・棄却の閾値は {REJECT_RULE}）=====")
    print(f"あらかじめ決めた本命: {PRIMARY[0]} ＋ フラグは{PRIMARY[1]}（BERT は案 C・1 つずつ）")
    n_of = {s: {k: len(v) for k, v in d.items()} for s, d in g.items()}
    md = [f"数える行: 学習側の交差検証（対象の依頼と雑談 {n_of['cv']['known']}・機能にない依頼 {n_of['cv']['os']}）/ test（{n_of['test']['known']}・{n_of['test']['os']}）/ "
          f"評価セット v1（{n_of['ch1']['known']}・{n_of['ch1']['os']}）/ 評価セット v2（{n_of['ch2']['known']}・{n_of['ch2']['os']}）。"
          f"機能にない依頼はシートヒーター・ハンドルヒーターを除く。フラグ = 確率 {BERT_LO} 以上。BERT は 1 つずつ使う（seed 3 つ）\n"]
    res = {"text": {}, "kpi": {}, "detail": {}, "rows": n_of}
    for k in keys:
        ts = by_mode(tss[k], MODES[0])
        # ---- 1. 閾値と、断った行
        print(f"\n########## BERT: 案 {k}（{title[k]}）##########")
        md += [f"\n## BERT: 案 {k}（{title[k]}）\n", "### 1. 決め方ごとの閾値と、BERT が断った行（seed ごと。かっこの外は平均）\n",
               "|断る閾値の決め方|閾値（seed 0・1・2）|学習側: 対象を断った（行）|学習側: 機能にない依頼を断った|test: 対象（行）|v1: 対象（行）|v1: 機能にない依頼|v2: 対象（行）|v2: 機能にない依頼|",
               "|---|---|---|---|---|---|---|---|---|"]
        for rname, rule in RULES:
            t = text_side(ts, g, rule)
            res["text"][f"{k}|{rname}"] = t
            row = ["・".join(f"{x:.3f}" for x in t["hi"]), "・".join(str(x) for x in t["cv_known"]), cell(agg(t["cv_os"])),
                   "・".join(str(x) for x in t["test_known"]), "・".join(str(x) for x in t["ch1_known"]), cell(agg(t["ch1_os"])),
                   "・".join(str(x) for x in t["ch2_known"]), cell(agg(t["ch2_os"]))]
            print(f"{rname:<22} 閾値 {row[0]} | 学習側 対象 {row[1]} 行・機能にない依頼 {row[2]} | test 対象 {row[3]} | v1 対象 {row[4]}・機能にない依頼 {row[5]} | "
                  f"v2 対象 {row[6]}・機能にない依頼 {row[7]}")
            mark = "**" if rname == PRIMARY[0] else ""
            md.append(f"|{mark}{rname}{mark}|" + "|".join(row) + "|")
        # ---- 2. KPI
        out = {}
        for rname, rule in RULES:
            for sname, flag_all in SCOPES:
                blob = run(ns, make_resolve=make_resolve, out=None, heads_hook=make_hook("val", 0.02, {}), steps=[final], sources=[SRC], quiet=True,
                           ts=ts, watch=set(), bert_hi=lambda name, neg, rule=rule: rule(neg), flag_all=flag_all)
                for n in ns:
                    key = f"{n}|{SRC}|{final[0]}"
                    if key in blob["kpi"]:
                        out[(n, rname, sname)] = (blob["kpi"][key], blob["detail"][key])
        if k == "C" and (P11_RES / "bert_variants.json").exists():      # 確かめ: いまの決め方・操作系だけ = P11 の案 C
            p11 = json.loads((P11_RES / "bert_variants.json").read_text(encoding="utf-8"))["kpi"]
            diff = [f"層 {n}・{key}" for n in ns for _, key, _ in KPI if key != "normal" and (n, RULES[0][0], SCOPES[0][0]) in out
                    and abs(out[(n, RULES[0][0], SCOPES[0][0])][0][key]["mean"] - p11[f"{n}|C|{MODES[0]}"][key]["mean"]) > 1e-9]
            line = "確かめ（いまの決め方・操作系だけ ↔ P11 の案 C）: " + ("同じ数字" if not diff else f"違う {diff}")
            print("\n" + line)
            md.append("\n" + line)
            res["check_p11"] = not diff
            if diff:
                raise SystemExit("P11 と数字が合わない。流れのどこかが変わっている")
        for n in ns:
            print(f"\n===== 案 {k}・層 {n} 個の KPI（評価セット v2）=====")
            md += [f"\n### 2. 評価セット v2 の KPI・層 {n} 個（ヘッドの seed 3 つ × BERT の seed 3 つの平均。かっこは最小〜最大）\n",
                   "|断る閾値の決め方|フラグの範囲|" + "|".join(lab for lab, _, _ in KPI) + "|", "|---|---|" + "---|" * len(KPI)]
            for rname, _ in RULES:
                for sname, _ in SCOPES:
                    if (n, rname, sname) not in out:
                        continue
                    kp, d = out[(n, rname, sname)]
                    cells = [cell(kp[key], nd) for _, key, nd in KPI]
                    res["kpi"][f"{k}|{n}|{rname}|{sname}"] = {key: kp[key] for _, key, _ in KPI}
                    res["detail"][f"{k}|{n}|{rname}|{sname}"] = d
                    mark = "**" if (rname, sname) == PRIMARY else ""
                    print(f"{rname:<22} {sname:<7} " + "  ".join(f"{lab}: {x}" for (lab, _, _), x in zip(KPI, cells)))
                    md.append(f"|{mark}{rname}{mark}|{mark}{sname}{mark}|" + "|".join(cells) + "|")
            if (n, *PRIMARY) in out:
                base, prim = out[(n, RULES[0][0], SCOPES[0][0])][1], out[(n, *PRIMARY)][1]
                for lab, d in (("いまの決め方・操作系だけ", base), ("本命", prim)):
                    md += [f"\n{lab}のとき:\n", "|見るもの|行（出た割合）|", "|---|---|",
                           "|通常の誤りになった行|" + " / ".join(f"[{x['category']}]{x['utterance']} → {x['outcome']}（{x['share']:.2f}）" for x in d["normal_rows"][:16]) + "|",
                           "|BERT のフラグで聞き返した対象の行|" + " / ".join(f"{x['utterance']}（{x['tool']}・{x['share']:.2f}）" for x in d["flag_rows"][:12]) + "|",
                           "|残った重大な誤り|" + " / ".join(f"{x['utterance']} → {x['executed']}（{x['share']:.2f}）" for x in d["severe_rows"][:12]) + "|"]
                    print(f"  {lab}・通常の誤り: " + " / ".join(f"[{x['category']}]{x['utterance']}→{x['outcome']}（{x['share']:.2f}）" for x in d["normal_rows"][:10]))
                print("  本命・フラグで聞き返した対象の行: " + " / ".join(f"{x['utterance']}（{x['share']:.2f}）" for x in prim["flag_rows"][:10]))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "block_rule.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "block_rule.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（block_rule.md・block_rule.json）")


if __name__ == "__main__":
    main()
