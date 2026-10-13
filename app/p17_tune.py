# -*- coding: utf-8 -*-
"""P17: 流れのヘッドを 1,101 行に切り替え、不要な聞き返しを減らす手当てを足して、評価セット v3 で測る。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P17_不要な聞き返しを減らす.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

|構成|変えるもの|
|---|---|
|0 いまの構成（P14）|ヘッド 933 行・引数は P9・規則は P14・確信度の閾値あり|
|1 ヘッドを 1,101 行に|P7 で学習済みのヘッド（機能にない依頼 180 行・機能の語を含む機能にある依頼 48 行を足したもの）|
|2 ＋引数の取り出し|`p17_arguments`（ナビの言い方・小さめ）|
|3 ＋規則の例外|`p17_rules`（指示語でない「これまで」「直してくれる」、曲の情報の「これ」）|
|4 ＋確信度の閾値を外す|棄却ヘッド・BERT・規則は残す|
|参考: 933 行のまま 2〜4|ヘッドを切り替えずに、同じ手当てを足す|

どの構成も、BERT は案 E・断る閾値は学習側の 99.5% 点・フラグはすべての機能・棄却の閾値は validation の 2%・
数え方は P15（意図が不明を雑談として受けるのは正解）。

**手当ては評価セット v3 の聞き返した行を見て作った。評価セット v3 は「直す用」として使っている。合否の数字としては読まない。**

    python -m app.p17_tune        # results/P17/tune.md・tune.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import PROJECT_ROOT
from .p6_seeds import SEEDS, head_probs, load_pt
from .p7_block import HEADS, WORK as P7
from .p7_data import TR2, read_jsonl
from .p8_gate import STEPS, run
from .p9_arguments import make_resolve as args_p9
from .p10_reject import SRC as SRC933, make_hook as hook933, quantile
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import WORK as P13, cell, hi_rule, interval, load_eval3
from .p14_rules import make_rules as rules_p14
from .p17_arguments import make_resolve as args_p17
from .p17_rules import make_rules as rules_p17

RES = PROJECT_ROOT / "results" / "P17"
SRC1101 = "1,101 行（P7）"
CONFIGS = [("0 いまの構成（P14・ヘッド 933 行）", SRC933, args_p9, rules_p14, True),
           ("1 ヘッドを 1,101 行に", SRC1101, args_p9, rules_p14, True),
           ("2 ＋引数の取り出し", SRC1101, args_p17, rules_p14, True),
           ("3 ＋規則の例外", SRC1101, args_p17, rules_p17, True),
           ("4 ＋確信度の閾値を外す", SRC1101, args_p17, rules_p17, False),
           ("参考: 933 行のまま 2・3", SRC933, args_p17, rules_p17, True),
           ("参考: 933 行のまま 2・3・4", SRC933, args_p17, rules_p17, False)]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("通常の誤り", "normal", 3),
       ("1 回目で正しく処理", "first", 3), ("対象を断った（件）", "known_refused_n", 1)]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}


def hook1101(n: int, src: str, heads: list) -> list:
    """1,101 行のヘッドの棄却の閾値を、validation の行き先のある行の 2% を誤って回す値にする（P10 と同じ決め方）"""
    rows = read_jsonl(TR2 / "utterances.jsonl")
    d = P7 / f"L{n:02d}" / "features"
    ids = json.loads((d / "features_F2.meta.json").read_text(encoding="utf-8"))["ids"]
    if ids != [r["id"] for r in rows]:
        raise SystemExit(f"層 {n}: 1,101 行の特徴の並びがデータと違う")
    pick = [i for i, r in enumerate(rows) if r["split"] == "validation" and r["route"] != "UNKNOWN"]
    z = np.load(d / "features_F2.npz")
    last, mean = z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)
    hd = HEADS[src](n)
    reps = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
    out = []
    for (seed, classes, th, p) in heads:
        name = f"heads_mlp_seed{seed}.pt"
        val = head_probs(load_pt(hd / name), float(reps[name]["temperature"]), last, mean)["reject"]
        out.append((seed, classes, {**th, "REJECT": quantile(val, 0.02)}, p))
    return out


def make_hook(src: str, keep_conf: bool, log: dict):
    base = hook933("val", 0.02, {}) if src == SRC933 else hook1101

    def hook(n: int, src_: str, heads: list) -> list:
        heads = base(n, src_, heads)
        log[n] = [h[2]["REJECT"] for h in heads]
        if keep_conf:
            return heads
        return [(seed, classes, {**th, "TOOL_write": 0.0, "TOOL_read": 0.0, "CHAT": 0.0}, p) for seed, classes, th, p in heads]
    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    ts = by_mode(json.loads((P13 / "text_scores_E.json").read_text(encoding="utf-8")), MODES[0])
    rows = read_jsonl(CH3 / "utterances.jsonl")
    final = STEPS[-1]
    print("\n===== P17: ヘッドを 1,101 行に切り替え、不要な聞き返しを減らす（評価セット v3・P15 の数え方）=====")
    print("注意: 手当ては評価セット v3 の聞き返した行を見て作った。合否の数字としては読まない")
    md = ["評価セット v3（800 行）・P15 の数え方。数字はヘッドの seed 3 つ × BERT の seed 3 つの平均（かっこは最小〜最大）。"
          "手当ては評価セット v3 の聞き返した行を見て作ったので、合否の数字としては読まない\n"]
    res = {"kpi": {}, "taus": {}, "stops": {}, "severe": {}, "detail": {}}
    for n in ns:
        print(f"\n===== 層 {n} 個 =====")
        heads = [lab for lab, _, _ in KPI] + ["EtoE の成功の区間"]
        md += [f"\n### 層 {n} 個\n", "|構成|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
        keep = {}
        for cname, src, args, rules, keep_conf in CONFIGS:
            log = {}
            blob = run([n], make_resolve=args, out=None, heads_hook=make_hook(src, keep_conf, log), steps=[final], sources=[src], quiet=True,
                       ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=load_eval3, make_rules=rules, amb_chat_ok=True)
            key = f"{n}|{src}|{final[0]}"
            if key not in blob["kpi"]:
                continue
            k, d = blob["kpi"][key], blob["detail"][key]
            cells = [cell(k[x], nd) for _, x, nd in KPI] + [interval(k["e2e"]["mean"], len(rows))]
            res["kpi"][f"{n}|{cname}"] = {**{x: k[x] for _, x, _ in KPI}, "severe": k["severe"], "e2e_ci": cells[-1]}
            res["taus"][f"{n}|{cname}"] = log.get(n, [])
            keep[cname] = d
            print(f"{cname:<24} " + "  ".join(f"{h}: {x}" for h, x in zip(heads, cells)))
            md.append(f"|{cname}|" + "|".join(cells) + "|")
        # 対象の行を止めた段・重大な誤りの内訳
        md += ["\n対象の依頼と雑談 480 行を止めた段（1 回あたりの行数）:\n", "|構成|規則|引数が足りない|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|検査|合計|",
               "|---|---|---|---|---|---|---|---|---|"]
        for cname, d in keep.items():
            c = collections.Counter()
            for x in d["stop_rows"]:
                c[x["stop"]] += x["share"]
            v = [c["聞き返し（規則）"], c["聞き返し（引数が足りない）"], c["聞き返し（棄却ヘッド）"], c["聞き返し（確信度）"], c["聞き返し（BERT のフラグ）"],
                 c["拒否（BERT が断る）"], c["聞き返し（検査）"] + c["拒否（検査）"]]
            res["stops"][f"{n}|{cname}"] = v
            md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in v) + f"|{sum(v):.1f}|")
            print(f"  {cname[:14]:<14} 止めた段: 規則 {v[0]:.1f}・引数 {v[1]:.1f}・棄却 {v[2]:.1f}・確信度 {v[3]:.1f}・BERT フラグ {v[4]:.1f}・BERT 断る {v[5]:.1f}・検査 {v[6]:.1f}")
        md += ["\n重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|合計|", "|---|" + "---|" * (len(CAT) + 1)]
        for cname, d in keep.items():
            c = collections.Counter()
            for x in d["by_group"]:
                if x["grade"] == "重大":
                    c[x["category"]] += x["rows"]
            res["severe"][f"{n}|{cname}"] = dict(c)
            md.append(f"|{cname}|" + "|".join(f"{c[k]:.1f}" for k in CAT) + f"|{sum(c.values()):.1f}|")
            print(f"  {cname[:14]:<14} 重大: " + "・".join(f"{CAT[k]} {c[k]:.1f}" for k in CAT))
        for cname in (CONFIGS[3][0], CONFIGS[4][0]):
            if cname in keep:
                d = keep[cname]
                res["detail"][f"{n}|{cname}"] = {x: d[x] for x in ("severe_rows", "normal_rows", "stop_rows", "by_group", "stages")}
                md += [f"\n{cname} で残る重大な誤り: " + " / ".join(f"[{x['category']}]{x['utterance']} → {x['executed']}（{x['share']:.2f}）" for x in d["severe_rows"][:24])]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tune.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "tune.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（tune.md・tune.json）")


if __name__ == "__main__":
    main()
