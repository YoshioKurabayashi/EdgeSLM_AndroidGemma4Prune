# -*- coding: utf-8 -*-
"""P14: 対象の語の食い違いを確かめる規則（S4）を足して、評価セット v3・v2 で測る。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P14_対象の語の食い違いを確かめる.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

|構成|規則|
|---|---|
|P13 の本命|EdgeSLM_MultiHead_Multi の規則 ＋ `p13_rules`（言い直し・取り消し・言いよどみ）|
|＋対象の語の食い違い|上に `p14_rules` の S4 を重ねる|

どちらも、ヘッドは 933 行（P5）・引数は P9・棄却の閾値は validation の 2%（P10）・BERT は機能にない依頼 36 機能を足した案 E（P13）・
断る閾値は学習側の 99.5% 点・フラグはすべての機能（P12）。

**S4 は評価セット v3 の重大な誤りを見て作ったので、v3 の数字は甘く出る**（設計書 4 節）。

    python -m app.p14_eval        # results/P14/mismatch.md・mismatch.json
"""
from __future__ import annotations

import argparse
import collections
import json

from .common import PROJECT_ROOT
from .p7_data import CH2, read_jsonl
from .p8_gate import STEPS, run
from .p9_arguments import make_resolve
from .p10_reject import SRC, make_hook
from .p11_bert import MODES, by_mode
from .p13_data import CH3, NOTES
from .p13_eval import WORK as P13, cell, hi_rule, interval, load_eval3
from .p13_rules import make_rules as make_p13
from .p14_rules import make_rules as make_p14

RES = PROJECT_ROOT / "results" / "P14"
CONFIGS = [("P13 の本命", make_p13), ("＋対象の語の食い違い", make_p14)]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("重大な誤りの割合", "severe", 4), ("不要な聞き返し", "over", 3),
       ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3)]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    f = P13 / "text_scores_E.json"
    if not f.exists():
        raise SystemExit("P13 の BERT の結果が無い（先に .\\scripts\\run_p13.ps1）")
    ts = by_mode(json.loads(f.read_text(encoding="utf-8")), MODES[0])
    final = STEPS[-1]
    sets = [("評価セット v3", read_jsonl(CH3 / "utterances.jsonl"), load_eval3), ("評価セット v2", read_jsonl(CH2 / "utterances.jsonl"), None)]
    print("\n===== P14: 対象の語の食い違いを確かめる規則（S4）=====")
    print("注意: S4 は評価セット v3 の重大な誤りを見て作った。評価セット v3 の数字は甘く出る")
    md = ["S4 は評価セット v3 の重大な誤りを見て作ったので、評価セット v3 の数字は甘く出る。数字はヘッドの seed 3 つ × BERT の seed 3 つの平均（かっこは最小〜最大）。"
          "区間は 95% 信頼区間の目安（Wilson）\n"]
    res = {"kpi": {}, "severe_by_category": {}, "moved": {}, "detail": {}}
    for sname, rows, loader in sets:
        ids = {r["id"] for r in rows}
        by_utt = {r["utterance"]: r for r in rows}
        out = {}
        for cname, mk in CONFIGS:
            blob = run(ns, make_resolve=make_resolve, out=None, heads_hook=make_hook("val", 0.02, {}), steps=[final], sources=[SRC], quiet=True,
                       ts=ts, watch=ids, bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=mk)
            out[cname] = {n: (blob["kpi"][f"{n}|{SRC}|{final[0]}"], blob["detail"][f"{n}|{SRC}|{final[0]}"]) for n in ns if f"{n}|{SRC}|{final[0]}" in blob["kpi"]}
        for n in ns:
            if n not in out[CONFIGS[0][0]]:
                continue
            print(f"\n===== {sname}（{len(rows)} 行）・層 {n} 個 =====")
            heads = [lab for lab, _, _ in KPI] + ["EtoE の成功の区間", "重大な誤りの割合の区間"]
            md += [f"\n### {sname}（{len(rows)} 行）・層 {n} 個\n", "|構成|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
            for cname, _ in CONFIGS:
                k, d = out[cname][n]
                cells = [cell(k[key], nd) for _, key, nd in KPI] + [interval(k["e2e"]["mean"], len(rows)), interval(k["severe"]["mean"], len(rows))]
                res["kpi"][f"{sname}|{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, "e2e_ci": cells[-2], "severe_ci": cells[-1]}
                res["detail"][f"{sname}|{n}|{cname}"] = {x: d[x] for x in ("stages", "severe_rows", "normal_rows", "by_group")}
                print(f"{cname:<14} " + "  ".join(f"{h}: {x}" for h, x in zip(heads, cells)))
                md.append(f"|{cname}|" + "|".join(cells) + "|")
            # 重大な誤りの内訳（種類ごとの行数）
            md += ["\n重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|合計|", "|---|" + "---|" * (len(CAT) + 1)]
            for cname, _ in CONFIGS:
                c = collections.Counter()
                for x in out[cname][n][1]["by_group"]:
                    if x["grade"] == "重大":
                        c[x["category"]] += x["rows"]
                res["severe_by_category"][f"{sname}|{n}|{cname}"] = dict(c)
                print(f"  {cname} の重大な誤り: " + "・".join(f"{CAT[k]} {c[k]:.1f}" for k in CAT) + f" / 合計 {sum(c.values()):.1f}")
                md.append(f"|{cname}|" + "|".join(f"{c[k]:.1f}" for k in CAT) + f"|{sum(c.values()):.1f}|")
            # S4 で行き先が変わった行（実行 → 聞き返し）
            def share(d, kind):
                o = collections.Counter()
                for x in d["watch"]:
                    if x["outcome"].startswith(kind):
                        o[x["utterance"]] += x["share"]
                return o
            ex0, ex1 = share(out[CONFIGS[0][0]][n][1], "実行"), share(out[CONFIGS[1][0]][n][1], "実行")
            moved = sorted(((ex0[u] - ex1[u], u) for u in ex0 if ex0[u] - ex1[u] > 1e-9), reverse=True)
            groups = collections.Counter()
            for v, u in moved:
                r = by_utt[u]
                groups[CAT[r["category"]]] += v
            res["moved"][f"{sname}|{n}"] = [{"utterance": u, "category": by_utt[u]["category"], "action": by_utt[u]["action"], "share": v} for v, u in moved]
            line = "・".join(f"{k} {v:.1f} 行" for k, v in groups.items())
            print(f"  S4 が実行を止めて聞き返しに回した行（1 回あたり）: {line}")
            print("    " + " / ".join(f"[{by_utt[u]['category']}]{u}（{v:.2f}）" for v, u in moved[:16]))
            md += [f"\nS4 が実行を止めて聞き返しに回した行（1 回あたり）: {line}\n", "|種類|発話（正解）|割合|", "|---|---|---|"]
            md += [f"|{CAT[by_utt[u]['category']]}|{u}（{by_utt[u]['action'].replace('TOOL_', '')}）|{v:.2f}|" for v, u in moved[:40]]
            d1 = out[CONFIGS[1][0]][n][1]
            md += ["\nS4 を足しても残る重大な誤り:\n", "|種類|発話 → 実行したツール（割合）|", "|---|---|"]
            left = collections.defaultdict(list)
            for x in d1["severe_rows"]:
                left[CAT[x["category"]]].append(f"{x['utterance']} → {x['executed']}（{x['share']:.2f}）")
            md += [f"|{k}|" + " / ".join(v[:20]) + "|" for k, v in left.items()]
            print("  残る重大な誤り: " + " / ".join(f"[{x['category']}]{x['utterance']}→{x['executed']}（{x['share']:.2f}）" for x in d1["severe_rows"][:16]))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "mismatch.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "mismatch.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（mismatch.md・mismatch.json）")


if __name__ == "__main__":
    main()
