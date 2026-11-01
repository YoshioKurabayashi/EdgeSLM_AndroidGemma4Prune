# -*- coding: utf-8 -*-
"""P35: E3（止める語の後ろに別の操作）・返すもので絞るのを 4 つの読み出しの機能へ・施設の語の末尾 1 文字の不具合の当たり。端末なし・学習なし。

設計: DoC/設計書_P35_重大の余裕と返すものの広げ.md の 5 節。`app/p33_check.py` を写して、構成と評価セットを替えたもの。
**評価セット v6〜v10 はどれも直す用。ここの数字は当たりで、合否ではない**（合否は v11）。数え方は改定後（P33）だけ。

1. 引数の値の確かめ: 正解の引数がある行（v5〜v10 の 744 行・arg_eval_v1〜v4 の 960 行）で、P33 → P35 で悪くなった行・良くなった行
2. 対象を断ってしまう割合（4 つの読み出しの機能。正しい機能のとき）: 学習 1,656 行・v1〜v10 の対象の行
3. 流れの KPI: P33 ↔ P35 × seed（S1・3 つの平均）× v6〜v10 × 層 17・26

    python -m app.p35_check        # results/P35/check.md・check.json
"""
from __future__ import annotations

import collections
import contextlib
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from . import p8_gate
from .p7_data import read_jsonl
from .p11_bert import MODES, by_mode
from .p13_eval import cell
from .p18_retrain import cached
from .p19_config import bert_check
from .p21_data import CH6
from .p21_verdict import load_eval6
from .p22_fix import value_kpi
from .p22_labels import load_all, match
from .p27_data import CH7, labeled as labeled7
from .p27_verdict import load_eval7
from .p28_data import CH8, TR4, labeled as labeled8
from .p28_verdict import WORK as P28, load_eval8
from .p31_arguments import make_resolve as args_p31
from .p31_check import measure, pick_seed
from .p31_rules import make_rules as rules_p31
from .p32_data import CH9, labeled as labeled9
from .p32_verdict import load_eval9
from .p33_arguments import make_resolve as args_p33
from .p33_check import arg_rows as arg_rows33, grading
from .p34_data import CH10, labeled as labeled10
from .p34_verdict import WORK as P34W, load_eval10
from .p35_arguments import make_resolve as args_p35
from .p35_scope import explains

RES = PROJECT_ROOT / "results" / "P35"
CONFIGS = [("P33", args_p33), ("P35", args_p35)]
def arg_rows() -> list[dict]:
    return arg_rows33() + labeled10()


def main() -> None:
    use_multi_library()
    from edge_router.dispatch import arguments as A
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    md = ["# P35 重大の余裕と返すものの広げの当たり（端末なし・学習なし）\n",
          "**評価セット v6〜v10 はどれも直す用。ここの数字は当たりで、合否ではない**（合否は v11）。ヘッド 1,656 行・規則 P31・BERT bert_s0・数え方は改定後\n"]
    res = {"args": {}, "decline": {}, "kpi": {}}
    # 1. 引数の値
    r31, r33 = args_p33(A), args_p35(A)
    worse, better, n = [], [], 0
    for x in arg_rows():
        o1, o3 = r31(x["tool"], None, x["utterance"], state), r33(x["tool"], None, x["utterance"], state)
        m1, m3 = match(x["tool"], x["expected"], o1.complete, o1.values), match(x["tool"], x["expected"], o3.complete, o3.values)
        n += 1
        if m1 and not m3:
            worse.append(f"{x['utterance']}（{x['set']}・正解 {x['label']}・P35 {o3.values}）")
        if m3 and not m1:
            better.append(f"{x['utterance']}（{x['set']}・正解 {x['label']}・P33 {o1.values} → P35 {o3.values}）")
    res["args"] = {"rows": n, "worse": worse, "better": better}
    md += ["## 1. 引数の値（正解の引数がある行）\n", f"{n} 行で、P33 → P35 で悪くなった行 **{len(worse)}**・良くなった行 **{len(better)}**\n",
           "|悪くなった|" + (" / ".join(worse) or "なし") + "|", "|---|---|", "|良くなった|" + (" / ".join(better) or "なし") + "|"]
    print(md[-4].strip(), flush=True)
    # 2. 対象を断ってしまう割合
    md += ["\n## 2. 対象を断ってしまう割合（何を返すかで。正しい機能のとき）\n",
           "車両の状態・曲の情報は、利用者が決めた返すものと学習の対象の行で語を書いた時点で、v1〜v10 の対象の行で断ったのが 車両の状態 393 行中 3（0.8%）・曲の情報 230 行中 4（1.7%）。これが見ていない言い方を断る割合の見積もり。その 7 行を見て言い方（合計・全部・ジャンル・題名）を足した後:\n",
           "|機能|行|断る|断る行|", "|---|---|---|---|"]
    srcs = [("学習", TR4)] + [(f"v{v}", PROJECT_ROOT / "data" / f"challenge_v{v}") for v in range(1, 11)]
    for tool in ("search_poi", "search_vehicle_manual", "get_vehicle_status", "get_media_info"):
        us = [r["utterance"] for _, ch in srcs for r in read_jsonl(ch / "utterances.jsonl") if r["action"] == f"TOOL_{tool}"]
        bad = [f"{u}（{explains(tool, u)[1]}）" for u in us if not explains(tool, u)[0]]
        res["decline"][tool] = {"rows": len(us), "bad": bad}
        md.append(f"|{tool}|{len(us)}|{len(bad)}|{' / '.join(bad) or 'なし'}|")
        print(f"{tool}: {len(us)} 行中 {len(bad)} 行を断る", flush=True)
    # 3. 流れ
    raw = json.loads((P28 / "text_scores_F.json").read_text(encoding="utf-8"))
    raw9 = json.loads((PROJECT_ROOT / "work" / "p32" / "text_scores_F.json").read_text(encoding="utf-8"))
    raw10 = json.loads((P34W / "text_scores_F.json").read_text(encoding="utf-8"))
    seeds = {n_: pick_seed(n_)[0] for n_ in (17, 26)}
    sets = (("v10", read_jsonl(CH10 / "utterances.jsonl"), cached(lambda n_: load_eval10(n_)), labeled10(), raw10),
            ("v9", read_jsonl(CH9 / "utterances.jsonl"), cached(lambda n_: load_eval9(n_)), labeled9(), raw9),
            ("v8", read_jsonl(CH8 / "utterances.jsonl"), cached(load_eval8), labeled8(), raw),
            ("v7", read_jsonl(CH7 / "utterances.jsonl"), cached(load_eval7), labeled7(), raw),
            ("v6", read_jsonl(CH6 / "utterances.jsonl"), cached(load_eval6), [x for x in load_all() if x["set"] == "v6"], raw))
    md += ["\n## 3. 流れの KPI（値まで見た EtoE・重大な誤り（件）／不要な聞き返し／通常の誤り）\n",
           "数え方「これまで」= 機能にない依頼を雑談として受けたら通常の誤り。「改定後」= 正解（利用者の決定 2026-10-13）\n",
           "|評価セット|層|構成|seed|数え方|値まで見た EtoE|重大（件）|不要な聞き返し|通常の誤り|対象を断った（件）|", "|---|---|---|---|---|---|---|---|---|---|"]
    detail = {}
    for tag, rows, loader, lab, rw in sets:
        ts = by_mode(rw, MODES[0], only=(bert_check(rw)["one"],))
        for n_ in (17, 26):
            for cname, make in CONFIGS:
                for sd in (seeds[n_], None):
                    for new in (True,):
                        with grading(new):
                            got = measure(n_, ts, loader, make, rules_p31, sd)
                        if not got:
                            continue
                        k, d = got
                        v = value_kpi(k, d, lab, make(A), state, len(rows))
                        sname = f"{sd}（S1）" if sd is not None else "3 つの平均"
                        gname = "改定後" if new else "これまで"
                        res["kpi"][f"{tag}|{n_}|{cname}|{sname}|{gname}"] = {"e2e_value": v["e2e_value"], "severe_value_n": v["severe_value_n"], "over": k["over"]["mean"], "normal": k["normal"]["mean"], "known_refused_n": k["known_refused_n"]["mean"]}
                        if sd is not None:
                            detail[(tag, n_, cname, new)] = d
                        md.append(f"|{tag}|{n_}|{cname}|{sname}|{gname}|{v['e2e_value']:.3f}|{v['severe_value_n']:.1f}|{cell(k['over'], 3)}|{cell(k['normal'], 3)}|{k['known_refused_n']['mean']:.1f}|")
                        print(f"{tag}・層 {n_}・{cname}・{sname:<8}・{gname} EtoE {v['e2e_value']:.3f}・重大 {v['severe_value_n']:.1f}・聞き返し {k['over']['mean']:.3f}・通常 {k['normal']['mean']:.3f}・対象を断った {k['known_refused_n']['mean']:.1f}", flush=True)
    md.append("\n## 4. P33 → P35 で通常の誤りでなくなった行・新しく通常の誤りになった行（S1 の seed・改定後の数え方）\n")
    for tag, *_ in sets:
        for n_ in (17, 26):
            a, b = detail.get((tag, n_, "P33", True)), detail.get((tag, n_, "P35", True))
            if not (a and b):
                continue
            na = {x["utterance"]: x for x in a["normal_rows"]}
            nb = {x["utterance"]: x for x in b["normal_rows"]}
            gone = [f"[{na[u]['category']}]{u}" for u in na if u not in nb]
            new_ = [f"[{nb[u]['category']}]{u} → {nb[u]['outcome']}" for u in nb if u not in na]
            md.append(f"- {tag}・層 {n_}: なくなった {len(gone)}（{' / '.join(gone) or 'なし'}）／新しく {len(new_)}（{' / '.join(new_) or 'なし'}）")
            sa_, sb_ = {x["utterance"] for x in a["severe_rows"]}, {x["utterance"] for x in b["severe_rows"]}
            md.append(f"  - 重大な誤り（ツールの取り違え）: P33 {len(sa_)} → P35 {len(sb_)}（なくなった {' / '.join(sorted(sa_ - sb_)) or 'なし'}・新しく {' / '.join(sorted(sb_ - sa_)) or 'なし'}）")
            ra = sum(1 for x in a["stop_rows"] if x["stop"] == "拒否（検査）")
            rb = [x["utterance"] for x in b["stop_rows"] if x["stop"] == "拒否（検査）"]
            md.append(f"  - 対象を断った行: P33 {ra} → P35 {len(rb)}（{' / '.join(rb) or 'なし'}）")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"保存しました: {RES}")


if __name__ == "__main__":
    main()
