# -*- coding: utf-8 -*-
"""P31: 仕上げ（範囲外は断る・語の一覧を機能ごとに・対象の言い方の取りこぼし・載せる seed の決め方）の当たり。端末なし・学習なし。

設計: DoC/設計書_P31_仕上げ.md の 5 節。**評価セット v6〜v8 はどれも直す用。ここの数字は当たりで、合否ではない**（合否は v9）。

|構成|規則|引数|範囲外|
|---|---|---|---|
|P28（比べる相手）|P21|P26|絞らない|
|P30|P21|P30（単語の一致で絞る）|聞き返す|
|**P31（本命の候補）**|P31（「〜に戻って」）|P31（直し F1〜F6）|**断る**|

載せる seed の決め方（S1）: 学習データの validation の対象の行で、**ヘッドが通す割合（棄却されず・確信度が閾値以上）が一番大きい seed**。評価セットは見ない。

    python -m app.p31_check        # results/P31/check.md・check.json
"""
from __future__ import annotations

import collections
import json

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from . import p18_retrain as R
from .p6_seeds import SEEDS, head_probs, load_pt
from .p7_data import read_jsonl
from .p8_gate import STEPS, run
from .p11_bert import MODES, by_mode
from .p13_eval import cell, hi_rule
from .p18_retrain import HEADS, cached, has_heads
from .p19_config import bert_check
from .p21_data import CH6
from .p21_rules import make_rules as rules_p21
from .p21_verdict import STOPS, load_eval6, share_by
from .p22_fix import value_kpi
from .p22_labels import load_all
from .p26_arguments import make_resolve as args_p26
from .p27_data import CH7, labeled as labeled7
from .p27_verdict import load_eval7
from .p28_data import CH8, TR4, labeled as labeled8
from .p28_verdict import SRC1656, WORK as P28, load_eval8
from .p30_scope import make_resolve as args_p30
from .p31_arguments import make_resolve as args_p31
from .p31_rules import make_rules as rules_p31

RES = PROJECT_ROOT / "results" / "P31"
CONFIGS = [("P28（P26・絞らない）", args_p26, rules_p21), ("P30（単語の一致で絞る・聞き返す）", args_p30, rules_p21), ("P31（仕上げ・断る）", args_p31, rules_p31)]
GRADES = ("正解", "回復可能", "通常", "重大")


def pick_seed(n: int) -> tuple[int, dict]:
    """S1: validation の対象の行で、ヘッドが通す割合が一番大きい seed"""
    use_multi_library()
    from edge_router.dispatch import policy
    effect = {k: v.get("effect", "read") for k, v in policy.load_tools().items()}
    rows = read_jsonl(TR4 / "utterances.jsonl")
    pick = [i for i, r in enumerate(rows) if r["split"] == "validation" and r["route"] != "UNKNOWN"]
    z = np.load(P28 / f"L{n:02d}" / "features" / "features_F2.npz")
    last, mean = z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)
    gold = np.array([rows[i]["action"] for i in pick])
    d = HEADS[SRC1656](n)
    reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
    heads = []
    for s in SEEDS:
        name = f"heads_mlp_seed{s}.pt"
        blob = load_pt(d / name)
        heads.append((s, blob["metadata"]["action_classes"], {k: float(v) for k, v in reps[name]["thresholds"].items()}, head_probs(blob, float(reps[name]["temperature"]), last, mean)))
    heads = R.hook_for(SRC1656)(n, SRC1656, heads)
    stat = {}
    for s, classes, th, p in heads:
        pred = np.array([classes[i] for i in p["action"].argmax(1)])
        conf = p["action"].max(1)
        tau = np.array([th["CHAT"] if c == "CHAT" else th["TOOL_write" if effect.get(c[5:]) == "write" else "TOOL_read"] for c in pred])
        ok = (p["reject"] < th["REJECT"]) & (conf >= tau)
        stat[s] = {"cover": float(ok.mean()), "acc": float((pred == gold).mean()), "rows": len(pick)}
    best = max(stat, key=lambda s: (stat[s]["cover"], stat[s]["acc"]))
    return best, stat


def measure(n: int, ts: dict, loader, make_resolve, make_rules, seed: int | None):
    if not has_heads(n, SRC1656) or loader(n) is None:
        return None
    base = R.hook_for(SRC1656)
    hook = base if seed is None else (lambda n_, src, heads: [h for h in base(n_, src, heads) if h[0] == seed])
    final = STEPS[-1]
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=hook, steps=[final], sources=[SRC1656], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=make_rules, amb_chat_ok=True)
    key = f"{n}|{SRC1656}|{final[0]}"
    return blob["kpi"][key], blob["detail"][key]


def groups(d) -> collections.Counter:
    g = collections.Counter()
    for x in d["by_group"]:
        key = "H unseen" if x["category"] == "H" and x["group"] == "unseen" else "H seen" if x["category"] == "H" else x["category"]
        g[(key, x["grade"])] += x["rows"]
    return g


def main() -> None:
    raw = json.loads((P28 / "text_scores_F.json").read_text(encoding="utf-8"))
    use_multi_library()
    from edge_router.dispatch import arguments
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    ts = by_mode(raw, MODES[0], only=(bert_check(raw)["one"],))
    md = ["# P31 仕上げの当たり（端末なし・学習なし）\n", "**評価セット v6〜v8 はどれも直す用。ここの数字は当たりで、合否ではない**。ヘッド 1,656 行（P28）・BERT は bert_s0。"
          "「断った」= 範囲外を断った行（検査で断る）。対象の行を断ったら「対象を断った」に数え、通常の誤りになる（不要な聞き返しには入らない）\n"]
    res = {"seed": {}, "kpi": {}, "groups": {}, "changed": {}}
    seeds = {}
    md += ["## 1. 載せる seed の決め方（S1: validation の対象の行で、ヘッドが通す割合が一番大きい seed）\n", "|層|seed ごとの通す割合（validation）|seed ごとの正解率（validation）|選んだ seed|", "|---|---|---|---|"]
    for n in (17, 26):
        best, stat = pick_seed(n)
        seeds[n] = best
        res["seed"][str(n)] = {"best": best, "stat": stat}
        md.append(f"|{n}|" + "・".join(f"{s}: {v['cover']:.3f}" for s, v in stat.items()) + "|" + "・".join(f"{s}: {v['acc']:.3f}" for s, v in stat.items()) + f"|**{best}**|")
        print(f"層 {n}: seed ごとの通す割合 " + "・".join(f"{s}: {v['cover']:.3f}" for s, v in stat.items()) + f" → seed {best}", flush=True)
    sets = (("v8", read_jsonl(CH8 / "utterances.jsonl"), cached(load_eval8), labeled8()),
            ("v7", read_jsonl(CH7 / "utterances.jsonl"), cached(load_eval7), labeled7()),
            ("v6", read_jsonl(CH6 / "utterances.jsonl"), cached(load_eval6), [x for x in load_all() if x["set"] == "v6"]))
    md.append("\n## 2. 流れの KPI\n")
    for tag, rows, loader, lab in sets:
        for n in (17, 26):
            keep = {}
            for cname, make, mrules in CONFIGS:
                for sd in (None, seeds[n]):
                    got = measure(n, ts, loader, make, mrules, sd)
                    if got:
                        k, d = got
                        keep[(cname, sd)] = (k, d, value_kpi(k, d, lab, make(arguments), state, len(rows)))
            heads_ = ["値まで見た EtoE", "値まで見た重大な誤り（件）", "不要な聞き返し", "通常の誤り", "対象を断った（件）"] + [f"止めた: {s_.replace('聞き返し（', '').replace('拒否（', '').rstrip('）')}" for s_ in STOPS] + ["断った（範囲外）"]
            md += [f"\n### {tag}・層 {n}\n", "|構成|seed|" + "|".join(heads_) + "|", "|---|---|" + "---|" * len(heads_)]
            for (cname, sd), (k, d, v) in keep.items():
                cnt = collections.Counter()
                for x in d["stop_rows"]:
                    cnt[x["stop"]] += x["share"]
                sname = "3 つの平均" if sd is None else f"{sd}（S1 で選んだ）"
                res["kpi"][f"{tag}|{n}|{cname}|{sname}"] = {"e2e_value": v["e2e_value"], "severe_value_n": v["severe_value_n"], "over": k["over"], "normal": k["normal"], "known_refused_n": k["known_refused_n"], "wrong_rows": v["rows"]}
                md.append(f"|{cname}|{sname}|{v['e2e_value']:.3f}|{v['severe_value_n']:.1f}|{cell(k['over'], 3)}|{cell(k['normal'], 3)}|{k['known_refused_n']['mean']:.1f}|" + "|".join(f"{cnt[s_]:.1f}" for s_ in STOPS) + f"|{cnt['拒否（検査）']:.1f}|")
                print(f"{tag}・層 {n}・{cname[:12]:<12}・{sname:<12} EtoE {v['e2e_value']:.3f}・重大 {v['severe_value_n']:.1f}・聞き返し {k['over']['mean']:.3f}・通常 {k['normal']['mean']:.3f}・対象を断った {k['known_refused_n']['mean']:.1f}", flush=True)
            a, b = keep[(CONFIGS[1][0], None)][1], keep[(CONFIGS[2][0], None)][1]
            ga, gb = groups(keep[(CONFIGS[0][0], None)][1]), groups(b)
            res["groups"][f"{tag}|{n}"] = {"P28": {f"{x}|{y}": w for (x, y), w in ga.items()}, "P31": {f"{x}|{y}": w for (x, y), w in gb.items()}}
            md += [f"\n種類ごとの扱い（{tag}・層 {n}・3 つの平均。P28 → P31。正解／回復可能／通常／重大）:\n", "|種類|P28|P31|", "|---|---|---|"]
            for key in ("H seen", "H unseen", "G", "K", "N", "F"):
                md.append(f"|{key}|" + "／".join(f"{ga[(key, g)]:.1f}" for g in GRADES) + "|" + "／".join(f"{gb[(key, g)]:.1f}" for g in GRADES) + "|")
            fmt = lambda items: " / ".join(f"{u}（{s:.2f}）" for u, s in items) if items else "なし"
            dif = lambda x, y: [[u, round(x[u] - y[u], 2)] for u in x if x[u] - y[u] > 1e-9]
            sa, sb = share_by(a["stop_rows"]), share_by(b["stop_rows"])
            refused = [[x["utterance"], round(x["share"], 2)] for x in b["stop_rows"] if x["stop"] == "拒否（検査）"]
            na, nb = share_by(a["normal_rows"]), share_by(b["normal_rows"])
            va, vb = share_by(a["severe_rows"]), share_by(b["severe_rows"])
            ch = {"stop_freed": dif(sa, sb), "stop_new": dif(sb, sa), "known_refused": refused, "normal_fixed": dif(na, nb), "normal_new": dif(nb, na), "severe_fixed": dif(va, vb), "severe_new": dif(vb, va)}
            res["changed"][f"{tag}|{n}"] = ch
            md += [f"\nP30 → P31 で変わった行（{tag}・層 {n}・3 つの平均）:\n", "|見るもの|行|", "|---|---|"]
            for lab_, key in (("止まらなくなった対象の行", "stop_freed"), ("新しく止まった・断った対象の行", "stop_new"), ("断った対象の行（範囲外と判断）", "known_refused"),
                              ("通常の誤りでなくなった行", "normal_fixed"), ("新しく通常の誤りになった行", "normal_new"), ("重大な誤りでなくなった行", "severe_fixed"), ("新しく重大な誤りになった行", "severe_new")):
                md.append(f"|{lab_}（{sum(s for _, s in ch[key]):.1f}）|{fmt(ch[key])}|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（check.md・check.json）")


if __name__ == "__main__":
    main()
