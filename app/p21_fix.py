# -*- coding: utf-8 -*-
"""P21 手順 1: 規則と引数の直し（`p21_rules`・`p21_arguments`）を、これまでの行で確かめる。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P21_規則と引数の誤りを直す.md の 5 節。

|確かめ|中身|
|---|---|
|1 規則（1 行ずつ）|学習の履歴なしの行・評価セット v1〜v5 で、直す前（`p19_rules`）と後（`p21_rules`）を比べる。対象の行は正しい判定のとき止まるか。機能にない依頼・意図が不明の行は、操作系と判定されたとき止まるか|
|2 引数（1 行ずつ）|対象の行で、直す前（`p17_arguments`）と後（`p21_arguments`）の取り出しを比べる。変わった行はすべて書き出す|
|3 流れの KPI|P19 で決めた構成（層 26 ほか・BERT 1 つ）で、直す前と後を評価セット v5・v4・v3 で比べる|

**評価セット v3・v4・v5 は「直す用」。ここの数字は合否ではない**（合否は `p21_verdict` で、評価セット v6 で測る）。

    python -m app.p21_fix        # results/P21/fix.md・fix.json
"""
from __future__ import annotations

import argparse
import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p7_data import CH1, CH2, read_jsonl
from .p8_gate import STEPS, run
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import cell, hi_rule, load_eval3
from .p17_arguments import make_resolve as args_p17
from .p18_analyze import hook1584
from .p18_data import CH4, TR3
from .p18_retrain import SRC1584, cached, has_heads, load_eval4
from .p19_config import bert_check
from .p19_rules import make_rules as rules_p19
from .p20_data import CH5
from .p20_verdict import WORK as P20, load_eval5
from .p21_arguments import make_resolve as args_p21
from .p21_rules import make_rules as rules_p21

RES = PROJECT_ROOT / "results" / "P21"
WRITE = ("TOOL_control_hvac", "TOOL_control_media", "TOOL_set_navigation")
J = {"TOOL_control_hvac": "空調", "TOOL_control_media": "メディア", "TOOL_set_navigation": "ナビ"}
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3)]
CONFIGS = [("直す前（P19 で決めた構成）", args_p17, rules_p19), ("規則だけ直す", args_p17, rules_p21), ("引数だけ直す", args_p21, rules_p19), ("両方直す（P21）", args_p21, rules_p21)]


def measure(n: int, ts: dict, loader, make_resolve, make_rules):
    if not has_heads(n, SRC1584) or loader(n) is None:
        return None
    final = STEPS[-1]
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=hook1584(0.02), steps=[final], sources=[SRC1584], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=make_rules, amb_chat_ok=True)
    key = f"{n}|{SRC1584}|{final[0]}"
    return blob["kpi"][key], blob["detail"][key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    use_multi_library()
    from edge_router.dispatch import arguments, rules
    r19, r21 = rules_p19(rules), rules_p21(rules)
    a17, a21 = args_p17(arguments), args_p21(arguments)
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    sets = {"学習（履歴なし）": [r for r in read_jsonl(TR3 / "utterances.jsonl") if not r.get("history")], "評価セット v1": read_jsonl(CH1),
            "評価セット v2": read_jsonl(CH2 / "utterances.jsonl"), "評価セット v3": read_jsonl(CH3 / "utterances.jsonl"),
            "評価セット v4": read_jsonl(CH4 / "utterances.jsonl"), "評価セット v5": read_jsonl(CH5 / "utterances.jsonl")}
    print("\n===== P21: 規則と引数の直しを確かめる（評価セット v3〜v5 は直す用。合否ではない）=====")
    md = ["**評価セット v3・v4・v5 は「直す用」。ここの数字は合否ではない**（合否は評価セット v6・`results/P21/verdict.md`）\n"]
    res = {"rules": {}, "args": {}, "kpi": {}, "stops": {}, "severe": {}}
    row = lambda r: {"utterance": r["utterance"], "history": [], "last_tool": None}

    # ---- 1. 規則
    md += ["\n## 1. 規則（直す前 `p19_rules` → 後 `p21_rules`）\n",
           "|行|対象の行|規則が止める対象の行（正しい判定のとき）|通るようになった|新しく止まる|機能にない依頼・意図が不明の行|操作系と判定されたとき規則が止める（空調／メディア／ナビ）|止まらなくなった|", "|---|---|---|---|---|---|---|---|"]
    print("\n===== 1. 規則 =====")
    for name, rows in sets.items():
        known = [r for r in rows if r["action"].startswith("TOOL_")]
        unk = [r for r in rows if r["action"] == "UNKNOWN"]
        b = {r["id"]: r19(row(r), r["action"]) for r in known}
        c = {r["id"]: r21(row(r), r["action"]) for r in known}
        freed = [[r["utterance"], b[r["id"]].rule] for r in known if b[r["id"]].demote and not c[r["id"]].demote]
        newly = [[r["utterance"], c[r["id"]].rule] for r in known if not b[r["id"]].demote and c[r["id"]].demote]
        still = [[r["utterance"], c[r["id"]].rule] for r in known if c[r["id"]].demote]
        per, lost, gain = [], [], []
        for t in WRITE:
            nb, na = sum(r19(row(r), t).demote for r in unk), sum(r21(row(r), t).demote for r in unk)
            per.append(f"{nb} → {na}")
            lost += [[r["utterance"], J[t]] for r in unk if r19(row(r), t).demote and not r21(row(r), t).demote]
            gain += [[r["utterance"], J[t], r21(row(r), t).rule] for r in unk if not r19(row(r), t).demote and r21(row(r), t).demote]
        res["rules"][name] = {"known": len(known), "stopped_before": sum(d.demote for d in b.values()), "stopped_after": len(still), "freed": freed, "newly": newly,
                              "still": still, "unknown": len(unk), "unknown_stopped": per, "unknown_lost": lost, "unknown_gain": gain}
        x = res["rules"][name]
        print(f"{name}: 対象 {len(known)} 行のうち規則が止める {x['stopped_before']} → {x['stopped_after']}（通るようになった {len(freed)}・新しく止まる {len(newly)}）/ "
              f"機能にない依頼・意図が不明 {len(unk)} 行を止める: " + "／".join(per) + f"（止まらなくなった {len(lost)}）")
        md.append(f"|{name}|{len(known)}|{x['stopped_before']} → {x['stopped_after']}|{len(freed)}|{len(newly)}|{len(unk)}|" + "／".join(per) + f"|{len(lost)}|")
    for title, key, f in (("通るようになった対象の行", "freed", lambda v: f"{v[0]}（{v[1]}）"), ("新しく止まる対象の行", "newly", lambda v: f"{v[0]}（{v[1]}）"),
                          ("新しく止まる、機能にない依頼・意図が不明の行", "unknown_gain", lambda v: f"{v[0]}（{v[1]}と判定のとき・{v[2]}）"),
                          ("止まらなくなった、機能にない依頼・意図が不明の行", "unknown_lost", lambda v: f"{v[0]}（{v[1]}と判定のとき）")):
        items = [f"[{n.replace('評価セット ', '')}] " + f(v) for n, x in res["rules"].items() for v in x[key]]
        md.append(f"\n{title}（{len(items)}）: " + (" / ".join(items) if items else "なし"))
    left = [f"[{n.replace('評価セット ', '')}] {u}（{r_}）" for n, x in res["rules"].items() if n != "学習（履歴なし）" for u, r_ in x["still"]]
    md.append(f"\n直した後も、正しい判定で規則が止める評価セットの対象の行（{len(left)}）: " + (" / ".join(left) if left else "なし"))

    # ---- 2. 引数
    md += ["\n## 2. 引数（直す前 `p17_arguments` → 後 `p21_arguments`）\n", "|行|対象の行|引数が足りない行|変わった行|", "|---|---|---|---|"]
    print("\n===== 2. 引数 =====")
    changed_all = []
    for name, rows in sets.items():
        known = [r for r in rows if r["action"].startswith("TOOL_")]
        nb = na = 0
        changed, short = [], []
        for r in known:
            t = r["action"][5:]
            x, y = a17(t, None, r["utterance"], state), a21(t, None, r["utterance"], state)
            nb += not x.complete
            na += not y.complete
            if (x.complete, x.values) != (y.complete, y.values):
                changed.append([r["utterance"], t, json.dumps(x.values, ensure_ascii=False) if x.complete else "足りない", json.dumps(y.values, ensure_ascii=False) if y.complete else "足りない"])
            if not y.complete:
                short.append(r["utterance"])
        res["args"][name] = {"known": len(known), "short_before": nb, "short_after": na, "changed": changed, "short": short}
        changed_all += [[name] + c for c in changed]
        print(f"{name}: 対象 {len(known)} 行のうち引数が足りない {nb} → {na} / 変わった {len(changed)}")
        md.append(f"|{name}|{len(known)}|{nb} → {na}|{len(changed)}|")
    md += ["\n引数が変わった行（すべて）:\n", "|行|発話|機能|直す前|直した後|", "|---|---|---|---|---|"] + [f"|{c[0]}|{c[1]}|{c[2]}|{c[3]}|{c[4]}|" for c in changed_all]
    md.append("\n直した後も引数が足りない、評価セット v5 の対象の行: " + " / ".join(res["args"]["評価セット v5"]["short"]))

    # ---- 3. 流れの KPI
    raw = json.loads((P20 / "text_scores_F.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    evals = [("v5", read_jsonl(CH5 / "utterances.jsonl"), cached(load_eval5)), ("v4", read_jsonl(CH4 / "utterances.jsonl"), cached(load_eval4)),
             ("v3", read_jsonl(CH3 / "utterances.jsonl"), cached(load_eval3))]
    md.append(f"\n## 3. 流れの KPI（ヘッド 1,584 行・BERT は {chk['one']}・棄却の閾値 2%。ヘッドの seed 3 つの平均）\n")
    for tag, rows, loader in evals:
        n_known = sum(r["action"] != "UNKNOWN" for r in rows)
        for n in ns:
            print(f"\n===== 3. {tag}・層 {n} 個 =====")
            heads = [lab for lab, _, _ in KPI] + ["対象を止めた段（規則／引数／棄却／確信度／BERT フラグ）", "重大の内訳（機能にない依頼／意図が不明／雑談／対象）"]
            md += [f"\n### 評価セット {tag}・層 {n} 個（対象の依頼と雑談 {n_known} 行）\n", "|構成|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
            for cname, mr, mk in CONFIGS:
                got = measure(n, ts, loader, mr, mk)
                if got is None:
                    continue
                k, d = got
                st = collections.Counter()
                for x in d["stop_rows"]:
                    st[x["stop"]] += x["share"]
                sev = collections.Counter()
                for x in d["by_group"]:
                    if x["grade"] == "重大":
                        sev[x["category"]] += x["rows"]
                stops = [st["聞き返し（規則）"], st["聞き返し（引数が足りない）"], st["聞き返し（棄却ヘッド）"], st["聞き返し（確信度）"], st["聞き返し（BERT のフラグ）"]]
                cells = [cell(k[key], nd) for _, key, nd in KPI] + ["／".join(f"{x:.1f}" for x in stops), f"{sev['H']:.1f}／{sev['G']:.1f}／{sev['F']:.1f}／{sev['K'] + sev['N']:.1f}"]
                res["kpi"][f"{tag}|{n}|{cname}"] = {key: k[key] for _, key, _ in KPI}
                res["stops"][f"{tag}|{n}|{cname}"] = stops
                res["severe"][f"{tag}|{n}|{cname}"] = dict(sev)
                print(f"{cname:<20} " + "  ".join(f"{h[:10]}: {x}" for h, x in zip(heads, cells)))
                md.append(f"|{cname}|" + "|".join(cells) + "|")
                if cname == CONFIGS[-1][0] and tag == "v5":
                    md += [f"\n両方直した後に残る重大な誤り（v5・層 {n}）: " + " / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:20]),
                           f"\n両方直した後に止めた対象の行（v5・層 {n}）: " + " / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:40])]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "fix.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "fix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（fix.md・fix.json）")


if __name__ == "__main__":
    main()
