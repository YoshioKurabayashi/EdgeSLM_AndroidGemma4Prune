# -*- coding: utf-8 -*-
"""P23 手順 2: 整える層（`p23_arguments`）を、これまでの行で確かめる。学習なし・端末なし。

設計: DoC/設計書_P22_引数の値を測って直す.md の 7 節。`p22_fix` と同じ見方で、`p22_arguments` → `p23_arguments` を比べる。

|確かめ|中身|
|---|---|
|1 値が合っている割合|正解の引数を付けた行（評価セット v5・v6 の 248 行＋arg_eval_v1 の 240 行）で、P21・P22・P23 を比べる。**どれも直す用**（arg_eval_v1 は、P22 の結果を見て P23 を作ったので直す用になった）|
|2 変わった行|学習の履歴なしの行・評価セット v1〜v6 の操作系の対象で、P22 と値が変わった行をすべて書き出す|
|3 対象でない行|機能にない依頼・意図が不明の行が操作系と判定されたとき、引数がそろう行数（P22 → P23）|
|4 流れの KPI|評価セット v6・v5 で、引数を P21・P22・P23 に替えて比べる（値まで見た KPI も）|

    python -m app.p23_fix        # results/P23/fix.md・fix.json
"""
from __future__ import annotations

import argparse
import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p7_data import CH1, CH2, read_jsonl
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import cell
from .p18_data import CH4, TR3
from .p18_retrain import cached
from .p19_config import bert_check
from .p20_data import CH5
from .p20_verdict import load_eval5
from .p21_arguments import make_resolve as args_p21
from .p21_data import CH6
from .p21_fix import measure
from .p21_rules import make_rules as rules_p21
from .p21_verdict import WORK as P21, load_eval6
from .p22_argeval import OUT as ARG1
from .p22_arguments import make_resolve as args_p22
from .p22_fix import J, KPI, show, value_kpi
from .p22_labels import WRITE, load_all, match, scored
from .p23_arguments import make_resolve as args_p23

RES = PROJECT_ROOT / "results" / "P23"
CONFIGS = [("P21（直す前）", args_p21), ("P22（主節を先に見る）", args_p22), ("P23（整えてから渡す）", args_p23)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="26")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    use_multi_library()
    from edge_router.dispatch import arguments
    res_fn = {name: make(arguments) for name, make in CONFIGS}
    a22, a23 = res_fn[CONFIGS[1][0]], res_fn[CONFIGS[2][0]]
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    labeled = load_all()
    arg1 = [dict(r, set="arg_eval_v1", category=r["form"]) for r in read_jsonl(ARG1 / "utterances.jsonl")]
    sets = {"学習（履歴なし）": [r for r in read_jsonl(TR3 / "utterances.jsonl") if not r.get("history")], "評価セット v1": read_jsonl(CH1),
            "評価セット v2": read_jsonl(CH2 / "utterances.jsonl"), "評価セット v3": read_jsonl(CH3 / "utterances.jsonl"),
            "評価セット v4": read_jsonl(CH4 / "utterances.jsonl"), "評価セット v5": read_jsonl(CH5 / "utterances.jsonl"),
            "評価セット v6": read_jsonl(CH6 / "utterances.jsonl")}
    print("\n===== P23: 整える層を確かめる（どの行も直す用。合否ではない）=====")
    md = ["**ここの行は、どれも「直す用」。数字は合否ではない**（合否は arg_eval_v2・`results/P23/argeval.md`）\n"]
    res = {"value": {}, "changed": {}, "unknown": {}, "kpi": {}}

    md += ["\n## 1. 値が合っている割合（合っている／聞き返す／違う値で実行）\n", "|行|行数|" + "|".join(n for n, _ in CONFIGS) + "|", "|---|---|" + "---|" * len(CONFIGS)]
    print("\n===== 1. 値が合っている割合（合っている／聞き返す／違う値で実行）=====")
    for name, rows in (("評価セット v5", [r for r in labeled if r["set"] == "v5"]), ("評価セット v6", [r for r in labeled if r["set"] == "v6"]), ("arg_eval_v1", arg1)):
        cells = []
        for cname, _ in CONFIGS:
            c = collections.Counter()
            for r in rows:
                x = res_fn[cname](r["tool"], None, r["utterance"], state)
                c["合っている" if match(r["tool"], r["expected"], x.complete, x.values) else "聞き返す" if not x.complete else "違う値で実行"] += 1
            res["value"][f"{name}|{cname}"] = dict(c)
            cells.append(f"{c['合っている']}／{c['聞き返す']}／{c['違う値で実行']}")
        md.append(f"|{name}|{len(rows)}|" + "|".join(cells) + "|")
        print(f"{name}（{len(rows)} 行）: " + "  ".join(f"{cn[:3]} {x}" for (cn, _), x in zip(CONFIGS, cells)))

    md += ["\n## 2. P22 と値が変わった行（操作系の対象）\n", "|行|操作系の対象|変わった|", "|---|---|---|"]
    print("\n===== 2. 変わった行（P22 → P23）=====")
    table = []
    for name, rows in sets.items():
        known = [r for r in rows if r["action"].startswith("TOOL_") and r["action"][5:] in WRITE]
        ch = []
        for r in known:
            t = r["action"][5:]
            x, y = a22(t, None, r["utterance"], state), a23(t, None, r["utterance"], state)
            if (x.complete, scored(t, x.values)) != (y.complete, scored(t, y.values)):
                ch.append([J[t], r["utterance"], show(x), show(y)])
        res["changed"][name] = {"known": len(known), "rows": ch}
        md.append(f"|{name}|{len(known)}|{len(ch)}|")
        print(f"{name}: 操作系の対象 {len(known)} 行のうち変わった {len(ch)}" + "".join(f"\n    {c[1]}: {c[2]} → {c[3]}" for c in ch))
        table += [f"|{name}|{c[0]}|{c[1]}|{c[2]}|{c[3]}|" for c in ch]
    md += ["\n|行|機能|発話|P22|P23|", "|---|---|---|---|---|"] + table

    md += ["\n## 3. 機能にない依頼・意図が不明の行が、操作系と判定されたとき、引数がそろう行数（空調／メディア／ナビ）\n", "|行|行数|P22|P23|新しくそろう|", "|---|---|---|---|---|"]
    print("\n===== 3. 対象でない行 =====")
    gain_all = []
    for name, rows in sets.items():
        unk = [r for r in rows if r["action"] == "UNKNOWN"]
        before, after, gain = [], [], []
        for t in WRITE:
            b = {r["utterance"]: a22(t, None, r["utterance"], state) for r in unk}
            c = {r["utterance"]: a23(t, None, r["utterance"], state) for r in unk}
            before.append(sum(x.complete for x in b.values()))
            after.append(sum(x.complete for x in c.values()))
            gain += [[u, J[t], show(c[u])] for u in b if not b[u].complete and c[u].complete]
        res["unknown"][name] = {"rows": len(unk), "before": before, "after": after, "gain": gain}
        md.append(f"|{name}|{len(unk)}|" + "／".join(map(str, before)) + "|" + "／".join(map(str, after)) + f"|{len(gain)}|")
        print(f"{name}: {len(unk)} 行 / そろう行 " + "／".join(map(str, before)) + " → " + "／".join(map(str, after)) + f"（新しくそろう {len(gain)}）")
        gain_all += [f"[{name.replace('評価セット ', '')}] {g[0]}（{g[1]}と判定のとき → {g[2]}）" for g in gain]
    md.append(f"\n新しく引数がそろう、対象でない行（{len(gain_all)}）: " + (" / ".join(gain_all) if gain_all else "なし"))
    print("新しくそろう行: " + (" / ".join(gain_all) if gain_all else "なし"))

    raw = json.loads((P21 / "text_scores_F.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    md.append(f"\n## 4. 流れの KPI（ヘッド 1,584 行・BERT は {chk['one']}・規則は P21。値まで見た KPI は、違う値での実行を EtoE の成功から引き、重大な誤りに足したもの）\n")
    for tag, rows, loader in (("v6", sets["評価セット v6"], cached(load_eval6)), ("v5", sets["評価セット v5"], cached(load_eval5))):
        lab = [r for r in labeled if r["set"] == tag]
        for n in ns:
            heads = [h for h, _, _ in KPI] + ["違う値での実行（行）", "値まで見た EtoE の成功", "値まで見た重大な誤り（件）"]
            md += [f"\n### 評価セット {tag}・層 {n} 個\n", "|引数|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
            print(f"\n===== 4. {tag}・層 {n} 個 =====")
            for cname, mr in CONFIGS:
                got = measure(n, ts, loader, mr, rules_p21)
                if got is None:
                    continue
                k, d = got
                v = value_kpi(k, d, lab, mr(arguments), state, len(rows))
                cells = [cell(k[key], nd) for _, key, nd in KPI] + [f"{v['wrong_exec']:.1f}", f"{v['e2e_value']:.3f}", f"{v['severe_value_n']:.1f}（{v['severe_value'] * 100:.2f}%）"]
                res["kpi"][f"{tag}|{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, "value": v}
                print(f"{cname:<14} " + "  ".join(f"{h[:10]}: {x}" for h, x in zip(heads, cells)))
                md.append(f"|{cname}|" + "|".join(cells) + "|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "fix.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "fix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（fix.md・fix.json）")


if __name__ == "__main__":
    main()
