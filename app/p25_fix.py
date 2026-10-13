# -*- coding: utf-8 -*-
"""P25 手順 2: 歯止め（`p25_arguments`）を、これまでの行で確かめる。学習なし・端末なし。**どの行も直す用**（合否は `p25_argeval`・`p25_verdict`）。

設計: DoC/設計書_P25_引数の歯止め.md の 4 節。

|確かめ|中身|
|---|---|
|1 値が合っている割合|正解の引数を付けた行（評価セット v5・v6 の 248 行・arg_eval_v1・v2 の 480 行）で、P23 と P25 を比べる（合っている／聞き返す／違う値で実行）|
|2 止めた行|学習の履歴なしの行・評価セット v1〜v6 の操作系の対象で、歯止めが止めた行をすべて書き出す（正解の引数が無い行も。止めすぎていないかを見る）|
|3 流れの KPI|評価セット v6・v5 で、引数を P23 → P25 に替えて比べる（値まで見た KPI・不要な聞き返し）|

    python -m app.p25_fix        # results/P25/fix.md・fix.json
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
from .p21_data import CH6
from .p21_fix import measure
from .p21_rules import make_rules as rules_p21
from .p21_verdict import WORK as P21, load_eval6
from .p22_argeval import OUT as ARG1
from .p22_fix import J, KPI, show, value_kpi
from .p22_labels import WRITE, load_all, match, parse
from .p23_argeval import OUT as ARG2
from .p25_argeval import OUT as ARG3
from .p23_arguments import make_resolve as args_p23
from .p25_arguments import make_resolve as args_p25

RES = PROJECT_ROOT / "results" / "P25"
CONFIGS = [("P23（いまの引数）", args_p23), ("P25（歯止めを足す）", args_p25)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    use_multi_library()
    from edge_router.dispatch import arguments
    res_fn = {name: make(arguments) for name, make in CONFIGS}
    a23, a25 = res_fn[CONFIGS[0][0]], res_fn[CONFIGS[1][0]]
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    labeled = load_all()
    args_rows = {"arg_eval_v1": read_jsonl(ARG1 / "utterances.jsonl"), "arg_eval_v2": read_jsonl(ARG2 / "utterances.jsonl")}
    if (ARG3 / "utterances.jsonl").exists():
        args_rows["arg_eval_v3"] = read_jsonl(ARG3 / "utterances.jsonl")      # P25 の結果を見たので、P26 からは直す用
    for rows in args_rows.values():
        for r in rows:
            r["expected"] = parse(r["tool"], r["label"])
    sets = {"学習（履歴なし）": [r for r in read_jsonl(TR3 / "utterances.jsonl") if not r.get("history")], "評価セット v1": read_jsonl(CH1),
            "評価セット v2": read_jsonl(CH2 / "utterances.jsonl"), "評価セット v3": read_jsonl(CH3 / "utterances.jsonl"),
            "評価セット v4": read_jsonl(CH4 / "utterances.jsonl"), "評価セット v5": read_jsonl(CH5 / "utterances.jsonl"),
            "評価セット v6": read_jsonl(CH6 / "utterances.jsonl")}
    print("\n===== P25: 歯止めを確かめる（どの行も直す用。合否ではない）=====")
    md = ["**ここの行は、どれも「直す用」（歯止めは、これらの行を見て作った）。数字は合否ではない**（合否は新しい行で測る。P25 は arg_eval_v3、P26 は arg_eval_v4）\n"]
    res = {"value": {}, "stopped": {}, "kpi": {}}

    md += ["\n## 1. 値が合っている割合（合っている／聞き返す／違う値で実行）\n", "|行|行数|" + "|".join(n for n, _ in CONFIGS) + "|止めた行（左の引数で合っていた／違っていた）|", "|---|---|---|---|---|"]
    print("\n===== 1. 値が合っている割合（合っている／聞き返す／違う値で実行）=====")
    groups = (("評価セット v5", [r for r in labeled if r["set"] == "v5"]), ("評価セット v6", [r for r in labeled if r["set"] == "v6"]),
              *[(k, v) for k, v in args_rows.items()])
    for name, rows in groups:
        cells, cnt = [], {}
        for cname, _ in CONFIGS:
            c = collections.Counter()
            for r in rows:
                x = res_fn[cname](r["tool"], None, r["utterance"], state)
                c["合っている" if match(r["tool"], r["expected"], x.complete, x.values) else "聞き返す" if not x.complete else "違う値で実行"] += 1
            cnt[cname] = c
            res["value"][f"{name}|{cname}"] = dict(c)
            cells.append(f"{c['合っている']}／{c['聞き返す']}／{c['違う値で実行']}")
        lost_ok = sum(1 for r in rows if match(r["tool"], r["expected"], *(lambda x: (x.complete, x.values))(a23(r["tool"], None, r["utterance"], state)))
                      and not a25(r["tool"], None, r["utterance"], state).complete)
        lost_bad = cnt[CONFIGS[0][0]]["違う値で実行"] - cnt[CONFIGS[1][0]]["違う値で実行"]
        md.append(f"|{name}|{len(rows)}|" + "|".join(cells) + f"|{lost_ok}／{lost_bad}|")
        print(f"{name}（{len(rows)} 行）: " + "  ".join(f"{cn[:3]} {x}" for (cn, _), x in zip(CONFIGS, cells)) + f"  止めた: 合っていた {lost_ok}・違っていた {lost_bad}")

    md += ["\n## 2. 歯止めが止めた行（操作系の対象。正解の引数が無い行も含む）\n", "|行|操作系の対象|左の引数でそろう|右の引数が止めた|", "|---|---|---|---|"]
    print("\n===== 2. 歯止めが止めた行 =====")
    table, why = [], collections.Counter()
    for name, rows in sets.items():
        known = [r for r in rows if r["action"].startswith("TOOL_") and r["action"][5:] in WRITE]
        full, stop = 0, []
        for r in known:
            t = r["action"][5:]
            x = a23(t, None, r["utterance"], state)
            if not x.complete:
                continue
            full += 1
            y = a25(t, None, r["utterance"], state)
            if not y.complete:
                g = y.notes[0].split(":")[1].strip()
                why[g] += 1
                stop.append([J[t], r["utterance"], show(x), y.notes[0].replace("P25: ", "").replace("P26: ", "").replace(" → 聞き返す", "")])
        res["stopped"][name] = {"known": len(known), "complete": full, "rows": stop}
        md.append(f"|{name}|{len(known)}|{full}|{len(stop)}|")
        print(f"{name}: 対象 {len(known)}・そろう {full}・止めた {len(stop)}" + "".join(f"\n    {s[1]}: {s[2]} … {s[3]}" for s in stop))
        table += [f"|{name}|{s[0]}|{s[1]}|{s[2]}|{s[3]}|" for s in stop]
    md += ["\n|行|機能|発話|左の引数の値|止めた理由|", "|---|---|---|---|---|"] + table
    md.append("\n止めた理由の内訳: " + "・".join(f"{k} {v} 行" for k, v in sorted(why.items())))

    raw = json.loads((P21 / "text_scores_F.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    md.append(f"\n## 3. 流れの KPI（ヘッド 1,584 行・seed 3 つの平均・BERT は {chk['one']}・規則は P21。値まで見た KPI は、違う値での実行を EtoE の成功から引き、重大な誤りに足したもの）\n")
    for tag, rows, loader in (("v6", sets["評価セット v6"], cached(load_eval6)), ("v5", sets["評価セット v5"], cached(load_eval5))):
        lab = [r for r in labeled if r["set"] == tag]
        for n in ns:
            heads = [h for h, _, _ in KPI] + ["違う値での実行（行）", "値まで見た EtoE の成功", "値まで見た重大な誤り（件）"]
            md += [f"\n### 評価セット {tag}・層 {n} 個\n", "|引数|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
            print(f"\n===== 3. {tag}・層 {n} 個 =====")
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
