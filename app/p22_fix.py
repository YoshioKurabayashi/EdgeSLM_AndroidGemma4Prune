# -*- coding: utf-8 -*-
"""P22 手順 2: 引数の値の直し（`p22_arguments`）を、これまでの行で確かめる。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P22_引数の値を測って直す.md の 5 節。

|確かめ|中身|
|---|---|
|1 値が合っている割合|正解の引数を付けた行（評価セット v5・v6 の操作系の対象 248 行）で、直す前（`p21_arguments`）と後（`p22_arguments`）を比べる|
|2 変わった行|学習の履歴なしの行・評価セット v1〜v6 の操作系の対象で、取り出した値が変わった行をすべて書き出す（正解の無い行は、目で見る）|
|3 対象でない行|機能にない依頼・意図が不明の行が、空調・メディア・ナビと判定されたとき、引数がそろう（＝この段で止まらない）行数。直しで増えていないか|
|4 流れの KPI|P21 で決めた構成で、引数だけを直す前と後に替えて、評価セット v6・v5・v4・v3 で比べる。v5・v6 は、値まで見た KPI も出す|

**値まで見た KPI**: 正しいツールを実行した行のうち、値が正解と違う行を「違う値での実行」として数え、EtoE の成功から引き、重大な誤りに足す（操作系の誤実行なので）。

**評価セット v3〜v6 は「直す用」。ここの数字は合否ではない**（合否は `p22_verdict` で、評価セット v7 で測る）。

    python -m app.p22_fix        # results/P22/fix.md・fix.json
"""
from __future__ import annotations

import argparse
import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p7_data import CH1, CH2, read_jsonl
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import cell, load_eval3
from .p18_data import CH4, TR3
from .p18_retrain import cached, load_eval4
from .p19_config import bert_check
from .p20_data import CH5
from .p20_verdict import load_eval5
from .p21_arguments import make_resolve as args_p21
from .p21_data import CH6
from .p21_fix import measure
from .p21_rules import make_rules as rules_p21
from .p21_verdict import WORK as P21, load_eval6
from .p22_arguments import make_resolve as args_p22
from .p22_labels import WRITE, load_all, match, scored

RES = PROJECT_ROOT / "results" / "P22"
J = {"control_hvac": "空調", "control_media": "メディア", "set_navigation": "ナビ"}
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3)]
CONFIGS = [("直す前（P21 の構成）", args_p21), ("引数の値を直す（P22）", args_p22)]


def show(x) -> str:
    return json.dumps(x.values, ensure_ascii=False) if x.complete else "聞き返す"


def value_kpi(k: dict, d: dict, labeled: list[dict], resolve, state: dict, n_rows: int) -> dict:
    """値まで見た KPI。正しいツールを実行した割合（1 − 止めた・誤った割合）のうち、値が正解と違う行を数える"""
    lost = collections.Counter()
    for key in ("stop_rows", "normal_rows", "severe_rows"):
        for x in d[key]:
            lost[x["utterance"]] += x["share"]
    wrong, rows = 0.0, []
    for r in labeled:
        x = resolve(r["tool"], None, r["utterance"], state)
        if x.complete and not match(r["tool"], r["expected"], True, x.values):
            share = max(0.0, 1.0 - lost[r["utterance"]])
            wrong += share
            if share > 1e-9:
                rows.append({"utterance": r["utterance"], "tool": r["tool"], "got": show(x), "label": r["label"], "share": round(share, 2)})
    return {"wrong_exec": wrong, "e2e_value": k["e2e"]["mean"] - wrong / n_rows, "severe_value_n": k["severe_n"]["mean"] + wrong,
            "severe_value": (k["severe_n"]["mean"] + wrong) / n_rows, "first_value": k["first"]["mean"] - wrong / n_rows, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    use_multi_library()
    from edge_router.dispatch import arguments
    a21, a22 = args_p21(arguments), args_p22(arguments)
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    labeled = load_all()
    sets = {"学習（履歴なし）": [r for r in read_jsonl(TR3 / "utterances.jsonl") if not r.get("history")], "評価セット v1": read_jsonl(CH1),
            "評価セット v2": read_jsonl(CH2 / "utterances.jsonl"), "評価セット v3": read_jsonl(CH3 / "utterances.jsonl"),
            "評価セット v4": read_jsonl(CH4 / "utterances.jsonl"), "評価セット v5": read_jsonl(CH5 / "utterances.jsonl"),
            "評価セット v6": read_jsonl(CH6 / "utterances.jsonl")}
    print("\n===== P22: 引数の値の直しを確かめる（評価セット v3〜v6 は直す用。合否ではない）=====")
    md = ["**評価セット v3〜v6 は「直す用」。ここの数字は合否ではない**（合否は評価セット v7・`results/P22/verdict.md`）\n"]
    res = {"value": {}, "changed": {}, "unknown": {}, "kpi": {}}

    # ---- 1. 値が合っている割合
    md += ["\n## 1. 値が合っている割合（正解の引数を付けた行。直す前 `p21_arguments` → 後 `p22_arguments`）\n",
           "|行|" + "|".join(J.values()) + "|機能の語を含む依頼（前置きつき）|ふつうの依頼|合計|", "|---|" + "---|" * (len(J) + 3)]
    print("\n===== 1. 値が合っている割合 =====")
    wrong_after = []
    for name in ("v5", "v6"):
        sub = [r for r in labeled if r["set"] == name]
        ok = {"前": collections.Counter(), "後": collections.Counter()}
        n = collections.Counter()
        for r in sub:
            x, y = a21(r["tool"], None, r["utterance"], state), a22(r["tool"], None, r["utterance"], state)
            for key in (r["tool"], r["category"], "all"):
                n[key] += 1
                ok["前"][key] += match(r["tool"], r["expected"], x.complete, x.values)
                ok["後"][key] += match(r["tool"], r["expected"], y.complete, y.values)
            if not match(r["tool"], r["expected"], y.complete, y.values):
                wrong_after.append([name, J[r["tool"]], r["utterance"], show(y), r["label"]])
        keys = list(J) + ["K", "N", "all"]
        res["value"][name] = {key: {"rows": n[key], "before": ok["前"][key], "after": ok["後"][key]} for key in keys}
        md.append(f"|評価セット {name}|" + "|".join(f"{ok['前'][key]} → {ok['後'][key]}／{n[key]}" for key in keys) + "|")
        print(f"評価セット {name}: " + " / ".join(f"{J.get(key, {'K': '前置きつき', 'N': 'ふつう', 'all': '合計'}.get(key))} {ok['前'][key]}→{ok['後'][key]}/{n[key]}" for key in keys))
    tot = {s: sum(res["value"][nm]["all"][s] for nm in res["value"]) for s in ("rows", "before", "after")}
    md.append(f"\n合計 {tot['rows']} 行: {tot['before']}（{tot['before'] / tot['rows']:.3f}）→ {tot['after']}（{tot['after'] / tot['rows']:.3f}）。"
              "**v5・v6 を見ながら直したので、後の数字は当然高い**")
    md.append("\n直した後も正解と違う行: " + (" / ".join(f"[{c[0]}・{c[1]}] {c[2]} → {c[3]}（正解 {c[4]}）" for c in wrong_after) if wrong_after else "なし"))
    res["value"]["wrong_after"] = wrong_after

    # ---- 2. 変わった行
    md += ["\n## 2. 取り出した値が変わった行（操作系の対象。すべて書き出す）\n", "|行|操作系の対象|変わった|聞き返す → 値が出る|値が出る → 聞き返す|", "|---|---|---|---|---|"]
    print("\n===== 2. 変わった行 =====")
    table = []
    for name, rows in sets.items():
        known = [r for r in rows if r["action"].startswith("TOOL_") and r["action"][5:] in WRITE]
        ch = []
        for r in known:
            t = r["action"][5:]
            x, y = a21(t, None, r["utterance"], state), a22(t, None, r["utterance"], state)
            if (x.complete, scored(t, x.values)) != (y.complete, scored(t, y.values)):
                ch.append({"tool": J[t], "utterance": r["utterance"], "before": show(x), "after": show(y), "a2v": (not x.complete) and y.complete, "v2a": x.complete and not y.complete})
        res["changed"][name] = {"known": len(known), "rows": ch}
        md.append(f"|{name}|{len(known)}|{len(ch)}|{sum(c['a2v'] for c in ch)}|{sum(c['v2a'] for c in ch)}|")
        print(f"{name}: 操作系の対象 {len(known)} 行のうち変わった {len(ch)}（聞き返す→値 {sum(c['a2v'] for c in ch)}・値→聞き返す {sum(c['v2a'] for c in ch)}）")
        table += [f"|{name}|{c['tool']}|{c['utterance']}|{c['before']}|{c['after']}|" for c in ch]
    md += ["\n|行|機能|発話|直す前|直した後|", "|---|---|---|---|---|"] + table

    # ---- 3. 対象でない行
    md += ["\n## 3. 機能にない依頼・意図が不明の行が、操作系と判定されたとき、引数がそろう行数（空調／メディア／ナビ）\n",
           "規則・BERT・棄却が止めなかったとき、この段でも止まらない行。直しで増えると、重大な誤りが増えるおそれがある\n", "|行|機能にない依頼・意図が不明|直す前|直した後|新しくそろう|そろわなくなる|", "|---|---|---|---|---|---|"]
    print("\n===== 3. 対象でない行 =====")
    gain_all = []
    for name, rows in sets.items():
        unk = [r for r in rows if r["action"] == "UNKNOWN"]
        before, after, gain, lose = [], [], [], []
        for t in WRITE:
            b = {r["utterance"]: a21(t, None, r["utterance"], state) for r in unk}
            c = {r["utterance"]: a22(t, None, r["utterance"], state) for r in unk}
            before.append(sum(x.complete for x in b.values()))
            after.append(sum(x.complete for x in c.values()))
            gain += [[u, J[t], show(c[u])] for u in b if not b[u].complete and c[u].complete]
            lose += [[u, J[t]] for u in b if b[u].complete and not c[u].complete]
        res["unknown"][name] = {"rows": len(unk), "before": before, "after": after, "gain": gain, "lose": lose}
        md.append(f"|{name}|{len(unk)}|" + "／".join(map(str, before)) + "|" + "／".join(map(str, after)) + f"|{len(gain)}|{len(lose)}|")
        print(f"{name}: {len(unk)} 行 / そろう行 " + "／".join(map(str, before)) + " → " + "／".join(map(str, after)) + f"（新しくそろう {len(gain)}・そろわなくなる {len(lose)}）")
        gain_all += [f"[{name.replace('評価セット ', '')}] {g[0]}（{g[1]}と判定のとき → {g[2]}）" for g in gain]
    md.append(f"\n新しく引数がそろう、対象でない行（{len(gain_all)}）: " + (" / ".join(gain_all) if gain_all else "なし"))

    # ---- 4. 流れの KPI
    raw = json.loads((P21 / "text_scores_F.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    evals = [("v6", sets["評価セット v6"], cached(load_eval6)), ("v5", sets["評価セット v5"], cached(load_eval5)),
             ("v4", sets["評価セット v4"], cached(load_eval4)), ("v3", sets["評価セット v3"], cached(load_eval3))]
    md.append(f"\n## 4. 流れの KPI（ヘッド 1,584 行・BERT は {chk['one']}・規則は P21・棄却の閾値 2%。ヘッドの seed 3 つの平均）\n")
    md.append("「値まで見た」の列は、正しいツールを実行したが値が正解と違う行を、EtoE の成功から引き、重大な誤りに足したもの（正解の引数を付けた v5・v6 だけ）\n")
    for tag, rows, loader in evals:
        lab = [r for r in labeled if r["set"] == tag]
        for n in ns:
            heads = [h for h, _, _ in KPI] + (["違う値での実行（行）", "値まで見た EtoE の成功", "値まで見た重大な誤り（件）"] if lab else [])
            first = True
            for cname, mr in CONFIGS:
                got = measure(n, ts, loader, mr, rules_p21)
                if got is None:
                    continue
                if first:
                    print(f"\n===== 4. {tag}・層 {n} 個 =====")
                    md += [f"\n### 評価セット {tag}・層 {n} 個\n", "|構成|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
                    first = False
                k, d = got
                cells = [cell(k[key], nd) for _, key, nd in KPI]
                res["kpi"][f"{tag}|{n}|{cname}"] = {key: k[key] for _, key, _ in KPI}
                if lab:
                    v = value_kpi(k, d, lab, mr(arguments), state, len(rows))
                    cells += [f"{v['wrong_exec']:.1f}", f"{v['e2e_value']:.3f}", f"{v['severe_value_n']:.1f}（{v['severe_value'] * 100:.2f}%）"]
                    res["kpi"][f"{tag}|{n}|{cname}"]["value"] = v
                print(f"{cname:<16} " + "  ".join(f"{h[:12]}: {x}" for h, x in zip(heads, cells)))
                md.append(f"|{cname}|" + "|".join(cells) + "|")
                if cname == CONFIGS[-1][0] and tag in ("v6", "v5"):
                    md.append(f"\n直した後の重大な誤り（{tag}・層 {n}）: " + " / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:20]))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "fix.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "fix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（fix.md・fix.json）")


if __name__ == "__main__":
    main()
