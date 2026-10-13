# -*- coding: utf-8 -*-
"""P8: 「3 つがそろったときだけ実行する」流れで、評価セット v2 を測り直す。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P8_三つがそろったときだけ実行.md。

これまで（P4〜P7）の数字は「ヘッドだけ」の判定だった。本来の流れ（EdgeSLM_MultiHead_Multi の Dispatcher）には、
決定的な規則・引数の決定・検査がある。P8 では、その順番どおりに部品を足していき、KPI がどう変わるかを見る。

実行する条件（3 つがそろったときだけ）:
  1. 文の分類器（BERT）が「対象」と判定        … 確率が高い → 断る（C3 相当）／中間 → フラグ（操作系は実行せず確認。C2 相当）／低い → 通す
  2. LM ヘッドの確信度が高い                 … 規則 → 棄却の得点 → 確信度の閾値（Dispatcher と同じ順）
  3. 引数が取り出せる                        … `dispatch.arguments.resolve`（決定的）と `dispatch.policy.inspect`（schema・値域・ポリシー）
どれか欠けたら聞き返す。

読むだけで使うもの: EdgeSLM_MultiHead_Multi の `edge_router.dispatch.rules`・`arguments`・`policy`（学習パラメータなしの決定的な処理）、
`data/samples/vehicle_state.json`（引数を決めるときの車両の状態）。P5・P7 のヘッド、P7 の評価セットの `h` と BERT の確率。

KPI（評価セット v2・200 行）:
- EtoE の成功 = 正しく処理 ＋ 聞き返しで回復できる
- 重大な誤り = 操作系ツール（空調・メディア・ナビ）を、正解でないのに実行
- 不要な聞き返し = 対象の依頼・雑談を聞き返した
- 通常の誤り = それ以外（対象を断った、別の読むだけの機能や雑談へ流した、など）

    python -m app.p8_gate        # results/P8/gate.md・gate.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p6_seeds import SEEDS, head_probs, load_pt
from .p7_block import HEADS, WORK as P7, load_eval

RES = PROJECT_ROOT / "results" / "P8"
BERT_LO = 0.5             # これ以上でフラグ（分類器の自然な境目。評価セットを見て決めた値ではない）
STEPS = [("ヘッドだけ", dict(rules=False, args=False, bert=False, flag=False)),
         ("＋規則", dict(rules=True, args=False, bert=False, flag=False)),
         ("＋引数の検査", dict(rules=True, args=True, bert=False, flag=False)),
         ("＋BERT（断るだけ）", dict(rules=True, args=True, bert=True, flag=False)),
         ("＋BERT（断る＋フラグ）", dict(rules=True, args=True, bert=True, flag=True))]
KPI = [("EtoE の成功（正しく処理＋聞き返しで回復）", "e2e", 3), ("　うち 1 回目で正しく処理", "first", 3), ("重大な誤り（操作系の誤実行）の割合", "severe", 3),
       ("　件数（200 行中）", "severe_n", 1), ("不要な聞き返し（対象の依頼・雑談 120 行のうち）", "over", 3), ("通常の誤りの割合", "normal", 3),
       ("機能にない依頼 60 行: 断った", "os_refuse", 3), ("　聞き返した", "os_clarify", 3), ("　操作として実行した（件）", "os_severe_n", 1),
       ("意図が不明 20 行: 聞き返した", "amb_ok", 3), ("対象の依頼・雑談 120 行: 1 回目で正しく処理", "known_first", 3),
       ("　断ってしまった（件）", "known_refused_n", 1)]


def grade(r: dict, outcome: str, tool: str | None, write: set, amb_chat_ok: bool = False) -> str:
    """行の正解と、流れの結果（実行・雑談・聞き返し・拒否）から、正解／回復可能／通常の誤り／重大な誤り を決める。
    `amb_chat_ok=True`（P15 からの数え方）なら、意図が不明の発話を雑談として受けたのも正解にする（対話の中で必要に応じて聞き返せるため）"""
    exp, cat = r["action"], r["category"]
    done = "CHAT" if outcome == "雑談" else (f"TOOL_{tool}" if outcome == "実行" else None)
    if outcome == "実行" and tool in write and done != exp:
        return "重大"
    if exp == "UNKNOWN":
        if cat == "H":                                   # 機能にない依頼: 断るのが正解。聞き返しは安全側（回復可能）
            return "正解" if outcome == "拒否" else "回復可能" if outcome == "聞き返し" else "通常"
        if amb_chat_ok and outcome == "雑談":
            return "正解"                                   # 意図が不明を雑談として受ける: 実行はしていない。対話の中で聞き返せる
        return "正解" if outcome == "聞き返し" else "通常"      # 意図が不明: 聞き返すのが正解。断るのは応答がずれる
    if done == exp:
        return "正解"
    return "回復可能" if outcome == "聞き返し" else "通常"


def run(ns: list[int], make_resolve=None, out: str | None = "gate", res_dir=None, heads_hook=None, steps=None, sources=None,
        quiet: bool = False, ts: dict | None = None, watch=None, bert_hi=None, flag_all: bool = False,
        eval_loader=None, make_rules=None, amb_chat_ok: bool = False) -> dict:
    """`make_resolve` を渡すと、引数の決定をそれに差し替える（P9: 元の `arguments` モジュールを受け取り resolve を返す関数）。
    `heads_hook(n, src, heads)` を渡すと、ヘッドの組（seed・クラス・閾値・確率）を差し替える（P10: 棄却の閾値の決め方・seed の平均）。
    `steps`・`sources` で測る構成とヘッドをしぼる。`out=None` なら保存しない。`quiet` なら画面に出さない。
    `ts` を渡すと、文の分類器の確率をそれに差し替える（P11: 学習データを変えた BERT。形は work/p7/text_scores.json と同じ）。
    `watch`（行 id の並び）を渡すと、`detail` に「BERT のフラグで聞き返した対象の行」と、その行 id の行き先を足す。
    `bert_hi(name, neg)` を渡すと、BERT の断る閾値をそれで決める（P12。neg は学習側の交差検証の、機能にない依頼でない行の確率）。
    `flag_all=True` なら、BERT のフラグを操作系だけでなく読むだけの機能と雑談にも効かせる（P12）。
    `eval_loader(n)` を渡すと、評価セットをそれで読む（P13: 評価セット v3。(行, last, mean) か None を返す関数）。
    `make_rules(rules)` を渡すと、決定的な規則をそれに差し替える（P13: 元の `rules` モジュールを受け取り apply(row, 予測) を返す関数）。
    `amb_chat_ok=True` なら、意図が不明の発話を雑談として受けたのも正解に数える（P15。評価方針 4 節の改定）"""
    say = (lambda *a, **k: None) if quiet else print
    steps = steps or STEPS
    use_multi_library()
    from edge_router.dispatch import arguments, policy, rules      # 決定的な処理だけ（学習パラメータなし）
    resolve = arguments.resolve if make_resolve is None else make_resolve(arguments)
    apply_rules = rules.apply_rules if make_rules is None else make_rules(rules)
    res_dir = res_dir or RES
    tools = policy.load_tools()
    effect = {k: v.get("effect", "read") for k, v in tools.items()}
    write = {k for k, v in effect.items() if v == "write"}
    base_state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    ts = ts or json.loads((P7 / "text_scores.json").read_text(encoding="utf-8"))
    fit_ids = list(ts["labels_fit"])
    y = np.array([ts["labels_fit"][i] for i in fit_ids])
    berts = {}
    for name, m in ts["models"].items():
        if name.startswith("bert"):
            neg = np.array([m["cv"][i] for i in fit_ids])[y == 0]
            hi = float(np.quantile(neg, 0.99, method="higher")) if bert_hi is None else float(bert_hi(name, neg))
            berts[name] = {"hi": hi, "pred": m["pred"]}     # 既定は、学習側で 1% 止める値
    say("\n===== P8: 3 つがそろったときだけ実行（BERT が対象・ヘッドの確信度・引数）=====")
    say("BERT の閾値: 断る = " + "・".join(f"{v['hi']:.3f}" for v in berts.values()) + f"（学習側の交差検証で決めた値）/ フラグ = {BERT_LO}")

    def flow(r, k, p, classes, th, cfg, bp, bhi):
        """Dispatcher._handle と同じ順: （L1）→ 規則 → 棄却 → 確信度 → 引数 → 検査 →（L1 のフラグ）→ 実行"""
        if cfg["bert"] and bp >= bhi:
            return "拒否", "BERT が断る", None
        act = classes[int(p["action"][k].argmax())]
        conf, rej = float(p["action"][k].max()), float(p["reject"][k])
        if cfg["rules"]:
            d = apply_rules({"utterance": r["utterance"], "history": r.get("history") or [], "last_tool": None}, act)
            act = d.action
            if d.demote:
                return "聞き返し", "規則", None
        if rej >= th["REJECT"]:
            return "聞き返し", "棄却ヘッド", None
        tau = th["CHAT"] if act == "CHAT" else th["TOOL_write" if effect.get(act[5:]) == "write" else "TOOL_read"]
        if conf < tau:
            return "聞き返し", "確信度", None
        if act == "CHAT":
            if flag_all and cfg["flag"] and bp >= BERT_LO:
                return "聞き返し", "BERT のフラグ", None
            return "雑談", None, None
        tool = act[5:]
        if cfg["args"]:
            state = {**base_state, **r.get("state", {})}
            ar = resolve(tool, None, r["utterance"], state)
            if not ar.complete:
                return "聞き返し", "引数が足りない", tool
            v = policy.inspect(tool, ar.values, state)
            if not v.ok:
                return ("聞き返し" if v.missing else "拒否"), "検査", tool
        if cfg["flag"] and (flag_all or tool in write) and bp >= BERT_LO:
            return "聞き返し", "BERT のフラグ", tool
        return "実行", None, tool

    def kpis(rows, grades, outs) -> dict:
        n = len(rows)
        isH = [r["category"] == "H" for r in rows]
        isG = [r["category"] == "G" for r in rows]
        known = [r["action"] != "UNKNOWN" for r in rows]
        rate = lambda sel, f: float(np.mean([f(i) for i in range(n) if sel[i]]))
        return {"e2e": float(np.mean([g in ("正解", "回復可能") for g in grades])), "first": float(np.mean([g == "正解" for g in grades])),
                "severe": float(np.mean([g == "重大" for g in grades])), "severe_n": float(sum(g == "重大" for g in grades)),
                "over": rate(known, lambda i: outs[i][0] == "聞き返し"), "normal": float(np.mean([g == "通常" for g in grades])),
                "os_refuse": rate(isH, lambda i: outs[i][0] == "拒否"), "os_clarify": rate(isH, lambda i: outs[i][0] == "聞き返し"),
                "os_severe_n": float(sum(grades[i] == "重大" for i in range(n) if isH[i])),
                "amb_ok": rate(isG, lambda i: outs[i][0] == "聞き返し" or (amb_chat_ok and outs[i][0] == "雑談")), "known_first": rate(known, lambda i: grades[i] == "正解"),
                "known_refused_n": float(sum(outs[i][0] == "拒否" for i in range(n) if known[i]))}

    def agg(items):
        return {k: {"mean": float(np.mean([x[k] for x in items])), "min": float(min(x[k] for x in items)), "max": float(max(x[k] for x in items))}
                for k in items[0]}

    res, md, detail = {}, [], {}
    for src in (sources or HEADS):
        for n in ns:
            d = HEADS[src](n)
            if not (d / "evaluate_report.json").exists():
                continue
            if eval_loader is None:
                if not (P7 / f"device_eval_L{n}.json").exists():
                    continue
                loaded = load_eval(n)
            else:
                loaded = eval_loader(n)
                if loaded is None:
                    continue
            rows, last, mean = loaded
            reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
            heads = []
            for seed in SEEDS:
                name = f"heads_mlp_seed{seed}.pt"
                blob, rep = load_pt(d / name), reps[name]
                heads.append((seed, blob["metadata"]["action_classes"], {k: float(v) for k, v in rep["thresholds"].items()},
                              head_probs(blob, float(rep["temperature"]), last, mean)))
            if heads_hook is not None:
                heads = heads_hook(n, src, heads)
            for label, cfg in steps:
                items, stages, stuck = [], collections.Counter(), collections.Counter()
                flagged, seen, norm, groups = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
                stops = collections.Counter()
                for bname, b in (berts.items() if cfg["bert"] else [(None, None)]):
                    for seed, classes, th, p in heads:
                        outs = [flow(r, k, p, classes, th, cfg, 0.0 if b is None else b["pred"][r["id"]], 1.01 if b is None else b["hi"])
                                for k, r in enumerate(rows)]
                        grades = [grade(r, o[0], o[2], write, amb_chat_ok) for r, o in zip(rows, outs)]
                        items.append(kpis(rows, grades, outs))
                        for r, o, g in zip(rows, outs, grades):
                            who = "機能にない依頼" if r["category"] == "H" else "意図が不明" if r["category"] == "G" else "対象の依頼・雑談"
                            stages[(who, o[0] if o[1] is None else f"{o[0]}（{o[1]}）")] += 1
                            if g == "重大":
                                stuck[(r["category"], r["utterance"], f"{o[2]}")] += 1
                            if o[1] == "BERT のフラグ" and r["action"] != "UNKNOWN":
                                flagged[(r["category"], r["utterance"], f"{o[2]}")] += 1
                            if o[0] in ("聞き返し", "拒否") and r["action"] != "UNKNOWN":
                                stops[(r["category"], r["utterance"], f"{o[0]}（{o[1]}）", f"{o[2]}")] += 1
                            groups[(r["category"], r.get("os_group") or r.get("g_kind") or r["action"].replace("TOOL_", ""), g)] += 1
                            if g == "通常":
                                norm[(r["category"], r["utterance"], o[0] if o[2] is None else f"{o[0]}（{o[2]}）")] += 1
                            if watch is not None and r["id"] in watch:
                                seen[(r["utterance"], o[0] if o[2] is None else f"{o[0]}（{o[2]}）")] += 1
                key = f"{n}|{src}|{label}"
                res[key] = agg(items)
                k_runs = len(items)
                detail[key] = {"stages": {f"{a}|{b}": v / k_runs for (a, b), v in sorted(stages.items())},
                               "severe_rows": [{"category": c, "utterance": u, "executed": t, "share": v / k_runs} for (c, u, t), v in stuck.most_common()]}
                if watch is not None:
                    detail[key]["flag_rows"] = [{"category": c, "utterance": u, "tool": t, "share": v / k_runs} for (c, u, t), v in flagged.most_common()]
                    detail[key]["normal_rows"] = [{"category": c, "utterance": u, "outcome": w, "share": v / k_runs} for (c, u, w), v in norm.most_common()]
                    detail[key]["by_group"] = [{"category": c, "group": gr, "grade": g, "rows": v / k_runs} for (c, gr, g), v in sorted(groups.items())]
                    detail[key]["stop_rows"] = [{"category": c, "utterance": u, "stop": w, "tool": t, "share": v / k_runs} for (c, u, w, t), v in stops.most_common()]
                    detail[key]["watch"] = [{"utterance": u, "outcome": w, "share": v / k_runs} for (u, w), v in sorted(seen.items())]
            cols = [(label, res[f"{n}|{src}|{label}"]) for label, _ in steps]
            title = f"層 {n} 個・ヘッド {src}（mlp・seed 3 つ × BERT の seed の平均。かっこは最小〜最大）"
            say(f"\n===== {title} =====")
            md += [f"\n### {title}\n", "|KPI|" + "|".join(c for c, _ in cols) + "|", "|---|" + "---|" * len(cols)]
            for lab, key_, nd in KPI:
                cells = [f"{v[key_]['mean']:.{nd}f}（{v[key_]['min']:.{nd}f}〜{v[key_]['max']:.{nd}f}）" for _, v in cols]
                say(f"{lab:<30} " + "  ".join(f"[{c}] {x}" for (c, _), x in zip(cols, cells)))
                md.append(f"|{lab}|" + "|".join(cells) + "|")
            # 提案の構成で、どの段で止まったか・残った重大な誤り
            dk = detail[f"{n}|{src}|{steps[-1][0]}"]
            md += [f"\n{steps[-1][0]} のとき、どこで止まったか（1 回あたりの行数）:\n", "|行の種類|結果|行数|", "|---|---|---|"]
            for k2, v in dk["stages"].items():
                who, what = k2.split("|")
                md.append(f"|{who}|{what}|{v:.1f}|")
            md += ["\n残った重大な誤り（出た回数の割合）:\n", "|種類|発話|実行したツール|割合|", "|---|---|---|---|"]
            for x in dk["severe_rows"][:25]:
                md.append(f"|{x['category']}|{x['utterance']}|{x['executed']}|{x['share']:.2f}|")
            say("  残った重大な誤り: " + " / ".join(f"[{x['category']}]{x['utterance']}→{x['executed']}（{x['share']:.2f}）" for x in dk["severe_rows"][:12]))
    if not res:
        raise SystemExit("測れる組が無い（P7 の結果が要る）")
    # まとめ: 提案の構成を層ごとに
    top = ["## まとめ（＋BERT（断る＋フラグ）＝ 3 つがそろったときだけ実行）\n", "|KPI|" + "|".join(f"層 {n}・{src}" for src in (sources or HEADS) for n in ns if f"{n}|{src}|{steps[-1][0]}" in res) + "|"]
    keys = [f"{n}|{src}|{steps[-1][0]}" for src in (sources or HEADS) for n in ns if f"{n}|{src}|{steps[-1][0]}" in res]
    top.append("|---|" + "---|" * len(keys))
    say("\n===== まとめ（＋BERT（断る＋フラグ））=====")
    for lab, key_, nd in KPI:
        cells = [f"{res[k][key_]['mean']:.{nd}f}（{res[k][key_]['min']:.{nd}f}〜{res[k][key_]['max']:.{nd}f}）" for k in keys]
        top.append(f"|{lab}|" + "|".join(cells) + "|")
        say(f"{lab:<30} " + "  ".join(f"[{k.split('|')[0]}/{k.split('|')[1][:5]}] {res[k][key_]['mean']:.{nd}f}" for k in keys))
    blob = {"bert_hi": {k: v["hi"] for k, v in berts.items()}, "bert_lo": BERT_LO, "kpi": res, "detail": detail}
    if out is None:
        return blob
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{out}.json").write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    (res_dir / f"{out}.md").write_text("\n".join(top + ["\n## 層・ヘッドごと（部品を 1 つずつ足す）"] + md) + "\n", encoding="utf-8")
    say(f"\n保存しました: {res_dir}（{out}.md・{out}.json）")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    run([int(x) for x in a.layers.replace(" ", ",").split(",") if x])


if __name__ == "__main__":
    main()
