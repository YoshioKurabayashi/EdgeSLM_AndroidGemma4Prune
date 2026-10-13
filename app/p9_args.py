# -*- coding: utf-8 -*-
"""P9: 引数の取り出しを補う層（`app/p9_arguments.py`）の効果を測る。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P9_引数の取り出しを補う.md。

1. 引数の段で止まる割合を、元の処理と補った処理で比べる（正しいツールを選べたとして）。
   学習側（714 行・学習に足した行）は規則を作るのに使った行。**評価セット v1・v2 は規則を作るのに使っていない行**（v2 の 8 行は P8 で見ている）
2. 「3 つがそろったときだけ実行」の流れ（P8）を、補った処理で通し直し、KPI を前と後で並べる。
   引数の関門をゆるめると、機能にない依頼が通りやすくなる（重大な誤りが増える）ので、両方を見る

    python -m app.p9_args        # results/P9/args_compare.md・args_compare.json・gate_args2.md・gate_args2.json
"""
from __future__ import annotations

import argparse
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p7_block import HEADS
from .p8_gate import KPI, STEPS, run
from .p9_arguments import make_resolve

RES = PROJECT_ROOT / "results" / "P9"
P8 = PROJECT_ROOT / "results" / "P8" / "gate.json"
SETS = [("714 行（履歴なし）", "学習側（規則を作るのに使った）", "train_add_v2", lambda r: "added" not in r and not r["history"]),
        ("学習に足した行 v1・v2", "学習側（規則を作るのに使った）", "train_add_v2", lambda r: "added" in r),
        ("評価セット v1", "評価側（規則を作るのに使っていない）", "challenge_v1", lambda r: r["category"] != "M"),
        ("評価セット v2", "評価側（8 行は P8 で見ている）", "challenge_v2", lambda r: True)]


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def completeness() -> tuple[list[str], dict]:
    use_multi_library()
    from edge_router.dispatch import arguments, policy
    r2 = make_resolve(arguments)
    base_state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    md = ["|発話のまとまり|位置づけ|ツールの行|元の処理で止まる|補った処理で止まる|うちナビ|うち走行中のポリシーで断る（正しい）|", "|---|---|---|---|---|---|---|"]
    out = {}
    print("\n===== 引数の段で止まる行（正しいツールを選べたとして）=====")
    for name, role, folder, sel in SETS:
        rows = [r for r in read_jsonl(PROJECT_ROOT / "data" / folder / "utterances.jsonl") if sel(r) and r["action"].startswith("TOOL_")]
        n = b0 = b1 = nav = pol = 0
        left = []
        for r in rows:
            tool, state = r["action"][5:], {**base_state, **r.get("state", {})}
            n += 1
            a0, a1 = arguments.resolve(tool, None, r["utterance"], state), r2(tool, None, r["utterance"], state)
            ok0 = a0.complete and policy.inspect(tool, a0.values, state).ok
            v1 = policy.inspect(tool, a1.values, state) if a1.complete else None
            ok1 = bool(v1 and v1.ok)
            b0 += not ok0
            if not ok1:
                b1 += 1
                nav += tool == "set_navigation"
                pol += bool(v1 and not v1.ok and v1.stage == "policy")
                left.append({"tool": tool, "utterance": r["utterance"], "why": "検査: " + str(v1.stage) if v1 else "不足: " + "・".join(a1.missing)})
        out[name] = {"n": n, "before": b0, "after": b1, "after_nav": nav, "after_policy": pol, "left": left}
        print(f"{name}（{role}）: {n} 行 / 止まる 前 {b0}（{b0 / n:.1%}）→ 後 {b1}（{b1 / n:.1%}）/ うちナビ {nav}・走行中のポリシー {pol}")
        md.append(f"|{name}|{role}|{n}|{b0}（{b0 / n:.1%}）|{b1}（{b1 / n:.1%}）|{nav}|{pol}|")
    return md, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    if not P8.exists():
        raise SystemExit("P8 の結果（results/P8/gate.json）が無い。先に python -m app.p8_gate")
    cmd, comp = completeness()
    after = run(ns, make_resolve=make_resolve, out="gate_args2", res_dir=RES)["kpi"]
    before = json.loads(P8.read_text(encoding="utf-8"))["kpi"]
    fin = STEPS[-1][0]
    md = ["## 1. 引数の段で止まる行（正しいツールを選べたとして）\n"] + cmd
    md += ["\n補った処理でも止まる行（評価セット v2）:\n", "|ツール|発話|理由|", "|---|---|---|"]
    md += [f"|{x['tool']}|{x['utterance']}|{x['why']}|" for x in comp["評価セット v2"]["left"]]
    md += [f"\n## 2. 「3 つがそろったときだけ実行」の KPI（{fin}。引数の処理: 元 → 補った）\n"]
    print(f"\n===== KPI（{fin}）: 引数の処理 元 → 補った =====")
    for src in HEADS:
        keys = [n for n in ns if f"{n}|{src}|{fin}" in after and f"{n}|{src}|{fin}" in before]
        if not keys:
            continue
        md += [f"\nヘッド {src}:\n", "|KPI|" + "|".join(f"層 {n}" for n in keys) + "|", "|---|" + "---|" * len(keys)]
        print(f"--- ヘッド {src}")
        for lab, key, nd in KPI:
            cells = [f"{before[f'{n}|{src}|{fin}'][key]['mean']:.{nd}f} → {after[f'{n}|{src}|{fin}'][key]['mean']:.{nd}f}" for n in keys]
            md.append(f"|{lab}|" + "|".join(cells) + "|")
            print(f"{lab:<30} " + "   ".join(f"層 {n}: {c}" for n, c in zip(keys, cells)))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "args_compare.json").write_text(json.dumps({"completeness": comp, "kpi_before": before, "kpi_after": after}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (RES / "args_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（args_compare.md・args_compare.json・gate_args2.md・gate_args2.json）")


if __name__ == "__main__":
    main()
