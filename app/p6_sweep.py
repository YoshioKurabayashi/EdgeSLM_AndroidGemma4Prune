# -*- coding: utf-8 -*-
"""P6 追加: 閾値を動かしたときの正答率を並べる（学習済みの mlp ヘッド 3 つ × 3 層。学習なし・端末なし・numpy だけ）。

設計: DoC/設計書_P6_ヘッドのseedを変えた判定.md の 8 節。

閾値は 4 つある（棄却・操作系の確信度・読むだけの確信度・雑談の確信度）。validation で決めた値が、正解との合い方で見てよい場所にあるかを確かめる。

|表|動かすもの|ほかの閾値|
|---|---|---|
|棄却の閾値|0.05〜0.99 と「棄却を使わない」|validation で決めた値のまま|
|操作系の確信度の閾値|0.5〜0.9|棄却は 0.5 に固定・ほかは validation で決めた値|

測る行:
- 評価セット（challenge_v1・M を除く 350 行 = 行き先のある行 273・確認へ回すべき行 77）。`h` は P4 のもの
- 参考: 933 行の test（186 行 = 行き先のある行 147・確認へ回すべき行 39）。学習と同じ作り方の文。`h` は `work/p5/L<n>/features`

指標: 全体の正答率（行き先のある行は正しい行き先、確認へ回すべき行は確認・聞き返し、が正解）・行き先のある行の正解率・
行き先のある行を確認へ回した割合・確認へ回すべき行を回せた割合・誤実行の件数。seed 3 つの平均。

    python -m app.p6_sweep        # results/P6/threshold_sweep.md・threshold_sweep.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT
from .p5_retrain import DATA, P4W, WORK as P5, challenge_h, decide
from .p6_seeds import RES, SEEDS, head_probs, load_pt

REJECT_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]      # 1.01 = 棄却を使わない
WRITE_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_REJECT = 0.5
MATCH_OVER = 0.05         # 「行き先のある行を確認へ回す割合」をこの値にそろえて比べる（同じ手間のときの比べ）
COMMON = {"TOOL_write": 0.5, "TOOL_read": 0.3, "CHAT": 0.4}      # そろえて比べるときの確信度の閾値（seed ごとの違いを除く）
FINE_GRID = [round(x, 3) for x in np.linspace(0.005, 0.995, 199)] + [1.01]


def score(out: list[dict]) -> dict:
    sc = [r for r in out if r["category"] != "M"]
    known = [r for r in sc if r["expected"] != "UNKNOWN"]
    unk = [r for r in sc if r["expected"] == "UNKNOWN"]
    return {"n": len(sc), "n_known": len(known), "n_unknown": len(unk),
            "accuracy_all": float(np.mean([r["ok"] for r in sc])),
            "accuracy_known": float(np.mean([r["kind"] == "正解" for r in known])),
            "over_clarify_known": float(np.mean([r["kind"] == "確認へ" for r in known])),
            "reject_recall": float(np.mean([r["kind"] == "確認へ" for r in unk])),
            "wrong_write": int(sum(r["kind"] == "誤実行" for r in sc))}


def mean_of(items: list[dict]) -> dict:
    return {k: float(np.mean([x[k] for x in items])) for k in items[0]}


def test_rows(n: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """933 行の test（学習・閾値決めに使っていない行）。評価セットと同じ形の行にする"""
    rows = [json.loads(l) for l in (DATA / "utterances.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    d = P5 / f"L{n:02d}" / "features"
    ids = json.loads((d / "features_F2.meta.json").read_text(encoding="utf-8"))["ids"]
    z = np.load(d / "features_F2.npz")
    pick = [i for i, r in enumerate(rows) if r["split"] == "test"]
    if [rows[i]["id"] for i in pick] != [ids[i] for i in pick]:
        raise SystemExit(f"層 {n}: 特徴の並びがデータと違う")
    out = [{"id": rows[i]["id"], "category": "T", "utterance": rows[i]["utterance"], "action": rows[i]["action"]} for i in pick]
    return out, z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)


def main() -> None:
    from .p4_eval import load_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    tools = {t["name"]: t["effect"] for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    allrows = load_rows()
    res, md = {}, []
    print("\n===== P6 追加: 閾値を動かしたときの正答率（mlp・933 行のヘッド・seed 3 つの平均）=====", flush=True)
    for n in ns:
        d = P5 / f"L{n:02d}" / "heads"
        if not (P4W / f"device_L{n}.json").exists() or not (d / "evaluate_report.json").exists():
            print(f"層 {n}: 評価セットの h か、ヘッドが無い。飛ばす")
            continue
        reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
        sets = {"評価セット": challenge_h(n, allrows), "933 行の test": test_rows(n)}
        res[n] = {}
        for sname, (rows, last, mean) in sets.items():
            heads = []
            for seed in SEEDS:
                name = f"heads_mlp_seed{seed}.pt"
                blob, rep = load_pt(d / name), reps[name]
                meta = {"action_classes": blob["metadata"]["action_classes"], "tool_effect": tools,
                        "thresholds": {k: float(v) for k, v in rep["thresholds"].items()}}
                heads.append((seed, meta, head_probs(blob, float(rep["temperature"]), last, mean)))

            def run(over: dict) -> dict:
                return mean_of([score(decide(rows, {**m, "thresholds": {**m["thresholds"], **over}}, p)) for _, m, p in heads])

            own = run({})
            chosen = [m["thresholds"]["REJECT"] for _, m, _ in heads]
            rej = [{"tau": t, **run({"REJECT": t})} for t in REJECT_GRID]
            wr = [{"tau": t, **run({"REJECT": FIXED_REJECT, "TOOL_write": t})} for t in WRITE_GRID]
            best = max(rej, key=lambda x: x["accuracy_all"])
            # ヘッドごとに、評価する行の上で一番よい棄却の閾値を選んだ場合（後知恵の上限）との差
            per = []
            for seed, m, p in heads:
                o = score(decide(rows, m, p))
                b = max((score(decide(rows, {**m, "thresholds": {**m["thresholds"], "REJECT": t}}, p)) | {"tau": t} for t in REJECT_GRID),
                        key=lambda x: x["accuracy_all"])
                per.append({"seed": seed, "chosen_tau": m["thresholds"]["REJECT"], "chosen": o, "best_tau": b["tau"], "best": b})
            # 同じ手間（行き先のある行を確認へ回す割合が MATCH_OVER に一番近い棄却の閾値）のときの値を、seed ごとに。
            # 確信度の閾値も seed ごとに違う（操作系 0.5〜0.7・雑談 0.2〜0.5）ので、ここでは COMMON にそろえる
            matched = []
            for seed, m, p in heads:
                cand = [score(decide(rows, {**m, "thresholds": {**COMMON, "REJECT": t}}, p)) | {"tau": t} for t in FINE_GRID]
                matched.append({"seed": seed, **min(cand, key=lambda x: (abs(x["over_clarify_known"] - MATCH_OVER), -x["tau"]))})
            # 棄却の得点だけで見る: 行き先のある行の FPR 分を誤って回す閾値で、確認へ回すべき行を何割回せるか（確信度の閾値は関係しない）
            # あわせて、棄却を使わないときに確信度だけで聞き返しになる行き先のある行の割合（COMMON の閾値）
            cat = np.array([r["category"] for r in rows])
            known = np.array([r["action"] not in ("UNKNOWN", "MULTI") for r in rows])
            unk = np.array([r["action"] == "UNKNOWN" for r in rows])
            alone = []
            for seed, m, p in heads:
                x = {"seed": seed, "clarify_by_confidence_only": score(decide(rows, {**m, "thresholds": {**COMMON, "REJECT": 1.01}}, p))["over_clarify_known"]}
                for fpr in (0.02, 0.05):
                    tau = float(np.quantile(p["reject"][known], 1.0 - fpr, method="higher"))
                    x[f"tpr_all_at_{fpr}"] = float((p["reject"][unk] >= tau).mean())
                    for c in ("G", "H"):
                        if (cat == c).any():
                            x[f"tpr_{c}_at_{fpr}"] = float((p["reject"][cat == c] >= tau).mean())
                alone.append(x)
            res[n][sname] = {"validation_chosen": {"reject_taus": chosen, **own}, "reject_sweep": rej, "write_sweep": wr, "per_seed": per,
                             "matched_over_clarify": matched, "reject_score_alone": alone}
            k = score(decide(rows, heads[0][1], heads[0][2]))
            print(f"\n########## 層 {n} 個・{sname}（{k['n']} 行 = 行き先のある行 {k['n_known']}・確認へ回すべき行 {k['n_unknown']}）##########")
            print(f"{'棄却の閾値':<22} {'全体の正答率':>8} {'行き先の正解率':>8} {'確認へ回した':>8} {'回すべき行を回せた':>10} {'誤実行':>6}")
            md += [f"\n### 層 {n} 個・{sname}（{k['n']} 行 = 行き先のある行 {k['n_known']}・確認へ回すべき行 {k['n_unknown']}）\n",
                   "|棄却の閾値|全体の正答率|行き先のある行の正解率|行き先のある行を確認へ回した割合|確認へ回すべき行を回せた割合|誤実行（件）|", "|---|---|---|---|---|---|"]
            lines = [("validation で決めた値（" + "・".join(f"{t:.2f}" for t in chosen) + "）", own)] + \
                    [("使わない" if x["tau"] > 1 else f"{x['tau']:.2f}" + ("　← 一番よい" if x is best else ""), x) for x in rej]
            for name, x in lines:
                print(f"{name:<22} {x['accuracy_all']:>10.3f} {x['accuracy_known']:>12.3f} {x['over_clarify_known']:>10.3f} {x['reject_recall']:>14.3f} {x['wrong_write']:>8.1f}")
                md.append(f"|{name}|{x['accuracy_all']:.3f}|{x['accuracy_known']:.3f}|{x['over_clarify_known']:.3f}|{x['reject_recall']:.3f}|{x['wrong_write']:.1f}|")
            md += [f"\n操作系の確信度の閾値を動かす（棄却は {FIXED_REJECT} に固定）:\n",
                   "|操作系の確信度の閾値|全体の正答率|行き先のある行の正解率|行き先のある行を確認へ回した割合|確認へ回すべき行を回せた割合|誤実行（件）|", "|---|---|---|---|---|---|"]
            print(f"操作系の確信度の閾値を動かす（棄却は {FIXED_REJECT} に固定）:")
            for x in wr:
                print(f"  {x['tau']:.1f}: 全体 {x['accuracy_all']:.3f}・行き先 {x['accuracy_known']:.3f}・確認へ {x['over_clarify_known']:.3f}・回せた {x['reject_recall']:.3f}・誤実行 {x['wrong_write']:.1f}")
                md.append(f"|{x['tau']:.1f}|{x['accuracy_all']:.3f}|{x['accuracy_known']:.3f}|{x['over_clarify_known']:.3f}|{x['reject_recall']:.3f}|{x['wrong_write']:.1f}|")
            md += ["\nseed ごと（validation で決めた閾値 ↔ この行の上で一番よい閾値）:\n",
                   "|seed|決めた閾値|全体の正答率|誤実行|一番よい閾値|全体の正答率|誤実行|", "|---|---|---|---|---|---|---|"]
            for x in per:
                t = "使わない" if x["best_tau"] > 1 else f"{x['best_tau']:.2f}"
                print(f"  seed {x['seed']:>2}: 決めた閾値 {x['chosen_tau']:.3f} → 全体 {x['chosen']['accuracy_all']:.3f}・誤実行 {x['chosen']['wrong_write']} / "
                      f"一番よい閾値 {t} → 全体 {x['best']['accuracy_all']:.3f}・誤実行 {x['best']['wrong_write']}")
                md.append(f"|{x['seed']}|{x['chosen_tau']:.3f}|{x['chosen']['accuracy_all']:.3f}|{x['chosen']['wrong_write']}|{t}|"
                          f"{x['best']['accuracy_all']:.3f}|{x['best']['wrong_write']}|")
    if not res:
        raise SystemExit("測れる層が無い")
    # 同じ手間にそろえたときの、層の比べ（seed 3 つの平均と最小〜最大）
    top = [f"## 同じ手間にそろえたとき（行き先のある行を確認へ回す割合が {MATCH_OVER:.0%} に一番近い棄却の閾値。確信度の閾値は操作系 {COMMON['TOOL_write']}・読むだけ {COMMON['TOOL_read']}・雑談 {COMMON['CHAT']} にそろえる。seed 3 つの平均と最小〜最大）\n"]
    print(f"\n===== 同じ手間にそろえたとき（確認へ回す割合 {MATCH_OVER:.0%}）=====")
    for sname in ("評価セット", "933 行の test"):
        top += [f"\n{sname}:\n", "|層|そろえた閾値（seed ごと）|確認へ回した割合|全体の正答率|行き先のある行の正解率|確認へ回すべき行を回せた割合|誤実行（件）|", "|---|---|---|---|---|---|---|"]
        for n in res:
            mt = res[n][sname]["matched_over_clarify"]
            f = lambda k, fmt: f"{fmt.format(np.mean([x[k] for x in mt]))}（{fmt.format(min(x[k] for x in mt))}〜{fmt.format(max(x[k] for x in mt))}）"
            taus = "・".join("使わない" if x["tau"] > 1 else f"{x['tau']:.2f}" for x in mt)
            line = (f"層 {n}・{sname}: 閾値 {taus} / 確認へ {f('over_clarify_known', '{:.3f}')}・全体 {f('accuracy_all', '{:.3f}')}・"
                    f"行き先 {f('accuracy_known', '{:.3f}')}・回せた {f('reject_recall', '{:.3f}')}・誤実行 {f('wrong_write', '{:.1f}')}")
            print(line)
            top.append(f"|層 {n}|{taus}|{f('over_clarify_known', '{:.3f}')}|{f('accuracy_all', '{:.3f}')}|{f('accuracy_known', '{:.3f}')}|"
                       f"{f('reject_recall', '{:.3f}')}|{f('wrong_write', '{:.1f}')}|")
    top += ["\n## 棄却の得点だけで見たとき（評価セット。行き先のある行の 2%・5% を誤って回す閾値で、回すべき行を何割回せるか。seed 3 つの平均と最小〜最大）\n",
            "|層|確信度だけで聞き返しになる行き先のある行|2% のとき 全体|機能にない依頼|意図が不明|5% のとき 全体|機能にない依頼|意図が不明|", "|---|---|---|---|---|---|---|---|"]
    print("\n===== 棄却の得点だけで見たとき（評価セット）=====")
    for n in res:
        al = res[n]["評価セット"]["reject_score_alone"]
        f = lambda k: f"{np.mean([x[k] for x in al]):.3f}（{min(x[k] for x in al):.3f}〜{max(x[k] for x in al):.3f}）"
        keys = ["clarify_by_confidence_only", "tpr_all_at_0.02", "tpr_H_at_0.02", "tpr_G_at_0.02", "tpr_all_at_0.05", "tpr_H_at_0.05", "tpr_G_at_0.05"]
        print(f"層 {n}: 確信度だけで聞き返し {f(keys[0])} / 2%: 全体 {f(keys[1])}・H {f(keys[2])}・G {f(keys[3])} / 5%: 全体 {f(keys[4])}・H {f(keys[5])}・G {f(keys[6])}")
        top.append(f"|層 {n}|" + "|".join(f(k) for k in keys) + "|")
    md = top + ["\n## 層ごと・行ごとの表\n"] + md
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "threshold_sweep.json").write_text(json.dumps({str(k): v for k, v in res.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "threshold_sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（threshold_sweep.md・threshold_sweep.json）")


if __name__ == "__main__":
    main()
