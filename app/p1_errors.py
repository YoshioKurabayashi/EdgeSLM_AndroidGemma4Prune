# -*- coding: utf-8 -*-
"""P1 手順 4: 層を減らしたときに出る誤実行（S2）・過剰な聞き返しを、行ごとに調べる。閾値で消せるかも見る。CPU で数分。

設計: DoC/設計書_P1_層ごとの分類性能.md の 9 節。

交差検証（`app/p1_heads.py` が作った `work/p1/L<n>/crossval/<kind>_seed<s>_fold<k>/`）の学習済みヘッドを**読み直し**、
EdgeSLM_MultiHead_Multi の `edge_router.heads.crossval` と同じ手順（fold の分け方・温度・閾値の選び方・判定）で
test の判定を作り直す（学習はしない）。そのうえで:

1. **誤実行に到達した行**: 発話・正解・予測・確信度と閾値・棄却の得点と閾値。どの fold・seed で出たか
2. **過剰な聞き返し**（正しく分類したのに確認へ回した行）の多い発話
3. **閾値の感度**: 閾値の選び方（validation で決める）はそのままに、test で REJECT を下げる・TOOL_write を上げたら
   誤実行・正解率・過剰な聞き返し・棄却の再現率がどう動くか（**test を見て閾値を動かすので楽観的**。
   「閾値で消せる種類の誤りか」を見るための目安で、選び直しではない）

    python -m app.p1_errors                       # 層 17・24・28・35（1 つの層で 3 分ほど）
    python -m app.p1_errors --layers 17,24,35
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import RESULTS, ROUTER_DATA, WORK, layer_dir, use_multi_library

REJECT_ALT = [None, 0.90, 0.80, 0.50]            # None = validation で選んだ値のまま。数字は min(選んだ値, 数字)
WRITE_ALT = [None, 0.70, 0.80, 0.90]             # None = そのまま。数字は max(選んだ値, 数字)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="")
    ap.add_argument("--kind", default="mlp")
    a = ap.parse_args()
    use_multi_library()
    import torch  # noqa: F401
    from edge_router.heads.crossval import SEEDS, fold_split, make_folds
    from edge_router.heads.dataset import load_bundle
    from edge_router.heads.evaluate import choose_thresholds, fit_temperature, predict
    from edge_router.heads.metrics import evaluate, load_evaluation_config
    from edge_router.heads.model import MultiHead
    from edge_router.io_utils import read_jsonl

    ns = [int(x) for x in (a.layers or "17,24,28,35").replace(" ", ",").split(",") if x]
    ns = [n for n in ns if (layer_dir(n) / "crossval" / "crossval_report.json").exists()]
    rows = read_jsonl(str(ROUTER_DATA))
    assignment = make_folds(rows, 5)
    config = load_evaluation_config()
    out = {}
    for n in ns:
        feats = layer_dir(n) / "features"
        cvd = layer_dir(n) / "crossval"
        reached, over_utts = [], collections.Counter()
        sweep = {(r, w): collections.Counter() for r in REJECT_ALT for w in WRITE_ALT}
        for seed in SEEDS:
            for k in range(5):
                bundle = load_bundle(str(ROUTER_DATA), str(feats), "F2", "last", "last",
                                     split_override=fold_split(assignment, k, 5))
                model, _ = MultiHead.load(cvd / f"{a.kind}_seed{seed}_fold{k}" / f"heads_{a.kind}_seed{seed}.pt",
                                          map_location="cpu")
                va, te, rj = bundle.splits["validation"], bundle.splits["test"], bundle.rejection["test"]
                model.temperature.fill_(fit_temperature(model, va.x, va.action))
                th, _ = choose_thresholds(model, bundle, config)
                tp, tc, tr = predict(model, te)
                rp, rc, rr = predict(model, rj)
                cls = bundle.action_classes
                res = evaluate(te.rows, te.action, tp, tc, rj.rows, rp, rc, cls, th, config,
                               reject_score=tr, r_reject_score=rr)
                for x in res["wrong_write_execution"]["reached"]:
                    hit = [("rejection", i) for i, r in enumerate(rj.rows) if r["utterance"] == x["utterance"]] + \
                          [("test", i) for i, r in enumerate(te.rows) if r["utterance"] == x["utterance"]]
                    src, i = hit[0]
                    conf, rej, row = (rc[i], rr[i], rj.rows[i]) if src == "rejection" else (tc[i], tr[i], te.rows[i])
                    reached.append({"seed": seed, "fold": k, "utterance": x["utterance"], "id": row.get("id"),
                                    "truth": row.get("action"), "note": row.get("note"), "predicted": x["predicted"],
                                    "confidence": round(float(conf), 4), "tau_write": th["TOOL_write"],
                                    "reject_score": round(float(rej), 4), "tau_reject": th["REJECT"]})
                # 過剰な聞き返し（正しく分類したのに降格）: 閾値での降格だけを数える（ルールでの降格は見ない）
                eff = {"write": th["TOOL_write"], "read": th["TOOL_read"]}
                for i, r in enumerate(te.rows):
                    if tp[i] != te.action[i]:
                        continue
                    name = cls[tp[i]]
                    tau = th["CHAT"] if name == "CHAT" else th["TOOL_write"] if name in (
                        "TOOL_control_hvac", "TOOL_control_media", "TOOL_set_navigation") else th["TOOL_read"]
                    if tr[i] >= th["REJECT"] or tc[i] < tau:
                        over_utts[r["utterance"]] += 1
                for ra in REJECT_ALT:
                    for wa in WRITE_ALT:
                        t2 = dict(th)
                        if ra is not None:
                            t2["REJECT"] = min(th["REJECT"], ra)
                        if wa is not None:
                            t2["TOOL_write"] = max(th["TOOL_write"], wa)
                        r2 = evaluate(te.rows, te.action, tp, tc, rj.rows, rp, rc, cls, t2, config,
                                      reject_score=tr, r_reject_score=rr)
                        c = sweep[(ra, wa)]
                        c["n"] += r2["n"]; c["correct"] += r2["correct_count"]; c["over"] += r2["over_clarification_count"]
                        c["n_rej"] += r2["n_rejection"]; c["rejected"] += r2["rejection"]["demoted"]
                        c["s2"] += r2["wrong_write_execution"]["S2"]["count"]; c["s3"] += r2["wrong_write_execution"]["S3"]["count"]
        sw = [{"reject": "選んだ値" if ra is None else f"≤{ra}", "write": "選んだ値" if wa is None else f"≥{wa}",
               "accuracy": c["correct"] / c["n"], "over_rate": c["over"] / c["n"], "rejection_recall": c["rejected"] / c["n_rej"],
               "s2": c["s2"], "s3": c["s3"]} for (ra, wa), c in sweep.items()]
        out[n] = {"reached": reached, "over_top": over_utts.most_common(15), "sweep": sw}
        print(f"\n########## 層 {n} 個（{a.kind}・seed {list(SEEDS)}・5 fold）##########", flush=True)
        rep = json.loads((cvd / "crossval_report.json").read_text(encoding="utf-8"))
        rep_n = sum(r["pooled"]["s2"] + r["pooled"]["s3"] for r in rep["results"] if r["kind"] == a.kind)
        out[n]["reported_wrong_writes"] = rep_n
        print(f"誤実行に到達した行: {len(reached)} 件（交差検証の報告 {rep_n} 件{'・一致' if rep_n == len(reached) else '・★不一致'}）", flush=True)
        for x in reached:
            print(f"  seed {x['seed']} fold {x['fold']}「{x['utterance']}」（正解 {x['truth']}・{x['note']}）→ {x['predicted']} "
                  f"確信度 {x['confidence']:.3f}（閾値 {x['tau_write']}）棄却 {x['reject_score']:.3f}（閾値 {x['tau_reject']}）", flush=True)
        print(f"過剰な聞き返しの多い発話（seed×fold で数えた回数）: " +
              "、".join(f"「{u}」{c}" for u, c in over_utts.most_common(8)), flush=True)
        print(f"閾値の感度（test で動かした楽観的な目安。3 seed の合計）:", flush=True)
        print(f"  {'REJECT':>8} {'write':>8} {'正解率':>7} {'過剰':>6} {'棄却再現':>8} {'S2':>3} {'S3':>3}", flush=True)
        for s in sw:
            print(f"  {s['reject']:>8} {s['write']:>8} {s['accuracy']:>7.4f} {s['over_rate']:>6.3f} {s['rejection_recall']:>8.3f} "
                  f"{s['s2']:>3} {s['s3']:>3}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "errors.json").write_text(json.dumps({str(k): v for k, v in out.items()}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"\n保存しました: {RESULTS / 'errors.json'}")


if __name__ == "__main__":
    main()
