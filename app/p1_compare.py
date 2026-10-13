# -*- coding: utf-8 -*-
"""P1 手順 3: 層の数ごとのヘッドの精度を並べ、「何層あれば分類が落ちないか」を出す。CPU で数秒。

設計: DoC/設計書_P1_層ごとの分類性能.md の 6 節。

指標の取り方は EdgeSLM_AndroidGemma4Head の `app/g1_compare.py`（md5 8ee38d3a）と同じ（関数を写した）。linear（線形の分類器）も並べる。
端末の目安（NPU の 1 回・PLE の表・グラフの大きさ）は、改善版 B の 35 層の実測を層の数で割り振った**粗い見込み**（層ごとの大きさの違いは見ていない）。

    python -m app.p1_compare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import G1_SRQ, N_LAYERS, RESULTS, SEQUENCE_CHECKPOINT, WORK, layer_dir

FULL_NPU_MS = 88.5        # 改善版 B（35 層・T=64・出力層なし）の端末の 1 回（EdgeSLM_AndroidGemma4Head B3a）
FULL_PLE_MB = 1120        # 端末の PLE の表（35 層ぶん）
FULL_CTX_MB = 1273        # 改善版 B の context binary（35 層）


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def summarize_evaluate(path: Path) -> dict:
    e = _load(path)
    if not e:
        return {}
    s7 = next((r for r in e["results"] if r["checkpoint"] == SEQUENCE_CHECKPOINT), None)
    return {"mlp_s7_test_accuracy": s7["test"]["accuracy"] if s7 else None,
            "mlp_s7_checks_failed": [k for k, v in (s7 or {}).get("test_checks", {}).items() if not v["ok"]]}


def summarize_crossval(path: Path) -> dict:
    c = _load(path)
    if not c:
        return {}
    out = {}
    for kind in ("mlp", "linear"):
        rs = [r for r in c["results"] if r["kind"] == kind]
        if not rs:
            continue
        accs = [r["aggregate"]["accuracy"] for r in rs]
        out[kind] = {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                     "rejection_recall_mean": float(np.mean([r["aggregate"]["rejection_recall"] for r in rs])),
                     "configs_all_ok": sum(all(v["ok"] for v in r["checks"].values()) for r in rs), "configs": len(rs),
                     "wrong_writes": sum(r["pooled"]["s2"] + r["pooled"]["s3"] for r in rs),
                     "failed_checks": sorted({k for r in rs for k, v in r["checks"].items() if not v["ok"]})}
    return out


def main() -> None:
    rows = []
    for d in sorted(WORK.glob("L*")):
        if not d.is_dir():
            continue
        n = int(d.name[1:])
        cv = summarize_crossval(d / "crossval" / "crossval_report.json")
        ev = summarize_evaluate(d / "heads" / "evaluate_report.json")
        if cv:
            rows.append({"n_layers": n, "crossval": cv, "evaluate": ev,
                         "device_estimate": {"npu_ms_T64": FULL_NPU_MS * n / N_LAYERS, "ple_mb": FULL_PLE_MB * n / N_LAYERS,
                                             "ctx_mb": FULL_CTX_MB * n / N_LAYERS}})
    rows.sort(key=lambda r: r["n_layers"])
    ref = {"crossval": summarize_crossval(G1_SRQ / "crossval" / "crossval_report.json"),
           "evaluate": summarize_evaluate(G1_SRQ / "heads" / "evaluate_report.json")}
    full = next((r for r in rows if r["n_layers"] == N_LAYERS), None)
    base_acc = (full or ref)["crossval"]["mlp"]["accuracy_mean"]

    head = f"{'層の数':>6} {'交差検証 mlp':>16} {'35 層との差':>10} {'linear':>8} {'全基準':>6} {'誤実行':>6} {'棄却再現':>8} {'test s7':>8} {'NPU 目安':>9} {'PLE 表':>8}"
    print("\n===== 層の数ごとの分類の精度（同じ 714 行・同じ基準）=====")
    print(head)
    md = ["|層の数（残す層）|交差検証 mlp|35 層との差|交差検証 linear|全基準（mlp）|誤実行（mlp）|棄却の再現率|test mlp s7|NPU の目安（T=64）|PLE の表の目安|",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, l, ev, de = r["crossval"]["mlp"], r["crossval"].get("linear", {}), r["evaluate"], r["device_estimate"]
        diff = m["accuracy_mean"] - base_acc
        t = ev.get("mlp_s7_test_accuracy")
        print(f"{r['n_layers']:>6} {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} {diff:>+10.4f} "
              f"{l.get('accuracy_mean', float('nan')):>8.4f} {m['configs_all_ok']}/{m['configs']:<4} {m['wrong_writes']:>6} "
              f"{m['rejection_recall_mean']:>8.3f} {t if t is None else f'{t:.4f}':>8} {de['npu_ms_T64']:>7.0f}ms {de['ple_mb']:>6.0f}MB")
        md.append(f"|{r['n_layers']}（0〜{r['n_layers'] - 1}）|{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}|{diff:+.4f}|"
                  f"{l.get('accuracy_mean', float('nan')):.4f}|{m['configs_all_ok']}/{m['configs']}|{m['wrong_writes']}|"
                  f"{m['rejection_recall_mean']:.3f}|{'—' if t is None else f'{t:.4f}'}|約 {de['npu_ms_T64']:.0f}ms|約 {de['ple_mb']:.0f}MB|")
    if ref["crossval"]:
        m = ref["crossval"]["mlp"]
        print(f"\n参考 G1（EdgeSLM_AndroidGemma4Head・35 層・同じ設定）: 交差検証 mlp {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}"
              f"・全基準 {m['configs_all_ok']}/{m['configs']}")
    # 35 層との差が 0.01 以内に収まる最小の層の数
    ok = [r["n_layers"] for r in rows if r["crossval"]["mlp"]["accuracy_mean"] >= base_acc - 0.01]
    best = max(rows, key=lambda r: r["crossval"]["mlp"]["accuracy_mean"]) if rows else None
    summ = {"base_accuracy_35": base_acc, "min_layers_within_0.01": min(ok) if ok else None,
            "best_layers": best["n_layers"] if best else None,
            "best_accuracy": best["crossval"]["mlp"]["accuracy_mean"] if best else None}
    print(f"\n35 層との差が 0.01 以内の最小の層の数: {summ['min_layers_within_0.01']} / 一番よい層の数: {summ['best_layers']}"
          f"（{summ['best_accuracy']:.4f}）" if best else "")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compare.json").write_text(json.dumps({"rows": rows, "g1_reference": ref, "summary": summ},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "compare_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"保存しました: {RESULTS / 'compare.json'}・compare_table.md")


if __name__ == "__main__":
    main()
