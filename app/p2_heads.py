# -*- coding: utf-8 -*-
"""P2 手順 3: 端末で集めた `h`（層 17・26・35、`app.p2_device collect`）でヘッドを学習し、精度を比べる。1 つ 3 分ほど。

設計: DoC/設計書_P2_層を減らしたグラフを端末で.md の 5 節。

改善版 B4（EdgeSLM_AndroidGemma4Head）・P1 と同じく、EdgeSLM_MultiHead_Multi の `train`・`evaluate`・`crossval` を**そのまま**呼ぶ。
比べるもの: 同じ層の数の PC の HF の `h`（P1）と、端末の `h`（P2）。端末の量子化で層を減らした効果が消えないかを見る。

    python -m app.p2_heads                    # 層 17・26・35（EdgeSLM_MultiHead_Multi の venv）
"""
from __future__ import annotations

import argparse
import json
import sys

from .common import PROJECT_ROOT, ROUTER_DATA, use_multi_library
from .p1_compare import summarize_crossval, summarize_evaluate

P2 = PROJECT_ROOT / "work" / "p2"
P1 = PROJECT_ROOT / "work" / "p1"
RES = PROJECT_ROOT / "results" / "P2"


def run(module, argv: list[str]) -> None:
    print(f"\n===== {module.__name__} =====", flush=True)
    sys.argv = [module.__name__] + argv
    module.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    use_multi_library()
    from edge_router.heads import crossval, evaluate, train
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    for n in ns:
        base = P2 / f"L{n:02d}" / "device"
        if not (base / "features" / "features_F2.npz").exists():
            print(f"層 {n}: 端末の h が無い（先に python -m app.p2_device collect）。飛ばす", flush=True)
            continue
        if (base / "crossval" / "crossval_report.json").exists() and not a.force:
            print(f"層 {n}: 作成済み（作り直すなら --force）", flush=True)
            continue
        print(f"\n########## 層 {n} 個（端末の h）##########", flush=True)
        common = ["--data", str(ROUTER_DATA), "--features", str(base / "features")]
        run(train, common + ["--out", str(base / "heads"), "--force"])
        run(evaluate, common + ["--run", str(base / "heads")])
        run(crossval, common + ["--out", str(base / "crossval")])

    rows = []
    print(f"\n{'h の出どころ':<26} {'交差検証 mlp':>16} {'全基準':>6} {'誤実行':>6} {'棄却再現':>8} {'test s7':>8}")
    for n in ns:
        for src, base in (("PC の HF（P1）", P1 / f"L{n:02d}"), ("端末（P2）", P2 / f"L{n:02d}" / "device")):
            cv = summarize_crossval(base / "crossval" / "crossval_report.json")
            ev = summarize_evaluate(base / "heads" / "evaluate_report.json")
            if not cv:
                continue
            m = cv["mlp"]
            t = ev.get("mlp_s7_test_accuracy")
            rows.append({"n_layers": n, "source": src, "crossval_mlp": m, "test_mlp_s7": t, "test_failed": ev.get("mlp_s7_checks_failed")})
            print(f"層 {n:>2}・{src:<18} {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} {m['configs_all_ok']}/{m['configs']:<4} "
                  f"{m['wrong_writes']:>6} {m['rejection_recall_mean']:>8.3f} {t if t is None else f'{t:.4f}':>8}", flush=True)
            if m["failed_checks"]:
                print(f"    満たさなかった基準: {m['failed_checks']}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "heads_compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存しました: {RES / 'heads_compare.json'}")


if __name__ == "__main__":
    main()
