# -*- coding: utf-8 -*-
"""P1 手順 2: 層の数ごとに、その `h` でヘッドを学習 → 評価 → 5 分割の交差検証。1 つの層の数で 3 分ほど（CPU/GPU）。

設計: DoC/設計書_P1_層ごとの分類性能.md の 5 節。

EdgeSLM_MultiHead_Multi の `edge_router.heads.train / evaluate / crossval` を**そのまま**・同じ既定値
（linear・mlp × seed 7/17/29、判定の基準 configs/evaluation.yaml）で呼ぶ（EdgeSLM_AndroidGemma4Head の G1・B4・C3 と同じ）。
変えるのは入力の `h` だけ（層を n 個残したモデルの h）。

|段|何をするか|
|---|---|
|`prepare`|`work/p1/all_layers.npz` から層の数 n の `h` を切り出し、`work/p1/L<n>/features/features_F2.npz`（edge_router の特徴の形式）に書く|
|`heads`|`work/p1/L<n>/heads`（学習・評価）と `crossval`（交差検証）|

    python -m app.p1_heads                        # 既定の層の数（4,6,8,10,12,14,15,17,20,24,28,35）
    python -m app.p1_heads --layers 14,15,16      # 足す・やり直す
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .common import DEFAULT_LAYERS, ROUTER_DATA, WORK, layer_dir, use_multi_library, write_provenance


def prepare(n: int) -> None:
    z = np.load(WORK / "all_layers.npz", mmap_mode="r")
    meta = json.loads((WORK / "all_layers.meta.json").read_text(encoding="utf-8"))
    d = layer_dir(n) / "features"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "features_F2.npz", last__last=np.asarray(z["last"][n - 1], np.float32),
             last__mean=np.asarray(z["mean"][n - 1], np.float32))
    m = {"format": "F2", "ids": meta["ids"], "layers": ["last"], "poolings": ["last", "mean"],
         "meta": {"source": f"Gemma 4 E2B QAT（SRQ あり）の層 0〜{n - 1}（{n} 層）の出力＋最後の正規化", "n_layers_kept": n,
                  "hidden_size": meta["hidden_size"], "final_norm_applied": True}}
    (d / "features_F2.meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "token_ids.json").write_text((WORK / "token_ids.json").read_text(encoding="utf-8"), encoding="utf-8")


def run(module, argv: list[str]) -> None:
    print(f"\n===== {module.__name__} =====", flush=True)
    sys.argv = [module.__name__] + argv
    module.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--force", action="store_true", help="作成済みのヘッド・交差検証も作り直す")
    a = ap.parse_args()
    if not (WORK / "all_layers.npz").exists():
        raise SystemExit("work/p1/all_layers.npz が無い。先に python -m app.p1_extract_layers")
    use_multi_library()
    from edge_router.heads import crossval, evaluate, train
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    t_all = time.time()
    for n in ns:
        print(f"\n########## 層 {n} 個（層 0〜{n - 1}）##########", flush=True)
        base = layer_dir(n)
        cvr = base / "crossval" / "crossval_report.json"
        if cvr.exists() and (base / "heads" / "evaluate_report.json").exists() and not a.force:
            print("作成済み（作り直すなら --force）", flush=True)
            continue
        t0 = time.time()
        prepare(n)
        common = ["--data", str(ROUTER_DATA), "--features", str(base / "features")]
        run(train, common + ["--out", str(base / "heads"), "--force"])
        run(evaluate, common + ["--run", str(base / "heads")])
        run(crossval, common + ["--out", str(base / "crossval")])
        print(f"[層 {n} 個] {time.time() - t0:.0f} 秒（通算 {time.time() - t_all:.0f} 秒）", flush=True)
    write_provenance("heads", {"layers": ns})


if __name__ == "__main__":
    main()
