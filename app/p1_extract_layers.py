# -*- coding: utf-8 -*-
"""P1 手順 1: Gemma 4 E2B（QAT）の**35 層それぞれの出力**を、ルーターの 714 行について取り出す。GPU で数分。

設計: DoC/設計書_P1_層ごとの分類性能.md の 4 節。

- モデルの読み方・符号化は EdgeSLM_AndroidGemma4Head の `app/g1_extract.py`（md5 91e29bfc）と同じ（SRQ あり＝端末と同じ・
  F2 の文を Gemma 4 のチャットテンプレート（system なし）で包む・1 行ずつ・パディングなし）。関数を写した（元のファイルは読まない・変えない）
- 各層（decoder layer）の出力をフックで取り、**最後の正規化（`norm`）を掛けたもの**を「層 0〜n-1 だけ残したモデルの h」とみなす
  （層を切ったモデル＝層 0〜n-1 ＋最後の正規化。分類ヘッドはその出力を受ける）
- 最後のトークン（行き先・ArgMode）と全トークンの平均（棄却の後半）を、35 層ぶん保存する（`work/p1/all_layers.npz`、約 300MB）
- 確かめ: 35 層目に正規化を掛けたもの ＝ モデルの `last_hidden_state`（同じ回の値）、また G1 の特徴（同じ設定で以前に取ったもの）とも比べる

    python -m app.p1_extract_layers
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .common import G1_SRQ, HF_DIR, MODEL_ID, N_LAYERS, ROUTER_DATA, WORK, env_offline, use_multi_library, write_provenance


def load(device: str):
    """EdgeSLM_AndroidGemma4Head `app/g1_extract.load`（SRQ あり）と同じ"""
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    model = AutoModelForMultimodalLM.from_pretrained(str(HF_DIR), dtype=torch.float32, local_files_only=True).eval()
    model.to(device)
    inner = getattr(model, "model", model)
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    return tok, model, lm


def encode(tok, text: str) -> list[int]:
    """EdgeSLM_AndroidGemma4Head `app/g1_extract.encode` と同じ"""
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    env_offline()
    use_multi_library()
    from edge_router.data.build import render
    from edge_router.io_utils import read_jsonl

    WORK.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(ROUTER_DATA)
    if a.limit:
        rows = rows[:a.limit]
    t0 = time.time()
    tok, model, lm = load(a.device)
    layers = lm.layers[:N_LAYERS]
    print(f"モデル {MODEL_ID}（SRQ あり）読み込み {time.time() - t0:.0f} 秒 / 層 {len(lm.layers)} / {type(lm).__name__}", flush=True)

    got: list = [None] * N_LAYERS

    def hook(i):
        def f(_m, _inp, out):
            got[i] = out[0] if isinstance(out, (tuple, list)) else out
        return f
    hs = [ly.register_forward_hook(hook(i)) for i, ly in enumerate(layers)]

    ids_all = [encode(tok, render(r, "F2")) for r in rows]
    H = lm.config.hidden_size
    last = np.zeros((N_LAYERS, len(rows), H), np.float32)
    mean = np.zeros((N_LAYERS, len(rows), H), np.float32)
    chk = []
    t1 = time.time()
    try:
        with torch.no_grad():
            for k, ids in enumerate(ids_all):
                out = lm(input_ids=torch.tensor([ids], device=a.device), use_cache=False).last_hidden_state[0]
                for i in range(N_LAYERS):
                    hn = lm.norm(got[i])[0]                     # 層 0〜i だけ残したモデルの h
                    last[i, k] = hn[-1].float().cpu().numpy()
                    mean[i, k] = hn.mean(0).float().cpu().numpy()
                if k < 4:
                    chk.append(rel(last[N_LAYERS - 1, k], out[-1].float().cpu().numpy()))
                if (k + 1) % 100 == 0:
                    print(f"  {k + 1}/{len(rows)} 行（{time.time() - t1:.0f} 秒）", flush=True)
    finally:
        for h in hs:
            h.remove()
    print(f"確かめ 1: 35 層目に最後の正規化 ↔ モデルの last_hidden_state 相対L2 最大 {max(chk):.2e}", flush=True)
    if max(chk) > 1e-5:
        raise SystemExit("35 層目の取り方がモデルの出力と合わない。フックの場所を見直す")
    info = {"check_last_hidden_rel_max": max(chk)}
    g1 = G1_SRQ / "features"
    if (g1 / "features_F2.npz").exists() and not a.limit:
        meta = json.loads((g1 / "features_F2.meta.json").read_text(encoding="utf-8"))
        ref = np.load(g1 / "features_F2.npz")
        if meta["ids"] == [r["id"] for r in rows]:
            r1 = np.linalg.norm(last[-1] - ref["last__last"], axis=1) / (np.linalg.norm(ref["last__last"], axis=1) + 1e-12)
            r2 = np.linalg.norm(mean[-1] - ref["last__mean"], axis=1) / (np.linalg.norm(ref["last__mean"], axis=1) + 1e-12)
            info.update({"vs_g1_last_rel_max": float(r1.max()), "vs_g1_mean_rel_max": float(r2.max())})
            print(f"確かめ 2: 35 層目 ↔ G1 の特徴（EdgeSLM_AndroidGemma4Head、読むだけ）最後 最大 {r1.max():.2e}・"
                  f"平均 最大 {r2.max():.2e}", flush=True)
    norms = [float(np.linalg.norm(last[i], axis=1).mean()) for i in range(N_LAYERS)]
    np.savez(WORK / "all_layers.npz", last=last, mean=mean)
    meta = {"format": "F2", "ids": [r["id"] for r in rows], "n_layers": N_LAYERS, "hidden_size": H,
            "note": "last[i]・mean[i] は層 0〜i の出力に最後の正規化を掛けたもの（層を i+1 個残したモデルの h）",
            "srq": True, "last_norm_by_layer": norms, **info,
            "token_count_mean": float(np.mean([len(i) for i in ids_all])), "elapsed_sec": round(time.time() - t1, 1)}
    (WORK / "all_layers.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (WORK / "token_ids.json").write_text(json.dumps(dict(zip(meta["ids"], ids_all))), encoding="utf-8")
    print(f"{len(rows)} 行 × {N_LAYERS} 層 / {time.time() - t1:.0f} 秒 → {WORK / 'all_layers.npz'}", flush=True)
    write_provenance("extract", {"rows": len(rows), **info})


if __name__ == "__main__":
    main()
