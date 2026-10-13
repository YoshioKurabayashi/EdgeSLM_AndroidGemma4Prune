# -*- coding: utf-8 -*-
"""Phase 2: 繋いだグラフで、PC（ORT CPU）のまま**文章を作ってみる**。

    python -m app.pipeline.gen_pc --q "日本の首都はどこですか。" --max 24

なぜ要るか: 1トークンごとの「1位の一致」は 18/24 だった（P2 5.2）。ただし食い違うのは
**HF 自身が迷っている位置**が多い。最後に効くのは「読める日本語になるか」なので、実機へ行く前に PC で確かめる。
ここで作る手順は、そのまま端末側（C++）の生成ループの仕様になる。

1ステップでやること（端末と同じ）:

| もの | 作り方 |
|---|---|
| `inputs_embeds` | 主の埋め込み（2bit・行ごと）を1行引いて 整数×スケール ×√1536 |
| `ple_tok` | PLE の表（4bit・語ごと層ごと）を1行引いて 整数×スケール ×√256 → [1,1,35,256] |
| 位置の表 | `rope_cos_s/sin_s`（局所 256次元）・`cos_g/sin_g`（全体 512次元）の、その位置の行 |
| `attn_mask` | 入っているぶんだけ 0、残りは −60（C-9: −inf は量子化できない） |
| `past_k_i`/`past_v_i` | 前のステップの `present_*` をそのまま渡す（ずらしはグラフの中でやる） |

CPU では 1トークン 8〜9秒（35層＋262144語の出力層を戻しながら計算するため）。実機の速さとは別物。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.pipeline import gemma4_common as G
from app.pipeline import gemma4_graph as M

TABLES = G.TABLES          # ★EdgeSLM_AndroidGemma4Head: 表は EdgeSLM_AndroidGemma4 の work を読む（2026-09-27）
STOP = {106, 1, 50}            # <turn|> / <eos> / <end_of_turn> 相当（P0A）
SYS = "あなたは車載アシスタントです。短く答えてください。"          # P0B のデモと同じ（答えを直に比べられる）


class Tables:
    """CPU 側で引く表（2bit の埋め込み、4bit の PLE、位置の表）。必要な行だけ読む。"""

    def __init__(self):
        self.meta = json.loads((TABLES / "tables.json").read_text(encoding="utf-8"))
        e, p = self.meta["embed"], self.meta["ple"]
        self.e, self.p = e, p
        self.emb = np.memmap(TABLES / e["file"], np.uint8, "r", shape=(e["rows"], e["packed_cols"]))
        self.emb_s = np.memmap(TABLES / e["scale_file"], np.float32, "r", shape=(e["rows"],))
        self.ple = np.memmap(TABLES / p["file"], np.uint8, "r", shape=(p["rows"], p["packed_cols"]))
        self.ple_s = np.memmap(TABLES / p["scale_file"], np.float32, "r", shape=(p["rows"], p["layers"]))
        self.rope = {k: np.fromfile(TABLES / f"rope_{k}.bin", np.float32).reshape(self.meta["rope"]["rows"], -1)
                     for k in self.meta["rope"]["dims"]}

    def embed(self, tid: int) -> np.ndarray:
        q = np.empty(self.e["cols"], np.float32)
        b = np.asarray(self.emb[tid], np.uint8)
        for k in range(4):                                   # 下位から 2bit ずつ 4個、0..3 から 2 を引く
            q[k::4] = ((b >> (2 * k)) & 0x03).astype(np.float32) - 2.0
        return (q * float(self.emb_s[tid]) * self.e["embed_scale"]).reshape(1, 1, -1)

    def ple_tok(self, tid: int) -> np.ndarray:
        b = np.asarray(self.ple[tid], np.uint8)
        q = np.empty(self.p["cols"], np.float32)
        q[0::2] = (b & 0x0F).astype(np.float32) - 8.0        # 下位4bit → 上位4bit
        q[1::2] = (b >> 4).astype(np.float32) - 8.0
        q = q.reshape(self.p["layers"], self.p["dim"])
        return (q * np.asarray(self.ple_s[tid], np.float32)[:, None] * self.p["embed_scale"]).reshape(1, 1, *q.shape)


def build_prompt(tok, question: str) -> list[int]:
    """P0B（`ref_qat.py` の `first_turn_ids`）と同じ作り方にそろえる（`apply_chat_template`）。"""
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def run(question: str, max_new: int, model: Path, L: int = 256, show: bool = True) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(G.HF_DIR), local_files_only=True)
    tb = Tables()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    ins = {v.name: [d if isinstance(d, int) else 1 for d in v.shape] for v in sess.get_inputs()}
    outs = [v.name for v in sess.get_outputs()]
    cache = {nm: np.zeros(sh, np.float32) for nm, sh in ins.items() if nm.startswith("past_")}
    ids = build_prompt(tok, question)
    print(f"入力 {len(ids)} トークン: {tok.decode(ids[1:])[:80]}…", flush=True)
    gen, filled, t0 = [], 0, time.time()
    for step in range(len(ids) + max_new):
        tid = ids[step] if step < len(ids) else gen[-1]
        pos = step
        feed = {"inputs_embeds": tb.embed(tid), "ple_tok": tb.ple_tok(tid)}
        for k, arr in tb.rope.items():
            feed[k] = arr[pos].reshape(1, 1, 1, -1).astype(np.float32)
        mask = np.full((1, 1, 1, L), M.MASK_NEG, np.float32)
        filled = min(filled + 1, L)
        mask[..., L - filled:] = 0.0
        feed["attn_mask"] = mask
        feed.update(cache)
        o = sess.run(None, feed)
        for nm, v in zip(outs, o):
            if nm.startswith("present_"):
                cache["past_" + nm[len("present_"):]] = np.asarray(v, np.float32)
        if step >= len(ids) - 1:
            nxt = int(np.asarray(o[outs.index("logits")]).reshape(-1).argmax())
            gen.append(nxt)
            if show:
                print(f"  [{len(gen):>2}] {nxt:>6} {tok.decode([nxt])!r}", flush=True)
            if nxt in STOP:
                break
    txt = tok.decode([t for t in gen if t not in STOP])
    dt = time.time() - t0
    print(f"\n答え: {txt}\n（{len(ids)}+{len(gen)} ステップ / {dt:.0f}s = {dt / max(len(ids) + len(gen), 1):.1f}s/トークン）")
    return {"question": question, "answer": txt, "ids": gen, "sec": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="日本の首都はどこですか。")
    ap.add_argument("--max", type=int, default=24)
    ap.add_argument("--model", type=Path, default=G.WORK / "ops" / "g4full_L256" / "g4full_L256_final_qdq.onnx")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    r = run(a.q, a.max, a.model)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
