# -*- coding: utf-8 -*-
"""Phase 1〜2: かたまり（chunk）のグラフを書き出し、配線を確かめる。

    python -m app.pipeline.export_chunk check    --layers 0-4 --head-ple    # 自前の配線 vs HF（丸めなしの正解）
    python -m app.pipeline.export_chunk export   --layers 0-4 --head-ple    # ONNX と spec
    python -m app.pipeline.export_chunk ortcheck --layers 0-4 --head-ple    # ONNX（ORT CPU）vs 自前の配線
    python -m app.pipeline.export_chunk export --with-head --with-hidden     # 出力層のかたまりに h を足す（本プロジェクト）

- 配線の確認は**丸めなし**の正解（`chunk_a_b_nosrq_L256.npz`）と比べる（自前の配線は丸めを掛けないため）
- 量子化の較正と、実機との比較は**丸めあり**（`chunk_a_b_L256.npz`）を使う（端末は QAT の 8bit の丸めを再現する）

★EdgeSLM_AndroidGemma4Head での改修（2026-09-27。設計: DoC/設計書_G2_hの出力を持つcontext_binary.md の 4.2 節）:
  - 状態（`capture/*.npz`）は EdgeSLM_AndroidGemma4 の work から読む（`G.CAPTURE`。読むだけ）
  - `--with-hidden`: 出力層のかたまり（`--with-head`）に、最後の正規化の出力 `h` を 2 本目の出力として足す（名前 `g4headh_L{L}`）
  - `hcheck`: `h` を足したかたまりの logits が、足さないかたまりと完全に同じか・`h` が最後の正規化の出力そのものかを確かめる
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from app.pipeline import gemma4_common as G
from app.pipeline import gemma4_graph as M

WORK_OPS = G.WORK / "ops"


def name_for(lo: int, hi: int, L: int, head: bool, with_head: bool = False, with_hidden: bool = False) -> str:
    if with_head:
        return f"g4head{'h' if with_hidden else ''}_L{L}"
    return f"g4c{lo}_{hi}{'h' if head else ''}_L{L}"


def npz_path(lo: int, hi: int, L: int, srq: bool, with_head: bool = False) -> Path:
    if with_head:
        return G.CAPTURE / f"head{'' if srq else '_nosrq'}_L{L}.npz"
    return G.CAPTURE / f"chunk_{lo}_{hi}{'' if srq else '_nosrq'}_L{L}.npz"


def load_states(lo: int, hi: int, L: int, srq: bool, with_head: bool = False) -> dict:
    z = np.load(npz_path(lo, hi, L, srq, with_head))
    s = {k: z[k] for k in z.files}
    # ★入口の無いかたまりは PLE の値（`ple_all`）を入力で受け取る。値はトークンだけで決まるので、
    #   入口のかたまり（A）の正解から読む（同じ会話・同じ位置で取っているので順番も同じ）
    if "ple_all" not in s and "ref_ple_all" not in s and not with_head:
        head = G.CAPTURE / f"chunk_0_8{'' if srq else '_nosrq'}_L{L}.npz"
        if head.exists():
            za = np.load(head)
            if len(za["pos"]) == len(s["pos"]) and np.array_equal(za["pos"], s["pos"]):
                s["ple_all"] = za["ref_ple_all"]
    if "ple_all" not in s and "ref_ple_all" in s:
        s["ple_all"] = s["ref_ple_all"]
    return s


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def build_module(lo: int, hi: int, head: bool, with_head: bool = False, with_hidden: bool = False):
    qf = M.QatFile()
    layers = [] if with_head else list(range(lo, hi + 1))
    return M.DecodeChunk(qf, layers, with_ple_proj=head, with_head=with_head, with_hidden=with_hidden).eval()


def run_module(m, s: dict, idx: int) -> list[np.ndarray]:
    args = [torch.from_numpy(s[n][idx:idx + 1].astype(np.float32)) for n in m.input_names()]
    with torch.no_grad():
        out = m(*args)
    return [o.numpy() for o in out]


def check(lo: int, hi: int, L: int, head: bool, n: int, srq: bool, with_head: bool = False):
    s = load_states(lo, hi, L, srq, with_head)
    m = build_module(lo, hi, head, with_head)
    names = m.output_names()
    ref_key = {"hidden_out": "ref_hidden_out", "ple_all": "ref_ple_all"}
    N = len(s["pos"])
    rows = []
    for i in np.linspace(0, N - 1, min(n, N)).astype(int):
        o = run_module(m, s, int(i))
        r = {"i": int(i), "pos": int(s["pos"][i])}
        for j, nm in enumerate(names):
            key = ref_key.get(nm, "ref_" + nm)
            if key in s:
                r[nm] = rel(o[j], s[key][i:i + 1])
        rows.append(r)
    worst = {k: max(r[k] for r in rows) for k in rows[0] if k not in ("i", "pos")}
    print(f"自前の配線 vs HF（{len(rows)} サンプル、SRQ={'あり' if srq else 'なし'}、相対L2 の最大）: "
          + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    out = G.RESULTS / "P1_20260920_chunk" / f"check_{name_for(lo, hi, L, head)}{'' if srq else '_nosrq'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"srq": srq, "worst": worst, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


def hcheck(L: int, n: int) -> dict:
    """`h` を足した出力層のかたまりを確かめる（本プロジェクトで追加）。

    1. logits が、`h` を足さないかたまりと**完全に同じ**（出力を足しても計算は変わらない）
    2. `h` が最後の正規化の出力そのもの（`final_norm(hidden_in)` と一致）で、`lm_head(h)` が logits と同じ
    3. HF（丸めあり）の 1 位（`ref_argmax`）との一致の数（参考。出力層の丸めで同点がありうる）
    """
    s = load_states(0, 0, L, True, True)
    m0 = build_module(0, 0, False, True, False)
    m1 = build_module(0, 0, False, True, True)
    N = len(s["pos"])
    worst_logit, worst_h, top1 = 0.0, 0.0, 0
    idx = np.linspace(0, N - 1, min(n, N)).astype(int)
    for i in idx:
        x = torch.from_numpy(s["hidden_in"][i:i + 1].astype(np.float32))
        with torch.no_grad():
            (lg0,) = m0(x)
            lg1, h = m1(x)
            hn = m1.final_norm(x)
            lg_h = m1.lm_head(h)
        worst_logit = max(worst_logit, float((lg0 - lg1).abs().max()), float((lg_h - lg1).abs().max()))
        worst_h = max(worst_h, float((h - hn).abs().max()))
        top1 += int(int(lg1.reshape(-1).argmax()) == int(s["ref_argmax"][i]))
    res = {"samples": len(idx), "logits_max_abs_diff": worst_logit, "h_vs_final_norm_max_abs_diff": worst_h,
           "top1_vs_hf": f"{top1}/{len(idx)}", "outputs": m1.output_names()}
    print(f"h を足したかたまり: logits の差の最大 {worst_logit:.2e} / h と最後の正規化の差 {worst_h:.2e} / "
          f"HF の 1 位との一致 {top1}/{len(idx)} / 出力 {m1.output_names()}")
    out = G.RESULTS / "G2" / "hcheck_head.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if worst_logit > 1e-4 or worst_h > 1e-5:
        raise SystemExit("h を足したかたまりが元と一致しない。止める")
    return res


def export(lo: int, hi: int, L: int, head: bool, with_head: bool = False, calib_max: int = 64,
           with_hidden: bool = False):
    s = load_states(lo, hi, L, True, with_head)
    m = build_module(lo, hi, head, with_head, with_hidden)
    name = name_for(lo, hi, L, head, with_head, with_hidden)
    d = WORK_OPS / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.onnx"
    args = tuple(torch.from_numpy(s[n][0:1].astype(np.float32)) for n in m.input_names())
    t0 = time.time()
    torch.onnx.export(m, args, str(path), input_names=m.input_names(), output_names=m.output_names(),
                      dynamic_axes=None, opset_version=21, do_constant_folding=True, dynamo=True)
    import onnx
    mo = onnx.load(str(path), load_external_data=False)
    print(f"書き出し {time.time() - t0:.0f}s  nodes={len(mo.graph.node)}  ops={sorted({n.op_type for n in mo.graph.node})}")
    print("  入力:", [(v.name, [dd.dim_value for dd in v.type.tensor_type.shape.dim]) for v in mo.graph.input])
    print("  出力:", [(v.name, [dd.dim_value for dd in v.type.tensor_type.shape.dim]) for v in mo.graph.output])
    # ★較正は代表サンプルだけ（全部だと入力の配列だけで 1GB を超え、PC のメモリが厳しい。値の範囲を取るには十分）
    calib = d / f"{name}_calib.npz"
    if not calib.exists():
        n_all = len(s["pos"])
        sel = np.linspace(0, n_all - 1, min(calib_max, n_all)).astype(int)
        np.savez(calib, **{nm: s[nm][sel] for nm in m.input_names()})
        print(f"  較正データ {len(sel)} サンプル -> {calib}（{calib.stat().st_size / 2**20:.0f}MiB）")
    spec = {"model": str(path).replace("\\", "/"),
            "note": (f"Gemma 4 E2B 最後の正規化＋出力層{'＋h の出力' if with_hidden else ''} decode L={L}" if with_head
                     else f"Gemma 4 E2B 層{lo}〜{hi}{'＋入口' if head else ''} decode L={L}"),
            "per_channel_weights": True, "weight_bits": 4,
            "calib_npz": str(calib).replace("\\", "/")}
    (d / f"{name}.spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path}\n-> {d / f'{name}.spec.json'}")


def ortcheck(lo: int, hi: int, L: int, head: bool, n: int, with_head: bool = False):
    import onnxruntime as ort
    s = load_states(lo, hi, L, True, with_head)
    m = build_module(lo, hi, head, with_head)
    name = name_for(lo, hi, L, head, with_head)
    sess = ort.InferenceSession(str(WORK_OPS / name / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    N = len(s["pos"])
    worst = 0.0
    for i in np.linspace(0, N - 1, min(n, N)).astype(int):
        feeds = {nm: s[nm][i:i + 1].astype(np.float32) for nm in m.input_names()}
        worst = max(worst, max(rel(a, b) for a, b in zip(sess.run(None, feeds), run_module(m, s, int(i)))))
    print(f"ONNX（ORT CPU）vs 自前の配線: 相対L2 の最大 {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "export", "ortcheck", "hcheck"])
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--head-ple", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--srq", action="store_true", help="check で丸めありの正解と比べる")
    ap.add_argument("--with-head", action="store_true", help="最後の正規化＋出力層だけのかたまり")
    ap.add_argument("--calib-max", type=int, default=64, help="較正に使うサンプル数")
    ap.add_argument("--with-hidden", action="store_true", help="出力層のかたまりに h（最後の正規化の出力）を足す")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))
    if a.cmd == "check":
        check(lo, hi, a.L, a.head_ple, a.n, a.srq, a.with_head)
    elif a.cmd == "export":
        export(lo, hi, a.L, a.head_ple, a.with_head, a.calib_max, a.with_hidden)
    elif a.cmd == "hcheck":
        hcheck(a.L, a.n)
    else:
        ortcheck(lo, hi, a.L, a.head_ple, a.n, a.with_head)


if __name__ == "__main__":
    main()
