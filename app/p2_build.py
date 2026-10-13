# -*- coding: utf-8 -*-
"""P2 手順 1: 層を減らした分類専用のまとめ読みグラフ（T=64・KV なし・出力 h）を作る。層 17（0〜16）と層 26（0〜25）。

設計: DoC/設計書_P2_層を減らしたグラフを端末で.md の 3 節。

EdgeSLM_AndroidGemma4Head の改善版 B（35 層、`app/b2_build_full.py`）と同じ作り方で、**後ろの層を外し、最後の正規化を
残した層の直後に付ける**。前の方のかたまりは改善版 B の量子化済みのものを**読むだけ**で使い回し（書き込まない）、
新しく作るのは「最後のかたまり（層の途中まで＋最後の正規化）」だけ。

|グラフ|かたまり|作るもの|
|---|---|---|
|`p2_L17`（層 0〜16）|A（入口＋0〜8・改善版 B のもの）＋ **N17（9〜16＋最後の正規化）**|N17|
|`p2_L26`（層 0〜25）|A ＋ B（9〜17・改善版 B のもの）＋ **N26（18〜25＋最後の正規化）**|N26|

- PLE（層ごとの埋め込み）の入力 `ple_tok` は 35 層ぶんのまま（入口のかたまり A を使い回すため）。後ろのかたまりは必要な層の分だけ使う。
  端末の PLE の表も 35 層ぶんのものをそのまま使う（表は CPU 側で読むだけで、NPU のメモリには載らない）
- 層 15・16 は層 13・14 の K/V を使い回す。N17 は 13・14 を自分の中で作る。N26 は B が作った K/V を入力で受ける
- 較正のデータは改善版 B の較正（`EdgeSLM_AndroidGemma4Head/work/b2_calib`。かたまりごとの入力。読むだけ）をそのまま使う
  （N17 の入力 ＝ B の入力、N26 の入力 ＝ C の入力）
- 量子化の手順は改善版 B と同じ（a16w8_rhs8・QAT の値の書き込み・量子化されない入力に Q/DQ・act×act の右辺を 8bit）。
  変換の道具は EdgeSLM_AndroidGemma4Head の `app/pipeline/`・`scripts/build_context.sh` を本プロジェクトへ写したもの

|段|何をするか|
|---|---|
|`chunks`|N17・N26 を 書き出し → 量子化 → QAT の値 → `fix_float_inputs` → 右辺を 8bit（`work/ops/p2cls64_c*`）|
|`merge`|かたまりを繋ぐ。出力は `h` だけ（`work/ops/p2_L<n>/p2_L<n>_final_qdq.onnx`）|
|`ortcheck`|繋いだ QDQ を PC の ORT（CPU）で 8 行。`h` を P1 の HF の値（同じ層の数・SRQ あり）と比べる（参考。有限か）|
|`convert`|QAIRT（`QAIRT_INT4_PATCH=1`）→ EPContext。層 17 で 1〜1.5 時間、層 26 で 2〜3 時間の見込み|

    python -m app.p2_build --steps chunks,merge,ortcheck --layers 17,26     # PC（30 分ほど）
    python -m app.p2_build --steps convert --layers 17,26                   # PC（数時間）
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from app.common import HEAD_PRJ, PROJECT_ROOT
from app.pipeline import gemma4_common as G
from app.pipeline import gemma4_graph as M
from app.pipeline.gemma4_prefill import PrefillChunk, causal_mask, PAD_ID

PY = sys.executable
SCHEME = "a16w8_rhs8"
T = 64
OPS = PROJECT_ROOT / "work" / "ops"
LOGS = PROJECT_ROOT / "work" / "logs"
RES = PROJECT_ROOT / "results" / "P2"
HEAD_OPS = HEAD_PRJ / "work" / "ops"                        # 読むだけ（改善版 B の量子化済みのかたまり）
HEAD_CAL = HEAD_PRJ / "work" / "b2_calib"                   # 読むだけ（改善版 B の較正のデータ）
P1 = PROJECT_ROOT / "work" / "p1"
G1_TOK = HEAD_PRJ / "work" / "g1" / "srq" / "features" / "token_ids.json"

A = HEAD_OPS / "g4cls64_c0_8" / "g4cls64_c0_8_final_qdq.onnx"
B = HEAD_OPS / "g4cls64_c9_17" / "g4cls64_c9_17_final_qdq.onnx"
# 層の数 → (使い回すかたまり, 新しく作るかたまり (lo, hi), その入力の較正のデータ)
PLAN = {
    17: {"reuse": [A], "new": (9, 16), "calib": HEAD_CAL / "g4cls64_c9_17_calib.npz", "refs": [A]},
    26: {"reuse": [A, B], "new": (18, 25), "calib": HEAD_CAL / "g4cls64_c18_22_calib.npz", "refs": [A, B]},
}
KV_NAMES = {"ple_all", "shared_k_13", "shared_v_13", "shared_k_14", "shared_v_14"}


def cname(n: int) -> str:
    lo, hi = PLAN[n]["new"]
    return f"p2cls{T}_c{lo}_{hi}n"


def full(n: int) -> str:
    return f"p2_L{n}"


def run(cmd: list, log: str, env: dict | None = None) -> None:
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / log).write_text((r.stdout or "") + "\n---- stderr ----\n" + (r.stderr or ""), encoding="utf-8")
    tail = "\n    ".join((r.stdout or "").strip().splitlines()[-2:])
    print(f"    {time.time() - t0:.0f}s  {tail}", flush=True)
    if r.returncode != 0:
        print((r.stderr or "")[-1500:])
        raise SystemExit(f"失敗: {log}")


def check_inputs() -> None:
    miss = [str(p) for p in (A, B, HEAD_CAL / "g4cls64_c9_17_calib.npz", HEAD_CAL / "g4cls64_c18_22_calib.npz") if not p.exists()]
    if miss:
        raise SystemExit(f"EdgeSLM_AndroidGemma4Head の改善版 B のファイルが無い（読むだけで使う）: {miss}")


# ---------------------------------------------------------------- chunks

def chunk(n: int, force: bool) -> None:
    lo, hi = PLAN[n]["new"]
    nm = cname(n)
    d = OPS / nm
    final = d / f"{nm}_final_qdq.onnx"
    if final.exists() and not force:
        print(f"[N{n}] {nm}: 作成済み（作り直すなら --force）", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    M.SRQ = False
    qf = M.QatFile()
    mod = PrefillChunk(qf, list(range(lo, hi + 1)), with_ple_proj=False, with_norm=True).eval()
    names = mod.input_names()
    cal_src = dict(np.load(PLAN[n]["calib"]))
    miss = [k for k in names if k not in cal_src]
    if miss:
        raise SystemExit(f"較正のデータに入力が無い: {miss}（{PLAN[n]['calib'].name}）")
    cal = {k: cal_src[k] for k in names}
    args = tuple(torch.from_numpy(cal[k][:1]) for k in names)
    print(f"[N{n}] {nm}: 層 {lo}〜{hi}＋最後の正規化 書き出し（入力 {names}・出力 {mod.output_names()}）", flush=True)
    torch.onnx.export(mod, args, str(d / f"{nm}.onnx"), input_names=names, output_names=mod.output_names(),
                      opset_version=21, dynamo=True, external_data=True)
    np.savez(d / f"{nm}_calib.npz", **cal)
    spec = {"model": str(d / f"{nm}.onnx").replace("\\", "/"),
            "note": f"層を減らした分類専用まとめ読み T={T} 層{lo}〜{hi}＋最後の正規化（P2・層 {n} 個のグラフの最後のかたまり）",
            "calib_npz": str(d / f"{nm}_calib.npz").replace("\\", "/"), "per_channel_weights": True, "weight_bits": 4}
    (d / f"{nm}.spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    del mod
    gc.collect()
    try:
        run([PY, "-m", "app.pipeline.p2b_ops", "prepare", "--work", "work/ops", "--name", nm, "--scheme", SCHEME],
            f"p2_prepare_N{n}.log")
    except SystemExit:
        # 改善版 B と同じ: 量子化されない入力（ple_all・層 13/14 の K/V。C-1）だけなら、後で Q/DQ をはさんで解消する
        st = json.loads((d / f"status_prepare_{SCHEME}.json").read_text(encoding="utf-8"))
        vio = st.get("checks", {}).get("C-1", {}).get("violations", [])
        if not vio or any(v.get("tensor") not in KV_NAMES for v in vio) or not st["stages"]["S4_quantize"]["ok"]:
            raise
        print(f"    C-1 の指摘は量子化されない入力だけ（{len(vio)} 件 {sorted({v['tensor'] for v in vio})}）→ この後 Q/DQ をはさむ",
              flush=True)
    ort_qdq = d / f"{nm}_{SCHEME}_qdq_ort.onnx"
    if not ort_qdq.exists():
        (d / f"{nm}_{SCHEME}_qdq.onnx").rename(ort_qdq)
    run([PY, "-m", "app.pipeline.qat_inject", "--qdq", str(ort_qdq), "--prepped", str(d / f"{nm}_prepped.onnx"),
         "--layers", f"{lo}-{hi}", "--out", str(d / f"{nm}_{SCHEME}_qdq.onnx")], f"p2_inject_N{n}.log")
    refs = []
    for r in PLAN[n]["refs"]:
        refs += ["--ref", str(r)]
    run([PY, "-m", "app.pipeline.fix_float_inputs", "--qdq", str(d / f"{nm}_{SCHEME}_qdq.onnx"),
         "--out", str(d / f"{nm}_{SCHEME}_qdq_fixin.onnx"), *refs], f"p2_fixin_N{n}.log")
    run([PY, "-m", "app.pipeline.downgrade_actact_rhs", "--qdq", str(d / f"{nm}_{SCHEME}_qdq_fixin.onnx"),
         "--prepped", str(d / f"{nm}_prepped.onnx"), "--out", str(final)], f"p2_downgrade_N{n}.log")


# ---------------------------------------------------------------- merge

def merge(n: int) -> None:
    parts = PLAN[n]["reuse"] + [OPS / cname(n) / f"{cname(n)}_final_qdq.onnx"]
    miss = [str(p) for p in parts if not p.exists()]
    if miss:
        raise SystemExit(f"繋ぐかたまりが無い: {miss}")
    drop = ["ple_all"] + ([f"c1_present_{kv}_{i}" for i in (13, 14) for kv in ("k", "v")] if len(parts) == 3 else [])
    out = OPS / full(n) / f"{full(n)}_final_qdq.onnx"
    run([PY, "-m", "app.pipeline.merge_chunks", "--chunks", ",".join(str(p) for p in parts), "--out", str(out),
         "--drop", ",".join(drop)], f"p2_merge_L{n}.log")
    import onnx
    g = onnx.load(str(out), load_external_data=False).graph
    ins, outs = [v.name for v in g.input], [v.name for v in g.output]
    print(f"  [L{n}] 入力 {ins} / 出力 {outs}", flush=True)
    if outs != ["h"]:
        raise SystemExit(f"出力が h の 1 本になっていない: {outs}")


# ---------------------------------------------------------------- ortcheck

def pick_rows(k: int) -> list:
    toks = json.loads(G1_TOK.read_text(encoding="utf-8"))
    items = sorted(toks.items(), key=lambda kv: len(kv[1]))
    return [items[i] for i in np.linspace(0, len(items) - 1, k).astype(int)]


def real_inputs(rows) -> dict:
    """EdgeSLM_AndroidGemma4Head `app/b1_prefill_bench.real_inputs` と同じ作り方（右詰め ID 0・因果マスク・表引き）"""
    from app.pipeline.gen_pc import Tables
    tb = Tables()
    out = {k: [] for k in ("inputs_embeds", "ple_tok", "cos_s", "sin_s", "cos_g", "sin_g", "attn_mask")}
    mask = causal_mask(T).numpy()
    for _, ids in rows:
        pad = list(ids) + [PAD_ID] * (T - len(ids))
        out["inputs_embeds"].append(np.concatenate([tb.embed(t) for t in pad], axis=1))
        out["ple_tok"].append(np.concatenate([tb.ple_tok(t) for t in pad], axis=1))
        for k in ("cos_s", "sin_s", "cos_g", "sin_g"):
            out[k].append(tb.rope[k][:T].reshape(1, T, 1, -1))
        out["attn_mask"].append(mask)
    res = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out.items()}
    res["n_real"] = np.array([len(ids) for _, ids in rows])
    return res


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def ortcheck(n: int, k: int = 8) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(OPS / full(n) / f"{full(n)}_final_qdq.onnx"), so, providers=["CPUExecutionProvider"])
    rows = pick_rows(k)
    ins = real_inputs(rows)
    names = [v.name for v in sess.get_inputs()]
    z = np.load(P1 / "all_layers.npz", mmap_mode="r")
    meta = json.loads((P1 / "all_layers.meta.json").read_text(encoding="utf-8"))
    idx = {i: j for j, i in enumerate(meta["ids"])}
    out = []
    for r in range(k):
        t1 = time.time()
        h = sess.run(["h"], {nm: ins[nm][r:r + 1] for nm in names})[0][0]
        m = int(ins["n_real"][r])
        j = idx[rows[r][0]]
        last, mean = h[m - 1], h[:m].mean(0)
        out.append({"id": rows[r][0], "n_tokens": m, "finite": bool(np.isfinite(h).all()),
                    "h_last_rel": rel(last, np.asarray(z["last"][n - 1, j])), "h_mean_rel": rel(mean, np.asarray(z["mean"][n - 1, j])),
                    "sec": round(time.time() - t1, 1)})
    res = {"rows": out, "h_last_rel_mean": float(np.mean([o["h_last_rel"] for o in out])),
           "h_mean_rel_mean": float(np.mean([o["h_mean_rel"] for o in out]))}
    print(f"  [L{n}] 繋いだ QDQ（PC の ORT）↔ P1 の HF（層 {n} 個・SRQ あり）: h（最後）相対L2 平均 {res['h_last_rel_mean'] * 100:.1f}%"
          f" / h（平均）{res['h_mean_rel_mean'] * 100:.1f}%（参考: 改善版 B の 35 層は最後 21%）・1 行 {out[0]['sec']} 秒", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"ortcheck_L{n}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if not all(o["finite"] for o in out):
        raise SystemExit("h に有限でない値がある。変換に進まない")
    return res


# ---------------------------------------------------------------- convert

def find_bash() -> str:
    for c in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if Path(c).exists():
            return c
    git = shutil.which("git")
    if git:
        c = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if c.exists():
            return str(c)
    raise SystemExit("Git Bash（bash.exe）が見つからない（System32 の bash.exe は WSL なので使わない）")


def convert(n: int) -> None:
    fd = OPS / full(n)
    run([find_bash(), "scripts/build_context.sh", str(fd / f"{full(n)}_final_qdq.onnx"), str(fd / "build"), f"p2L{n}"],
        f"p2_convert_L{n}.log", env={**os.environ, "QAIRT_INT4_PATCH": "1"})
    bins = sorted((fd / "build").glob("ctx_*/*.bin"))
    if not bins:
        raise SystemExit("context binary ができていない")
    from app.pipeline import wrap_epctx
    info = wrap_epctx.run(bins[0], fd / f"{full(n)}_final_qdq.onnx", fd / f"{full(n)}_epctx.onnx", full(n), embed=False)
    print(f"  [L{n}] EPContext: {info['epctx_path']}  context binary: {bins[0].name}（{bins[0].stat().st_size / 2**20:.0f}MiB）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="chunks,merge,ortcheck")
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    if set(ns) - set(PLAN):
        raise SystemExit(f"作れる層の数は {sorted(PLAN)}")
    steps = [x for x in a.steps.replace(" ", ",").split(",") if x]
    check_inputs()
    for st in steps:
        for n in ns:
            print(f"\n===== P2 {st}（層 {n} 個）=====", flush=True)
            {"chunks": lambda: chunk(n, a.force), "merge": lambda: merge(n), "ortcheck": lambda: ortcheck(n),
             "convert": lambda: convert(n)}[st]()
    print("おわり", flush=True)


if __name__ == "__main__":
    main()
