# -*- coding: utf-8 -*-
"""P2 手順 2: 層を減らしたグラフ（層 17・層 26）と 35 層（改善版 B）を、**同じ端末・同じ回**に流して比べる。

設計: DoC/設計書_P2_層を減らしたグラフを端末で.md の 4 節。

お試し用の別アプリ「Gemma4 層削減お試し」（`org.edgeslm.gemma4prune`、`android_app_prune/`。改善版 B のアプリの複製で、
applicationId と表示名だけ変えた）の Runner の `classify_prefill` を使う。**既存のアプリ（gemma4head・gemma4headfast・
gemma4headgen）とその置き場には触れない**。

|段|何をするか|
|---|---|
|`deploy`|お試しアプリの置き場へ、表（EdgeSLM_AndroidGemma4 の `work/tables`。読むだけ）・トークナイザ・3 つのグラフを送る（md5 を確かめる）。35 層は EdgeSLM_AndroidGemma4Head の改善版 B のもの（読むだけ）|
|`speed`|16 行を 3 つのグラフで流し、1 回の時間（NPU・準備・合計）を比べる（グラフごとにアプリを止めて開き直す）|
|`collect`|ルーター 714 行を 3 つのグラフで流し、`h`（最後・全位置の平均）を集める → `work/p2/L<n>/device/features`（ヘッドの学習用）|

    python -m app.p2_device deploy
    python -m app.p2_device speed
    python -m app.p2_device collect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from app.common import GEMMA_ROOT, HEAD_PRJ, PROJECT_ROOT

ADB = r"D:\Android\sdk\platform-tools\adb.exe"
PKG = "org.edgeslm.gemma4prune"
ACT = f"{PKG}/org.edgeslm.demo.RunnerActivity"
ACTION = "org.edgeslm.demo.RUN"
DEV = f"/sdcard/Android/data/{PKG}/files"
OTHERS = ["org.edgeslm.gemma4headgen", "org.edgeslm.gemma4headfast", "org.edgeslm.gemma4head", "org.edgeslm.gemma4"]
EP = {"provider": "QNN", "backend": "libQnnHtp.so",
      "options": {"offload_graph_io_quantization": "0", "qnn_context_priority": "high"},
      "session": {"intra_op_num_threads": 1, "log_severity_level": 3}}
TABLES = GEMMA_ROOT / "work" / "tables"                     # 読むだけ
TOKENIZER = GEMMA_ROOT / "work" / "tokenizer"               # 読むだけ
TOKENIZER_FILES = ["vocab.bin", "merges.bin", "specials.tsv", "tokenizer_meta.json", "golden.jsonl"]
G1 = HEAD_PRJ / "work" / "g1" / "srq" / "features"          # 読むだけ（714 行のトークン列・並び）
OPS = PROJECT_ROOT / "work" / "ops"
GRAPHS = {   # 層の数 → (EPContext, context binary を探す場所)
    17: (OPS / "p2_L17" / "p2_L17_epctx.onnx", OPS / "p2_L17"),
    26: (OPS / "p2_L26" / "p2_L26_epctx.onnx", OPS / "p2_L26"),
    35: (HEAD_PRJ / "work" / "ops" / "g4cls64_full" / "g4cls64_full_epctx.onnx", HEAD_PRJ / "work" / "ops" / "g4cls64_full"),
}
IO = PROJECT_ROOT / "work" / "p2" / "device_io"
RES = PROJECT_ROOT / "results" / "P2"
BATCH = 120


def adb(*a: str, timeout: int = 900, check: bool = True) -> str:
    r = subprocess.run([ADB, *a], capture_output=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"adb {' '.join(a)}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8", "replace")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dev_md5(rel: str) -> str:
    o = adb("shell", f"md5sum {DEV}/{rel} 2>/dev/null", check=False).split()
    return o[0] if o and len(o[0]) == 32 else ""


def push(src: Path, rel: str) -> None:
    want = md5(src)
    if dev_md5(rel) == want:
        print(f"  skip {rel}（端末に同じものがある）", flush=True)
        return
    adb("push", str(src), f"{DEV}/{rel}", timeout=1800)
    if dev_md5(rel) != want:
        raise SystemExit(f"md5 が合わない: {rel}")
    print(f"  push {rel}（{src.stat().st_size / 2**20:.0f}MiB）md5 一致", flush=True)


def ctx_of(epctx: Path, search: Path) -> Path:
    import onnx
    want = None
    for nd in onnx.load(str(epctx), load_external_data=False).graph.node:
        for at in nd.attribute:
            if at.name == "ep_cache_context" and at.s and len(at.s) < 4096:
                want = at.s.decode("utf-8", "ignore")
    cand = sorted(b for b in search.rglob("*.bin") if b.name == want) if want else []
    if not cand:
        raise SystemExit(f"{epctx.name} が参照する context binary が無い: {want}")
    return cand[0]


def model_dev(n: int) -> str:
    return f"models/ops/{GRAPHS[n][0].name}"


def ensure_app() -> None:
    if PKG not in adb("shell", f"pm list packages {PKG}"):
        raise SystemExit(f"端末に {PKG} が入っていない。先に .\\scripts\\p2_build_app.ps1")
    for p in OTHERS:                    # ほかの Gemma のアプリがモデルを開いたままだと RAM が足りない
        adb("shell", f"am force-stop {p}", check=False)


def alive() -> bool:
    return bool(adb("shell", f"pidof {PKG}", check=False).strip())


def run_job(job: dict, timeout_s: int = 900) -> dict:
    """job を送って Runner に渡し、結果を pull で取る（force-stop の後 8 秒待つ・消えたらやり直す）"""
    IO.mkdir(parents=True, exist_ok=True)
    jp = IO / f"{job['job_id']}.job.json"
    jp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    why = ""
    for attempt in range(3):
        adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
        adb("shell", f"am force-stop {PKG}")
        time.sleep(8.0)
        adb("push", str(jp), f"{DEV}/jobs/{jp.name}")
        adb("shell", f"rm -f {DEV}/out/last_result.json {DEV}/out/last_error.json {DEV}/out/last_describe.json", check=False)
        adb("shell", f"am start -n {ACT} -a {ACTION} --es job jobs/{jp.name}")
        t0 = time.time()
        why = f"{timeout_s} 秒たっても結果が返らない"
        while time.time() - t0 < timeout_s:
            time.sleep(3)
            ls = adb("shell", f"ls {DEV}/out/", check=False)
            if "last_result.json" in ls:
                local = IO / f"{job['job_id']}.result.json"
                adb("pull", f"{DEV}/out/last_result.json", str(local), check=False)
                try:
                    res = json.loads(local.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
                if res.get("job_id") == job["job_id"]:
                    return res
                continue
            if "last_error.json" in ls or "last_describe.json" in ls:
                why = adb("shell", f"cat {DEV}/out/last_*.json", check=False)[:600]
                break
            if time.time() - t0 > 20 and not alive():
                why = "プロセスが結果を出さずに消えた"
                break
        print(f"  ★{job['job_id']}（{attempt + 1} 回目）: {why}", flush=True)
    raise SystemExit(f"{job['job_id']}: 結果が返らない（{why}）")


def base_job(job_id: str, n: int) -> dict:
    t = json.loads((TABLES / "tables.json").read_text(encoding="utf-8"))
    e, p = t["embed"], t["ple"]
    return {"job_id": job_id, "task": "classify_prefill", "model": model_dev(n), "ep": EP, "mask_neg": -60.0, "pad_id": 0,
            "embed_table": f"models/ops/{e['file']}", "embed_dtype": "q2row",
            "embed_scales": f"models/ops/{e['scale_file']}", "embed_scale": e["embed_scale"],
            "ple_table": f"models/ops/{p['file']}", "ple_scales": f"models/ops/{p['scale_file']}", "ple_scale": p["embed_scale"],
            "rope_tables": {k: f"models/ops/{v}" for k, v in t["rope"]["files"].items()}}


def graphs(ns: list[int]) -> list[int]:
    ok = []
    for n in ns:
        if GRAPHS[n][0].exists():
            ok.append(n)
        else:
            print(f"  ★層 {n} 個のグラフが無い（{GRAPHS[n][0]}）。飛ばす", flush=True)
    return ok


# ---------------------------------------------------------------- deploy・speed・collect

def deploy(ns: list[int]) -> None:
    ensure_app()
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", f"am force-stop {PKG}")
    adb("shell", f"am start -n {ACT}")           # フォルダはアプリに作らせる（adb で作ると所有者が shell になり読めない）
    time.sleep(2.0)
    print(f"== {PKG} の置き場へ（PC の原本は読むだけ。既存のアプリには触れない）", flush=True)
    t = json.loads((TABLES / "tables.json").read_text(encoding="utf-8"))
    for f in [t["embed"]["file"], t["embed"]["scale_file"], t["ple"]["file"], t["ple"]["scale_file"], *t["rope"]["files"].values()]:
        push(TABLES / f, f"models/ops/{f}")
    for f in TOKENIZER_FILES:
        push(TOKENIZER / f, f"tokenizer/{f}")
    for n in graphs(ns):
        e, s = GRAPHS[n]
        b = ctx_of(e, s)
        push(b, f"models/ops/{b.name}")
        push(e, model_dev(n))


def speed(ns: list[int], rows: int = 16) -> dict:
    ensure_app()
    meta = json.loads((G1 / "features_F2.meta.json").read_text(encoding="utf-8"))
    toks = json.loads((G1 / "token_ids.json").read_text(encoding="utf-8"))
    ids = [meta["ids"][i] for i in np.linspace(0, len(meta["ids"]) - 1, rows).astype(int)]
    out = {}
    for n in graphs(ns):
        res = run_job({**base_job(f"p2_speed_L{n}", n), "prompts": [toks[i] for i in ids]})
        rs = res.get("results", [])
        if len(rs) != len(ids) or not all(r.get("ok") for r in rs):
            raise SystemExit(f"層 {n}: classify_prefill が通らない: {[r.get('error') for r in rs if not r.get('ok')][:3]} {res.get('error')}")
        d = res.get("describe", {})
        b = ctx_of(*GRAPHS[n])
        out[n] = {"run_ms_median": float(np.median([r["run_ms"] for r in rs])), "prep_ms_median": float(np.median([r["prep_ms"] for r in rs])),
                  "total_ms_median": float(np.median([r["total_ms"] for r in rs])), "context_mib": b.stat().st_size / 2**20,
                  "backend": d.get("backend"), "strict": d.get("strict_no_cpu_fallback")}
        o = out[n]
        print(f"  層 {n:>2} 個: NPU {o['run_ms_median']:.1f}ms・準備 {o['prep_ms_median']:.1f}ms・合計 {o['total_ms_median']:.1f}ms"
              f" / context {o['context_mib']:.0f}MiB / {o['backend']}・strict={o['strict']}", flush=True)
    if 35 in out:
        for n in out:
            if n != 35:
                print(f"  層 {n} 個 ÷ 35 層: NPU {out[n]['run_ms_median'] / out[35]['run_ms_median']:.2f} 倍・"
                      f"context {out[n]['context_mib'] / out[35]['context_mib']:.2f} 倍", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "speed.json").write_text(json.dumps({str(k): v for k, v in out.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def collect(ns: list[int]) -> dict:
    ensure_app()
    meta = json.loads((G1 / "features_F2.meta.json").read_text(encoding="utf-8"))
    ids = meta["ids"]
    toks = json.loads((G1 / "token_ids.json").read_text(encoding="utf-8"))
    summ = {}
    for n in graphs(ns):
        got = {}
        t_all = time.time()
        for k, s in enumerate(range(0, len(ids), BATCH)):
            part = ids[s:s + BATCH]
            f = IO / f"p2_collect_L{n}_{k:02d}.rows.json"
            if f.exists():
                for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
                    got[r["id"]] = r
                continue
            res = run_job({**base_job(f"p2_collect_L{n}_{k:02d}", n), "prompts": [toks[i] for i in part]}, 900)
            rs = res.get("results", [])
            bad = [(i, r.get("error")) for i, r in zip(part, rs) if not r.get("ok")]
            if len(rs) != len(part) or bad:
                raise SystemExit(f"層 {n}・{k}: 通らない行 {bad[:3]}（{len(rs)}/{len(part)}）")
            rows = [{"id": i, "last": r["h_last"], "mean": r["h_mean"], "run_ms": r["run_ms"]} for i, r in zip(part, rs)]
            f.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            for r in rows:
                got[r["id"]] = r
            print(f"  層 {n} 個 {k + 1}: {len(part)} 行・NPU 中央値 {np.median([r['run_ms'] for r in rows]):.1f}ms"
                  f"（通算 {time.time() - t_all:.0f} 秒）", flush=True)
        d = PROJECT_ROOT / "work" / "p2" / f"L{n:02d}" / "device" / "features"
        d.mkdir(parents=True, exist_ok=True)
        last = np.array([got[i]["last"] for i in ids], np.float32)
        mean = np.array([got[i]["mean"] for i in ids], np.float32)
        np.savez(d / "features_F2.npz", last__last=last, last__mean=mean)
        m = {"format": "F2", "ids": ids, "layers": ["last"], "poolings": ["last", "mean"],
             "meta": {"source": f"device (Xperia 5 III, 層 {n} 個のまとめ読み T=64, QNN HTP)", "n_layers_kept": n, "mean": "全位置の平均"}}
        (d / "features_F2.meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "token_ids.json").write_text(json.dumps(toks), encoding="utf-8")
        summ[n] = {"run_ms_median": float(np.median([got[i]["run_ms"] for i in ids])), "elapsed_s": round(time.time() - t_all, 1)}
        # 参考: 同じ層の数の PC の HF（P1）との差
        z = np.load(PROJECT_ROOT / "work" / "p1" / "all_layers.npz", mmap_mode="r")
        ref_l, ref_m = np.asarray(z["last"][n - 1]), np.asarray(z["mean"][n - 1])
        rl = np.linalg.norm(last - ref_l, axis=1) / (np.linalg.norm(ref_l, axis=1) + 1e-12)
        rm = np.linalg.norm(mean - ref_m, axis=1) / (np.linalg.norm(ref_m, axis=1) + 1e-12)
        summ[n].update({"h_last_rel_mean_vs_pc": float(rl.mean()), "h_mean_rel_mean_vs_pc": float(rm.mean())})
        print(f"  層 {n} 個 714 行: NPU 中央値 {summ[n]['run_ms_median']:.1f}ms / PC の HF（同じ層の数）との差 最後 {rl.mean() * 100:.1f}%・"
              f"平均 {rm.mean() * 100:.1f}% → {d}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "collect.json").write_text(json.dumps({str(k): v for k, v in summ.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["deploy", "speed", "collect"])
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    print(f"\n===== P2 {a.step}（層 {ns}）=====", flush=True)
    {"deploy": deploy, "speed": speed, "collect": collect}[a.step](ns)


if __name__ == "__main__":
    main()
