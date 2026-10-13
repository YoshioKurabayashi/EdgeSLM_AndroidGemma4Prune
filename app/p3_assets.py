# -*- coding: utf-8 -*-
"""P3 手順 1: お試しアプリの分類画面に載せる資材（分類ヘッド・例文・job の設定・見出し）を PC で作る。CPU で数十秒。

設計: DoC/設計書_P3_層17の分類デモ.md の 3 節。

EdgeSLM_AndroidGemma4Head の `app/g3_assets.py`（md5 63650270）と同じ作り（関数を写した。元のファイルは変更しない・書き込み先は本プロジェクト）。違い:
- ヘッドは P2 で端末の `h` から学習したもの（`work/p2/L<n>/device/heads/heads_mlp_seed7.pt`）
- `samples.json` の `job` に、使うグラフ（`models/ops/p2_L<n>_epctx.onnx`）を書く（画面は job.model でグラフを切り替える）
- 画面の見出し（`title`・`subtitle`・`model_label`）を書く
- 比べる相手は「一括収集での判定（P2 で端末の 714 行を流したとき・同じヘッド）」

作るもの（`work/p3/L<n>/head/`）: `head.json`・`head.bin`・`samples.json`

自己検査（外れたら止める）: Kotlin と同じ式 ↔ torch の確率の差 < 1e-4・判定の一致 ≥ 99.5% / 例文のトークン列が G1 の符号化と一致

    python -m app.p3_assets --layers 17        # EdgeSLM_MultiHead_Multi の venv
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from .common import GEMMA_ROOT, HF_DIR, PROJECT_ROOT, ROUTER_DATA, SEQUENCE_CHECKPOINT, use_multi_library
from .g3_headlib import decisions, export_head, final_of, numpy_head

TABLES = GEMMA_ROOT / "work" / "tables"
FULL_MIB = 1273
INFO = {   # 層の数 → 見出し
    17: {"pct": "49%", "ctx": 473},
    26: {"pct": "74%", "ctx": 873},
    35: {"pct": "100%", "ctx": 1273},
}


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_job() -> dict:
    t = json.loads((TABLES / "tables.json").read_text(encoding="utf-8"))
    e, p = t["embed"], t["ple"]
    return {"embed_table": f"models/ops/{e['file']}", "embed_dtype": "q2row",
            "embed_scales": f"models/ops/{e['scale_file']}", "embed_scale": e["embed_scale"],
            "ple_table": f"models/ops/{p['file']}", "ple_scales": f"models/ops/{p['scale_file']}",
            "ple_scale": p["embed_scale"],
            "rope_tables": {k: f"models/ops/{v}" for k, v in t["rope"]["files"].items()}}


def pick_samples(rows: list[dict], dec: dict, ids_map: dict, per_class: int = 2) -> list[dict]:
    """行き先ごとに test → validation の順で、判定が正解した行を選ぶ（g3_assets.pick_samples と同じ）"""
    out, seen = [], {}
    order = {"test": 0, "validation": 1, "train": 2}
    for r in sorted(rows, key=lambda r: (order.get(r["split"], 3), r["id"])):
        want = r["action"]
        exp = "REJECT" if want == "UNKNOWN" else want
        d = dec.get(r["id"])
        if d is None or d["final"] != exp or r["history"] or seen.get(want, 0) >= per_class:
            continue
        seen[want] = seen.get(want, 0) + 1
        out.append({"id": r["id"], "utterance": r["utterance"], "motion": r["state"]["motion"],
                    "expected": want, "pc_final": d["final"], "pc_top1": d["top1"], "pc_conf": round(d["conf"], 4),
                    "pc_reject": round(d["reject"], 4), "ids": ids_map[r["id"]]})
    return out


def build(n: int) -> None:
    import torch
    from transformers import AutoTokenizer
    from edge_router.data.build import render
    from edge_router.dispatch import policy
    from edge_router.heads.model import MultiHead
    from edge_router.io_utils import read_jsonl

    base = PROJECT_ROOT / "work" / "p2" / f"L{n:02d}" / "device"
    head_dir = base / "heads"
    if not (head_dir / SEQUENCE_CHECKPOINT).exists():
        raise SystemExit(f"{head_dir} にヘッドが無い。先に scripts/run_p2.ps1 -Stage heads")
    out = PROJECT_ROOT / "work" / "p3" / f"L{n:02d}" / "head"
    out.mkdir(parents=True, exist_ok=True)

    head, hmeta = MultiHead.load(head_dir / SEQUENCE_CHECKPOINT, map_location="cpu")
    report = next(x for x in json.loads((head_dir / "evaluate_report.json").read_text(encoding="utf-8"))["results"]
                  if x["checkpoint"] == SEQUENCE_CHECKPOINT)
    head.temperature.fill_(float(report["temperature"]))
    note = f"層 {n} 個（層 0〜{n - 1}）のグラフの端末の h で学習・全位置の平均（P2）"
    meta, flat = export_head(head, hmeta, report, policy.load_tools(),
                             {"checkpoint": str(head_dir / SEQUENCE_CHECKPOINT), "variant": f"prune_L{n}", "mean": note,
                              "mean_skip_ids": []})
    print(f"ヘッド: {head_dir / SEQUENCE_CHECKPOINT}（{note}・温度 {float(report['temperature']):.2f}・閾値 {report['thresholds']}）")
    flat.tofile(out / "head.bin")
    flat2 = np.fromfile(out / "head.bin", dtype="<f4")

    feats = np.load(base / "features" / "features_F2.npz")
    fmeta = json.loads((base / "features" / "features_F2.meta.json").read_text(encoding="utf-8"))
    ids = fmeta["ids"]
    last, mean = feats["last__last"].astype(np.float64), feats["last__mean"].astype(np.float64)
    mine = numpy_head(meta, flat2, last, mean)
    with torch.no_grad():
        tp = head.probabilities(torch.from_numpy(last.astype(np.float32)),
                                torch.from_numpy(np.hstack([last, mean]).astype(np.float32)))
    dp = float(np.abs(mine["action"] - tp["action"].numpy()).max())
    dr = float(np.abs(mine["reject"] - tp["reject"][:, 1].numpy()).max())
    finals = [final_of(meta, mine["action"][k], float(mine["reject"][k])) for k in range(len(ids))]
    t_finals = [final_of(meta, tp["action"][k].numpy().astype(np.float64), float(tp["reject"][k, 1])) for k in range(len(ids))]
    agree = float(np.mean([x == y for x, y in zip(finals, t_finals)]))
    meta["self_check"] = {"rows": len(ids), "action_prob_absdiff_max": dp, "reject_absdiff_max": dr, "final_agree": agree}
    print(f"ヘッドの自己検査（Kotlin と同じ式 ↔ torch、{len(ids)} 行）: 確率の差の最大 {dp:.2e} / 棄却 {dr:.2e} / 判定の一致 {agree * 100:.1f}%")
    if dp > 1e-4 or dr > 1e-4 or agree < 0.995:
        raise SystemExit("ヘッドの書き出しが torch と一致しない。止める")
    (out / "head.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    dec = decisions(meta, flat2, last, mean)
    dec_by_id = {i: {k: dec[k][m] for k in dec} for m, i in enumerate(ids)}
    (out.parent / "device_decisions.json").write_text(json.dumps(dec_by_id, ensure_ascii=False), encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    marker = "@@UTTERANCE@@"
    tmpl = tok.apply_chat_template([{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True)
    prefix, suffix = tmpl.split(marker)
    rows = read_jsonl(ROUTER_DATA)
    ids_map = json.loads((base / "features" / "token_ids.json").read_text(encoding="utf-8"))
    samples = pick_samples(rows, dec_by_id, ids_map)
    byid = {r["id"]: r for r in rows}
    for s in samples:
        enc = tok(prefix + render(byid[s["id"]], "F2") + suffix, add_special_tokens=False)["input_ids"]
        if enc != s["ids"]:
            raise SystemExit(f"{s['id']}: チャットテンプレートの前後で作った列が G1 の符号化と違う")
    info = INFO.get(n, {"pct": f"{n}/35", "ctx": 0})
    # 35 層は改善版 B のグラフ（EdgeSLM_AndroidGemma4Head の g4cls64_full。P2 でお試しアプリへ送ったもの）
    job = {**table_job(), "model": "models/ops/g4cls64_full_epctx.onnx" if n == 35 else f"models/ops/p2_L{n}_epctx.onnx"}
    ui = {"title": f"Gemma 4 E2B ヘッド分類（前半 {n} 層・スマホの NPU）",
          "subtitle": f"Gemma 4 E2B の 35 層のうち前半 {n} 層（{info['pct']}）だけを残したモデルが、発話の意味を表す h を NPU で作り、"
                      f"小さな分類ヘッドが行き先を決めます。グラフ {info['ctx']}MiB（35 層は {FULL_MIB}MiB）。通信は使いません。",
          "model_label": f"Gemma 4 E2B（前半 {n} 層・まとめ読み）",
          "ref_label": "一括収集での判定（端末で 714 行を流したとき・同じヘッド）", "ref_short": "一括収集での判定",
          "note": f"ヘッドは、層 {n} 個のグラフの端末の h で学習しています。比べる相手も、端末で一括収集したときの判定です。"}
    (out / "samples.json").write_text(json.dumps(
        {"chat_prefix": prefix, "chat_suffix": suffix, "variant": f"prune_L{n}", "job": job, **ui, "samples": samples},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"例文 {len(samples)} 文（" + "、".join(sorted({s['expected'] for s in samples})) + f"）/ グラフ {job['model']}")
    manifest = {p.name: {"bytes": p.stat().st_size, "md5": md5(p)} for p in sorted(out.iterdir()) if p.is_file()}
    (out.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存しました: {out}（{len(manifest)} ファイル）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17")
    a = ap.parse_args()
    use_multi_library()
    for n in [int(x) for x in a.layers.replace(" ", ",").split(",") if x]:
        print(f"\n===== P3 assets（層 {n} 個）=====", flush=True)
        build(n)


if __name__ == "__main__":
    main()
