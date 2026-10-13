# -*- coding: utf-8 -*-
"""P24: 判定デモアプリ（Gemma4 判定デモ・org.edgeslm.gemma4router）に載せる資材を作る。

設計: DoC/設計書_P24_判定デモアプリ.md の 5 節。3 つの段に分かれる（使う Python が違うため）。

|段|何を作るか|どこで動くか|
|---|---|---|
|`heads`|層 17・26 のヘッド（`head.json`・`head.bin`・`head.tsv`）・画面の設定（`router.json`）・端末での確かめ用の行（`eval_v6.jsonl`）・BERT の辞書（`sys_lex.bin` ほか）・Java の確かめ用の `h`|numpy だけ（学習なし・端末なし）|
|`truth`|PC の `BertJapaneseTokenizer` が出す ID 列（`work/p24/tok_truth.tsv`）。端末用の入力作り（Java）と突き合わせる正解|EdgeSLM_MultiHead_Multi の venv（fugashi がある）|
|`bert`|BERT（案 F・seed 0）を ONNX にする（`model.onnx`）。torch・保存済みの確率との差を確かめる|EdgeSLM_Android の venv（onnx・onnxruntime がある）|

作る場所は `work/p24/device/router/`（端末の `files/router/` と同じ並び）。ほかのプロジェクトは読むだけ。

    python -m app.p24_assets heads
    python -m app.p24_assets truth
    python -m app.p24_assets bert
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT
from .g3_headlib import LABELS_JA, numpy_head
from .p6_seeds import head_probs, load_pt
from .p7_block import HEADS
from .p7_data import read_jsonl
from .p8_gate import grade
from .p18_analyze import hook1584
from .p18_retrain import SRC1584
from .p21_data import CH6
from .p21_verdict import WORK as P21, load_eval6

WORK = PROJECT_ROOT / "work" / "p24"
FIX = WORK / "fixtures"
OUT = WORK / "device" / "router"
RES = PROJECT_ROOT / "results" / "P24"
BERT_DIR = PROJECT_ROOT / "work" / "p20" / "bert_F" / "bert_s0"
DICDIR = MULTI_ROOT / ".venv" / "Lib" / "site-packages" / "unidic_lite" / "dicdir"      # 読むだけ
TRUTH = WORK / "tok_truth.tsv"
LAYERS = (17, 26)
HEAD_SEED = 7
MAX_LENGTH = 64
INFO = {17: {"pct": "49%", "ctx": 473, "label": "前半 17 層（軽量）"}, 26: {"pct": "74%", "ctx": 873, "label": "前半 26 層"}}
TOOL_JA = {k[5:]: v for k, v in LABELS_JA.items() if k.startswith("TOOL_")}
# 入力作りの確かめに足す、形の違う文（全角・半角カナ・記号・絵文字・長い文・空白）
EXTRA = ["ＡＢＣ１２３　テスト　です", "ｶﾞｿﾘﾝｽﾀﾝﾄﾞ探して😀", "25.5℃に㈱", "  先頭と末尾に空白  ", "えっと、あのー、その…やっぱりいいや", "Blue in Greenを流して", "すもももももももものうち",
         "外国人参政権", "Wi-Fiのパスワードは？", "ｴｱｺﾝを２５．５度に!!", "~~~♪♪♪", "あ", " ", "1,234.5km走った", "東京都千代田区丸の内1-9-1まで", "URLはhttps://example.com/a?b=c です",
         "ヴァイオリンのコンチェルトをかけて", "髙島屋と﨑陽軒", "ｱｲｳｴｵｶﾞｷﾞｸﾞｹﾞｺﾞ", "ーーーー", "…………", "(笑)ww", "エアコン" * 30, "あいうえお" * 40, "abc def　ghi jkl"]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_flow(n: int) -> tuple[list[str], dict, list[dict]]:
    """`p24_fixtures` が作った flow_L<n>.tsv（評価セット v6 の行。並びは評価セットの読み込みと同じ）"""
    lines = (FIX / f"flow_L{n}.tsv").read_text(encoding="utf-8").split("\n")
    classes = lines[0].split("\t")[1:]
    th = {k: float(v) for k, v in (x.split("=") for x in lines[1].split("\t")[1:])}
    rows = []
    for line in lines[2:]:
        if not line:
            continue
        c = line.split("\t")
        rows.append({"utterance": c[0], "probs": [float(x) for x in c[1].split(" ")], "reject": float(c[2]), "bert": float(c[3]),
                     "outcome": c[4], "stage": None if c[5] == "-" else c[5], "tool": None if c[6] == "-" else c[6], "args": None if c[7] == "-" else c[7], "rule": None if c[8] == "-" else c[8]})
    return classes, th, rows


# ---------------------------------------------------------------- heads

def export_head(n: int) -> dict:
    """ヘッド（1,584 行・mlp・seed 7）を、端末の形に書き出す。棄却の閾値は validation の 2%（いまの構成）"""
    hd = HEADS[SRC1584](n)
    name = f"heads_mlp_seed{HEAD_SEED}.pt"
    blob = load_pt(hd / name)
    rep = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}[name]
    classes = list(blob["metadata"]["action_classes"])
    th = hook1584(0.02)(n, SRC1584, [(HEAD_SEED, classes, {k: float(v) for k, v in rep["thresholds"].items()}, None)])[0][2]
    state = {k: np.ascontiguousarray(np.asarray(v), dtype="<f4") for k, v in blob["state"].items() if k != "temperature"}
    tensors, chunks, off = {}, [], 0
    for key in sorted(state):
        tensors[key] = {"offset": off, "shape": list(state[key].shape)}
        chunks.append(state[key].ravel())
        off += state[key].size
    flat = np.concatenate(chunks).astype("<f4")
    tools = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    meta = {"hidden": int(state["standardizer.mean"].shape[0]), "reject_hidden": int(state["reject_standardizer.mean"].shape[0]),
            "width": int(state["heads.action.0.weight"].shape[0]), "kind": blob["config"]["kind"], "temperature": float(rep["temperature"]),
            "action_classes": classes, "argmode_classes": list(blob["metadata"].get("argmode_classes", [])), "thresholds": {k: float(v) for k, v in th.items()},
            "tool_effect": tools, "argmode_tools": ["control_hvac", "control_media"], "labels_ja": LABELS_JA, "tensors": tensors, "total_floats": int(flat.size),
            "source": {"checkpoint": str(hd / name), "variant": f"router_L{n}", "rows": "1,584 行（P18）", "reject_threshold": "validation の 2%（P19）"}}

    # ---- 確かめ: 端末と同じ式（numpy_head）↔ 評価で使った式（head_probs）↔ p24_fixtures の表
    rows, last, mean = load_eval6(n)
    mine = numpy_head(meta, flat, last, mean)
    ref = head_probs(blob, float(rep["temperature"]), last, mean)
    fcls, fth, frows = read_flow(n)
    if fcls != classes or len(frows) != len(rows) or any(a["utterance"] != b["utterance"] for a, b in zip(rows, frows)):
        raise SystemExit(f"層 {n}: flow_L{n}.tsv の並びが評価セットと違う（先に python -m app.p24_fixtures）")
    d1 = float(np.abs(mine["action"] - ref["action"]).max())
    d2 = float(np.abs(mine["reject"] - ref["reject"]).max())
    d3 = float(np.abs(mine["action"] - np.array([r["probs"] for r in frows])).max())
    d4 = float(np.abs(mine["reject"] - np.array([r["reject"] for r in frows])).max())
    dth = max(abs(fth[k] - meta["thresholds"][k]) for k in ("CHAT", "TOOL_read", "TOOL_write", "REJECT"))
    print(f"層 {n}: 温度 {meta['temperature']:.2f}・閾値 {meta['thresholds']} / 式の差 確率 {d1:.1e}・棄却 {d2:.1e} / 表との差 確率 {d3:.1e}・棄却 {d4:.1e}・閾値 {dth:.1e}")
    if max(d1, d2, d3, d4, dth) > 1e-9:
        raise SystemExit(f"層 {n}: 書き出したヘッドが、評価で使ったものと合わない")
    meta["self_check"] = {"rows": len(rows), "action_prob_absdiff_max": d3, "reject_absdiff_max": d4}

    out = OUT / f"L{n}"
    out.mkdir(parents=True, exist_ok=True)
    flat.tofile(out / "head.bin")
    (out / "head.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # head.tsv: head.json と同じ中身を、JSON の読み手なしで読める形にしたもの（Java の確かめを PC でも回せるように）
    tsv = [f"hidden\t{meta['hidden']}", f"reject_hidden\t{meta['reject_hidden']}", f"temperature\t{meta['temperature']!r}", "classes\t" + "\t".join(classes)]
    tsv += [f"threshold\t{k}\t{meta['thresholds'][k]!r}" for k in ("CHAT", "TOOL_read", "TOOL_write", "REJECT")]
    tsv += [f"tensor\t{k}\t{v['offset']}\t" + "\t".join(str(x) for x in v["shape"]) for k, v in tensors.items()]
    tsv.append(f"total_floats\t{flat.size}")
    (out / "head.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8", newline="\n")
    # Java の確かめ用: 評価セット v6 の端末の h（最後・平均。float32）
    np.hstack([last, mean]).astype("<f4").tofile(FIX / f"h_L{n}.bin")
    return meta


def dictionary() -> dict:
    """BERT の入力作りに使う辞書（unidic-lite。読むだけ）と語彙を、端末の並びに置く。`sys.dic` は、見出しと単語の表（先頭）だけを切り出す"""
    import struct
    out = OUT / "bert"
    out.mkdir(parents=True, exist_ok=True)
    with open(DICDIR / "sys.dic", "rb") as f:
        head = struct.unpack("<10I", f.read(40))
        f.seek(0)
        (out / "sys_lex.bin").write_bytes(f.read(72 + head[6] + head[7]))       # ヘッダ 40 ＋ 文字コード 32 ＋ 見出しの表 ＋ 単語の表
    for name in ("matrix.bin", "char.bin", "unk.dic"):
        if not (out / name).exists() or (out / name).stat().st_size != (DICDIR / name).stat().st_size:
            shutil.copyfile(DICDIR / name, out / name)
    shutil.copyfile(BERT_DIR / "vocab.txt", out / "vocab.txt")
    info = {p.name: {"bytes": p.stat().st_size, "md5": md5(p)} for p in sorted(out.iterdir()) if p.name != "model.onnx"}
    print("BERT の辞書: " + "・".join(f"{k} {v['bytes'] / 2**20:.1f}MiB" for k, v in info.items()))
    return info


def pick_samples(rows: list[dict], flows: dict) -> list[dict]:
    """画面の例文。層 17・26 のどちらでも PC の判定が正解の行から、結果の種類ごとに短い文を選ぶ"""
    want = [("実行", "control_hvac", 3), ("実行", "control_media", 2), ("実行", "set_navigation", 2), ("実行", "get_vehicle_status", 1), ("実行", "search_poi", 1),
            ("雑談", None, 1), ("聞き返し", "規則", 1), ("聞き返し", "引数が足りない", 1), ("聞き返し", "棄却ヘッド", 1), ("拒否", "BERT が断る", 2)]
    write = {"control_hvac", "control_media", "set_navigation"}
    out, used = [], set()
    for outcome, key, count in want:
        cand = []
        for i, r in enumerate(rows):
            f = [flows[n][i] for n in LAYERS]
            if any(x["outcome"] != outcome for x in f) or r["utterance"] in used or not 4 <= len(r["utterance"]) <= 16:
                continue
            if any(grade(r, x["outcome"], x["tool"] if x["outcome"] != "雑談" else None, write, True) != "正解" for x in f):
                continue
            if outcome == "実行" and (any(x["tool"] != key for x in f) or f[0]["args"] != f[1]["args"]):
                continue
            if outcome in ("聞き返し", "拒否") and any(x["stage"] != key for x in f):
                continue
            cand.append((len(r["utterance"]), i))
        seen_args, taken = set(), 0
        for _, i in sorted(cand):
            a = flows[LAYERS[0]][i]["args"]
            if outcome == "実行" and a in seen_args:
                continue                       # 同じ引数の例は 1 つにする
            seen_args.add(a)
            used.add(rows[i]["utterance"])
            out.append({"id": rows[i]["id"], "utterance": rows[i]["utterance"], "motion": rows[i]["state"]["motion"]})
            taken += 1
            if taken >= count:
                break
    return out


def heads() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    metas = {n: export_head(n) for n in LAYERS}
    dic = dictionary()
    rows = read_jsonl(CH6 / "utterances.jsonl")
    flows = {n: read_flow(n)[2] for n in LAYERS}
    bert_hi = read_flow(LAYERS[0])[1]["BERT_HI"]
    toks = json.loads((P21 / "token_ids_eval.json").read_text(encoding="utf-8"))
    p3 = json.loads((PROJECT_ROOT / "work" / "p3" / "L17" / "head" / "samples.json").read_text(encoding="utf-8"))      # チャットの前後・表の場所（P3 で PC の HF から作ったもの）
    job = {k: v for k, v in p3["job"].items() if k != "model"}
    with open(OUT / "eval_v6.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            pc = {str(n): {k: flows[n][i][k] for k in ("outcome", "stage", "tool", "args", "rule", "bert", "reject")} | {"conf": max(flows[n][i]["probs"])} for n in LAYERS}
            f.write(json.dumps({"id": r["id"], "utterance": r["utterance"], "motion": r["state"]["motion"], "gemma_ids": toks[r["id"]], "pc": pc}, ensure_ascii=False) + "\n")
    samples = pick_samples(rows, flows)
    byid = {r["id"]: i for i, r in enumerate(rows)}
    for s in samples:
        i = byid[s["id"]]
        s["pc"] = {str(n): {k: flows[n][i][k] for k in ("outcome", "stage", "tool", "args")} for n in LAYERS}
    cfg = {"title": "車載ルーター 判定デモ（スマホの中だけで判定）",
           "subtitle": "発話を、層を減らした Gemma 4 E2B（NPU）＋分類ヘッド・日本語 BERT・規則・引数の取り出し・検査に通し、実行／聞き返す／断る／雑談 を決めます。"
                       "操作は模擬です（実際の車は動かしません）。通信は使いません。",
           "note": "評価の流れ（P21・P23 の構成）と同じ順番・同じ部品です。ヘッドは 1,584 行で学習したもの（seed 7）、BERT は案 F（seed 0）。聞き返した後の 2 回目のやり取りは入っていません。",
           "chat_prefix": p3["chat_prefix"], "chat_suffix": p3["chat_suffix"], "job": job, "default_layer": 17,
           "layers": [{"n": n, "label": INFO[n]["label"], "model": f"models/ops/p2_L{n}_epctx.onnx", "head": f"router/L{n}", "context_mib": INFO[n]["ctx"],
                       "detail": f"Gemma 4 E2B の 35 層のうち前半 {n} 層（{INFO[n]['pct']}）・グラフ {INFO[n]['ctx']}MiB"} for n in LAYERS],
           "bert": {"dir": "router/bert", "model": "router/bert/model.onnx", "max_length": MAX_LENGTH, "hi": bert_hi, "lo": 0.5, "threads": 4,
                    "name": "cl-tohoku/bert-base-japanese-v3 を学習し直したもの（案 F・seed 0）"},
           "labels_ja": LABELS_JA, "tool_ja": TOOL_JA, "samples": samples, "eval": "router/eval_v6.jsonl"}
    (OUT / "router.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = {str(p.relative_to(OUT)).replace("\\", "/"): {"bytes": p.stat().st_size, "md5": md5(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "model.onnx")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"例文 {len(samples)} 文: " + "／".join(s["utterance"] for s in samples))
    print(f"確かめ用の行 {len(rows)}（評価セット v6）/ BERT が断る閾値 {bert_hi:.4f}")
    print(f"保存しました: {OUT}（{len(manifest)} ファイル）")
    (RES / "assets.json").write_text(json.dumps({"heads": {str(n): {k: metas[n][k] for k in ("temperature", "thresholds", "self_check")} for n in LAYERS},
                                                 "dictionary": dic, "samples": samples, "bert_hi": bert_hi}, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- truth（PC の BertJapaneseTokenizer）

def truth() -> None:
    from transformers import AutoTokenizer
    from .p24_fixtures import all_utterances
    tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
    us = all_utterances() + [u for u in EXTRA]
    words_of = getattr(getattr(tok, "word_tokenizer", None), "tokenize", None)
    WORK.mkdir(parents=True, exist_ok=True)
    n_long = 0
    with open(TRUTH, "w", encoding="utf-8", newline="\n") as f:
        for u in us:
            ids = tok(u, truncation=True, max_length=MAX_LENGTH)["input_ids"]
            n_long += len(ids) == MAX_LENGTH
            cols = [u, " ".join(str(i) for i in ids)]
            if words_of is not None:
                cols.append(" ".join(words_of(u)))
            f.write("\t".join(cols) + "\n")
    print(f"PC の BertJapaneseTokenizer の ID 列: {len(us)} 行（うち {MAX_LENGTH} で切った行 {n_long}）→ {TRUTH}" + ("" if words_of else "（単語の列は取れなかった。ID 列だけ）"))


# ---------------------------------------------------------------- bert（ONNX にする）

def bert() -> None:
    import onnx
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification
    from .p20_data import CH5
    if not TRUTH.exists():
        raise SystemExit(f"{TRUTH} が無い（先に truth の段）")
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR)).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    wrap = Wrap(model).eval()
    ids = torch.tensor([[2, 14356, 13160, 500, 12654, 2023, 461, 441, 456, 3]], dtype=torch.long)
    args = (ids, torch.ones_like(ids), torch.zeros_like(ids))
    names = ["input_ids", "attention_mask", "token_type_ids"]
    out = OUT / "bert" / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    axes = {k: {0: "batch", 1: "seq"} for k in names} | {"logits": {0: "batch"}}
    how = "TorchScript の書き出し（opset 17）"
    try:
        torch.onnx.export(wrap, args, str(out), input_names=names, output_names=["logits"], dynamic_axes=axes, opset_version=17, do_constant_folding=True, dynamo=False)
    except Exception as e:      # 古い書き出しが使えない版では、新しい書き出しに回す
        print(f"古い書き出しが通らない（{type(e).__name__}: {str(e)[:200]}）。新しい書き出しでやり直す", flush=True)
        how = "dynamo の書き出し（opset 18）"
        torch.onnx.export(wrap, args, str(out), input_names=names, output_names=["logits"], dynamic_axes=axes, opset_version=18, dynamo=True, external_data=False)
    m = onnx.load(str(out), load_external_data=False)
    ir0 = m.ir_version
    if m.ir_version > 10:       # 端末の ONNX Runtime（1.27）が読める版に下げる（中身は変えない）
        m = onnx.load(str(out))
        m.ir_version = 10
        onnx.save(m, str(out))
    opset = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    extra = [p.name for p in out.parent.iterdir() if p.name.startswith("model.onnx") and p.name != "model.onnx"]
    if extra:
        raise SystemExit(f"重みが別のファイルに分かれた（{extra}）。1 つのファイルにまとめる必要がある")
    print(f"書き出し: {how}・IR {ir0}{'→10' if ir0 > 10 else ''}・opset {opset}・{out.stat().st_size / 2**20:.0f}MiB", flush=True)

    # ---- 確かめ: 評価セット v6・v5 の行で、ONNX ↔ torch（同じ入力）↔ 保存済みの確率（P21。GPU・まとめて）
    table = {}
    for line in TRUTH.read_text(encoding="utf-8").split("\n"):
        c = line.split("\t")
        if len(c) >= 2 and c[1]:
            table[c[0]] = [int(x) for x in c[1].split(" ")]
    raw = json.loads((P21 / "text_scores_F.json").read_text(encoding="utf-8"))
    saved = raw["models"]["bert_s0"]["pred"]
    rows = read_jsonl(CH6 / "utterances.jsonl") + read_jsonl(CH5 / "utterances.jsonl")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    have = {i.name for i in sess.get_inputs()}

    def soft(z):
        z = np.asarray(z, np.float64)
        e = np.exp(z - z.max())
        return float(e[1] / e.sum())

    p_onnx, p_torch, p_saved = [], [], []
    for r in rows:
        x = np.array([table[r["utterance"]]], np.int64)
        feed = {k: v for k, v in (("input_ids", x), ("attention_mask", np.ones_like(x)), ("token_type_ids", np.zeros_like(x))) if k in have}
        p_onnx.append(soft(sess.run(None, feed)[0][0]))
        with torch.no_grad():
            t = torch.from_numpy(x)
            p_torch.append(soft(wrap(t, torch.ones_like(t), torch.zeros_like(t))[0].numpy()))
        p_saved.append(float(saved[r["id"]]))
    p_onnx, p_torch, p_saved = np.array(p_onnx), np.array(p_torch), np.array(p_saved)
    hi = json.loads((OUT / "router.json").read_text(encoding="utf-8"))["bert"]["hi"]
    d1, d2 = float(np.abs(p_onnx - p_torch).max()), float(np.abs(p_onnx - p_saved).max())
    flip = int(np.sum((p_onnx >= hi) != (p_saved >= hi)) + np.sum((p_onnx >= 0.5) != (p_saved >= 0.5)))
    print(f"確かめ（{len(rows)} 行）: ONNX ↔ torch の確率の差の最大 {d1:.2e} / ONNX ↔ 保存済みの確率（P21）の差の最大 {d2:.2e} / 閾値（断る {hi:.4f}・フラグ 0.5）をまたいだ行 {flip}")
    info = {"how": how, "ir_version": int(min(ir0, 10)), "opset": opset, "bytes": out.stat().st_size, "md5": md5(out), "rows": len(rows), "inputs": sorted(have),
            "onnx_vs_torch_absdiff_max": d1, "onnx_vs_saved_absdiff_max": d2, "threshold_flips": flip}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "bert_onnx.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "bert_onnx.md").write_text(
        f"BERT（案 F・seed 0。`work/p20/bert_F/bert_s0`）を ONNX にしたもの（`work/p24/device/router/bert/model.onnx`）の確かめ。行は評価セット v6・v5 の {len(rows)} 行\n\n"
        "|項目|値|\n|---|---|\n"
        f"|書き出し|{how}・opset {opset.get('ai.onnx')}|\n|大きさ|{out.stat().st_size / 2**20:.0f}MiB|\n"
        f"|ONNX ↔ torch（同じ入力・CPU）の確率の差の最大|{d1:.2e}|\n|ONNX ↔ 保存済みの確率（P21。GPU・まとめて計算）の差の最大|{d2:.2e}|\n"
        f"|閾値（断る {hi:.4f}・フラグ 0.5）をまたいだ行|{flip}|\n", encoding="utf-8")
    if d1 > 1e-3:
        raise SystemExit("ONNX が torch と合わない。止める")
    print(f"保存しました: {out}・{RES / 'bert_onnx.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["heads", "truth", "bert"])
    a = ap.parse_args()
    print(f"\n===== P24 資材（{a.step}）=====", flush=True)
    {"heads": heads, "truth": truth, "bert": bert}[a.step]()


if __name__ == "__main__":
    main()
