# -*- coding: utf-8 -*-
"""P18 手順 2: ヘッドと BERT を学習し直し、新しい評価セット（challenge_v4・600 行）で合否を測る。

設計: DoC/設計書_P18_BERTとヘッドを学習し直す.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

|構成|ヘッド|BERT|
|---|---|---|
|いまの構成（P17 で採った構成）|933 行（P5）|案 E（P13）|
|＋BERT を学習し直す|933 行|案 F（案 E ＋ train_add_v3 の新しい 285 行）|
|＋ヘッドを学習し直す|1,584 行（P18）|案 E|
|**＋両方（本命）**|1,584 行|案 F|

どの構成も、引数は `p17_arguments`・規則は `p17_rules`・棄却の閾値は validation の 2%（P10）・BERT の断る閾値は学習側の 99.5% 点・
フラグはすべての機能（P12）・数え方は P15。**本命と合否の基準は、測る前に決めた**（設計書 6 節）。

|段|何をするか|venv|
|---|---|---|
|`encode`|足す 483 行と評価セット v4 の 600 行をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、上の行を層 17・26・35 のグラフに流し、`h` を集める（途中で止めても済んだ分は残る）|EdgeSLM_Android|
|`heads`|P7 の 1,101 行の `h` に足す行の `h` を繋ぎ、ヘッドを学習・評価（EdgeSLM_MultiHead_Multi のコードをそのまま呼ぶ。交差検証はしない）|EdgeSLM_MultiHead_Multi|
|`text`|BERT（案 E・案 F × seed 3 つ）を学習し、学習側の交差検証と評価セット（v4・v3・v2・v1・test）の確率を保存|EdgeSLM_MultiHead_Multi（GPU）|
|`judge`|評価セット v4（合否）と v3（直す用。参考）で、4 つの構成の KPI と部品ごとの精度を出す。numpy と PyYAML だけ|どちらでも|

    python -m app.p18_retrain encode
    python -m app.p18_retrain device --layers 17,26,35
    python -m app.p18_retrain heads --layers 17,26,35
    python -m app.p18_retrain text
    python -m app.p18_retrain judge --layers 17,26,35      # results/P18/retrain.md・retrain.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p5_retrain import P3
from .p6_seeds import SEEDS, auc, head_probs, load_pt
from .p7_block import HEADS, WORK as P7
from .p7_data import CH1, CH2, TR2, read_jsonl
from .p8_gate import BERT_LO, STEPS, run
from .p10_reject import SRC as SRC933, make_hook as hook933, quantile
from .p11_bert import MODES, by_mode
from .p11_data import ADD as BERT_ADD1
from .p13_data import BERT_ADD2, CH3
from .p13_eval import WORK as P13, cell, hi_rule, interval, load_eval3
from .p16_parts import J, ORDER, WRITE, ms
from .p17_arguments import make_resolve
from .p17_rules import make_rules
from .p18_data import CH4, NOTES, TR3

WORK = PROJECT_ROOT / "work" / "p18"
RES = PROJECT_ROOT / "results" / "P18"
T = 64
BERT_SEEDS = (0, 1, 2)
SRC1584 = "1,584 行（P18）"
HEADS[SRC1584] = lambda n: WORK / f"L{n:02d}" / "heads"          # 流れ（p8_gate.run）から選べるように、ヘッドの置き場所を足す
VARIANTS = [("E", "いまの BERT（P13 の案 E）", False), ("F", "案 E ＋ train_add_v3 の新しい 285 行", True)]
CONFIGS = [("いまの構成（P17 で採った構成）", SRC933, "E"), ("＋BERT を学習し直す", SRC933, "F"),
           ("＋ヘッドを学習し直す", SRC1584, "E"), ("＋両方（本命）", SRC1584, "F")]
PRIMARY = "＋両方（本命）"
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("重大な誤りの割合", "severe", 4), ("不要な聞き返し", "over", 3),
       ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3), ("対象を断った（件）", "known_refused_n", 1)]
# 検証の目標（評価方針 2 節）。合否は評価セット v4・本命の構成で見る
TARGETS = [("EtoE の成功", "e2e", "以上", 0.95), ("重大な誤りの割合", "severe", "以下", 0.01), ("不要な聞き返し", "over", "以下", 0.05),
           ("通常の誤り", "normal", "以下", 0.03)]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}


def layers_of(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", ",").split(",") if x]


# ---------------------------------------------------------------- encode・device

def encode() -> None:
    from transformers import AutoTokenizer
    from .p2_device import G1
    from .p4_eval import render_f2
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    o = json.loads((P3 / "L17" / "head" / "samples.json").read_text(encoding="utf-8"))
    pre, suf = o["chat_prefix"], o["chat_suffix"]
    enc = lambda r: tok(pre + render_f2(r) + suf, add_special_tokens=False)["input_ids"]
    g1 = json.loads((G1 / "token_ids.json").read_text(encoding="utf-8"))
    base = [r for r in read_jsonl(TR2 / "utterances.jsonl") if "added" not in r and not r["history"]]
    pick = base[:: max(1, len(base) // 40)]
    bad = [r["id"] for r in pick if enc(r) != g1[r["id"]]]
    print(f"符号化の確かめ（714 行のうち履歴なし {len(pick)} 行 ↔ G1 のトークン列）: 不一致 {len(bad)}")
    if bad:
        raise SystemExit(f"714 行と同じ符号化にならない: {bad[:5]}")
    WORK.mkdir(parents=True, exist_ok=True)
    for kind, rows in (("train", read_jsonl(TR3 / "added.jsonl")), ("eval", read_jsonl(CH4 / "utterances.jsonl"))):
        ids = {r["id"]: enc(r) for r in rows}
        over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
        if over:
            raise SystemExit(f"{kind}: {T} トークンを超える行がある: {over}")
        ln = [len(v) for v in ids.values()]
        (WORK / f"token_ids_{kind}.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"{kind}: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)}（グラフは {T} まで）")


def device(ns: list[int]) -> None:
    from . import p2_device as D
    D.ensure_app()
    parts = WORK / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for kind in ("train", "eval"):
        ids = json.loads((WORK / f"token_ids_{kind}.json").read_text(encoding="utf-8"))
        keys = list(ids)
        for n in D.graphs(ns):
            f = WORK / f"device_{kind}_L{n}.json"
            if f.exists() and [r["id"] for r in json.loads(f.read_text(encoding="utf-8"))["rows"]] == keys:
                print(f"層 {n}・{kind}: 済み（{f.name}）", flush=True)
                continue
            got, t0 = [], time.time()
            for k, s in enumerate(range(0, len(keys), D.BATCH)):
                part = keys[s:s + D.BATCH]
                pf = parts / f"{kind}_L{n}_{k:02d}.json"                # 途中で止まっても、済んだ分は残す
                if pf.exists():
                    rows = json.loads(pf.read_text(encoding="utf-8"))["rows"]
                    if [r["id"] for r in rows] == part:
                        got += rows
                        continue
                res = D.run_job({**D.base_job(f"p18_{kind}_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
                rs = res.get("results", [])
                bad = [(i, r.get("error")) for i, r in zip(part, rs) if not r.get("ok")]
                if len(rs) != len(part) or bad:
                    raise SystemExit(f"層 {n}・{kind}・{k}: 通らない行 {bad[:3]}（{len(rs)}/{len(part)}）")
                rows = [{"id": i, "last": r["h_last"], "mean": r["h_mean"], "run_ms": r["run_ms"]} for i, r in zip(part, rs)]
                pf.write_text(json.dumps({"rows": rows}), encoding="utf-8")
                got += rows
                print(f"  層 {n} 個・{kind} {k + 1}: {len(part)} 行・NPU 中央値 {np.median([r['run_ms'] for r in rs]):.1f}ms（通算 {time.time() - t0:.0f} 秒）", flush=True)
            f.write_text(json.dumps({"rows": got}), encoding="utf-8")
            print(f"層 {n}・{kind}: {len(got)} 行 → {f}", flush=True)


def load_eval4(n: int):
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH4 / "utterances.jsonl") if r["id"] in got]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


# ---------------------------------------------------------------- heads

def features(n: int) -> None:
    src = P7 / f"L{n:02d}" / "features"
    z = np.load(src / "features_F2.npz")
    meta = json.loads((src / "features_F2.meta.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in read_jsonl(TR3 / "utterances.jsonl")]
    k = len(meta["ids"])
    if ids[:k] != meta["ids"]:
        raise SystemExit(f"層 {n}: 合わせた学習データの先頭 {k} 行が、P7 の特徴の並びと違う")
    got = {r["id"]: r for r in json.loads((WORK / f"device_train_L{n}.json").read_text(encoding="utf-8"))["rows"]}
    miss = [i for i in ids[k:] if i not in got]
    if miss:
        raise SystemExit(f"層 {n}: 端末の h が無い行 {miss[:5]}（先に device）")
    last = np.vstack([z["last__last"], np.array([got[i]["last"] for i in ids[k:]], np.float32)]).astype(np.float32)
    mean = np.vstack([z["last__mean"], np.array([got[i]["mean"] for i in ids[k:]], np.float32)]).astype(np.float32)
    d = WORK / f"L{n:02d}" / "features"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "features_F2.npz", last__last=last, last__mean=mean)
    m = {"format": "F2", "ids": ids, "layers": ["last"], "poolings": ["last", "mean"],
         "meta": {"source": f"device (Xperia 5 III, 層 {n} 個のまとめ読み T=64, QNN HTP)。先頭 {k} 行は P2・P5・P7、残りは P18", "n_layers_kept": n}}
    (d / "features_F2.meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"層 {n}: 特徴 {last.shape}（{k} 行＋{len(ids) - k} 行）", flush=True)


def heads(ns: list[int], force: bool) -> None:
    use_multi_library()
    from edge_router.heads import evaluate, train
    from .p1_compare import summarize_evaluate

    def call(module, argv: list[str]) -> None:
        print(f"\n===== {module.__name__} =====", flush=True)
        sys.argv = [module.__name__] + argv
        module.main()

    for n in ns:
        base = WORK / f"L{n:02d}"
        if not (WORK / f"device_train_L{n}.json").exists():
            print(f"層 {n}: 足した行の端末の h が無い（先に device）。飛ばす", flush=True)
            continue
        print(f"\n########## 層 {n} 個（1,101 行＋足した 483 行）##########", flush=True)
        features(n)
        if (base / "heads" / "evaluate_report.json").exists() and not force:
            print(f"層 {n}: ヘッドは作成済み（作り直すなら --force）", flush=True)
            continue
        common = ["--data", str(TR3 / "utterances.jsonl"), "--features", str(base / "features")]
        call(train, common + ["--out", str(base / "heads"), "--force"])
        call(evaluate, common + ["--run", str(base / "heads")])
    print(f"\n{'学習データ':<18} {'test の正解率（mlp・seed 7）':>24}")
    for n in ns:
        for src in ("1,101 行（P7）", SRC1584):
            ev = summarize_evaluate(HEADS[src](n) / "evaluate_report.json")
            t = ev.get("mlp_s7_test_accuracy")
            print(f"層 {n:>2}・{src:<14} {'—' if t is None else f'{t:.4f}'}", flush=True)
    print("（1,101 行と 1,584 行は test の行が違うので、数字をそのまま比べない。比べるのは judge の評価セット v4）")


# ---------------------------------------------------------------- text（GPU）

def fit_set(with_new: bool) -> tuple[list[list[dict]], dict]:
    """学習の行（組ごと）とラベル（1 = 機能にない依頼）。もとの 882 行・対象側の 90 行・機能にない依頼 108 行・（案 F だけ）新しい 285 行"""
    base = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] in ("train", "validation")]
    parts = [base, read_jsonl(BERT_ADD1 / "added.jsonl"), read_jsonl(BERT_ADD2 / "added.jsonl")]
    if with_new:
        parts.append([r for r in read_jsonl(TR3 / "added.jsonl") if r.get("added") == "train_add_v3"])
    y = {r["id"]: int(r.get("add_type") == "OS") for part in parts for r in part}
    return parts, y


def text(device_name: str, seeds: tuple) -> None:
    use_multi_library()
    from edge_router.heads.crossval import make_folds
    from edge_router.l1.clear.train import fit_predict_bert
    from edge_router.l1.decide import load_config
    cfg = load_config()["model"]
    pred_rows = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] == "test"] + read_jsonl(CH4 / "utterances.jsonl") + \
        read_jsonl(CH3 / "utterances.jsonl") + read_jsonl(CH2 / "utterances.jsonl") + read_jsonl(CH1)
    WORK.mkdir(parents=True, exist_ok=True)
    p13 = P13 / "text_scores_E.json"
    old = json.loads(p13.read_text(encoding="utf-8")) if p13.exists() else None
    print(f"BERT: {cfg['name']}・{cfg['epochs']} epoch・seed {[cfg['seed'] + s for s in seeds]} / 確率を出す行 {len(pred_rows)}（test・評価セット v4・v3・v2・v1）", flush=True)
    for key, title, with_new in VARIANTS:
        parts, y = fit_set(with_new)
        fit_rows = [r for part in parts for r in part]
        fold = {}
        for part in parts:                                           # 組ごとに分ける（もとの 882 行の分け方は、どの案でも同じ）
            fold.update(make_folds(part, 5))
        folds = np.array([int(fold[r["id"]].replace("fold", "")) for r in fit_rows])
        yy = np.array([y[r["id"]] for r in fit_rows])
        texts = [r["utterance"] for r in fit_rows]
        f = WORK / f"text_scores_{key}.json"
        out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"variant": key, "title": title, "labels_fit": y, "models": {}, "bert": cfg, "check": {}}
        if out["labels_fit"] != y:
            raise SystemExit(f"案 {key}: 保存済みの結果とラベルが違う（データを変えたなら {f.name} を別の名前にしてから実行）")
        print(f"\n##### 案 {key}（{title}）: 学習 {len(fit_rows)} 行（機能にない依頼 {int(yy.sum())}・それ以外 {int((1 - yy).sum())}）#####", flush=True)
        for s in seeds:
            name = f"bert_s{s}"
            m = out["models"].get(name)
            if m and all(r["id"] in m["pred"] for r in pred_rows):
                print(f"{name}: 済み", flush=True)
                continue
            t0 = time.time()
            kw = {"cfg": cfg, "device": device_name, "seed": cfg["seed"] + s}
            prev = old["models"].get(name) if (key == "E" and old and old.get("labels_fit") == y) else None
            if prev:                                                 # 案 E の交差検証は P13 と同じ学習なので、P13 の結果を使う（全体の学習だけやり直す）
                cv = np.array([prev["cv"][r["id"]] for r in fit_rows])
            else:
                cv = np.zeros(len(fit_rows))
                for k in range(5):
                    tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
                    cv[te] = fit_predict_bert([texts[i] for i in tr], yy[tr], [texts[i] for i in te], **kw)
                    print(f"  {name}: 交差検証 {k + 1}/5", flush=True)
            pred = fit_predict_bert(texts, yy, [r["utterance"] for r in pred_rows], **kw)
            out["models"][name] = {"cv": {r["id"]: float(p) for r, p in zip(fit_rows, cv)},
                                   "pred": {r["id"]: float(p) for r, p in zip(pred_rows, pred)}}
            note = ""
            if prev:                                                 # 確かめ: 学習し直した確率が、P13 のときと同じか（同じ行で比べる）
                diff = [abs(float(p) - prev["pred"][r["id"]]) for r, p in zip(pred_rows, pred) if r["id"] in prev["pred"]]
                out.setdefault("check", {})[name] = {"rows": len(diff), "max_abs_diff": float(max(diff)) if diff else None}
                note = f" / P13 のときの確率との差の最大 {max(diff):.4f}（{len(diff)} 行。交差検証は P13 の結果を使った）" if diff else ""
            f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            neg = cv[yy == 0]
            print(f"{name}: 学習側の交差検証 AUC {auc(cv[yy == 1], neg):.3f} / それ以外の行でフラグ（{BERT_LO} 以上）{float((neg >= BERT_LO).mean()):.3f}・"
                  f"99.5% 点 {np.quantile(neg, 0.995, method='higher'):.3f} / {time.time() - t0:.0f} 秒{note}", flush=True)
        print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge（numpy だけ）

def hook1584(n: int, src: str, heads: list) -> list:
    """1,584 行のヘッドの棄却の閾値を、validation の行き先のある行の 2% を誤って回す値にする（P10 と同じ決め方）"""
    rows = read_jsonl(TR3 / "utterances.jsonl")
    d = WORK / f"L{n:02d}" / "features"
    ids = json.loads((d / "features_F2.meta.json").read_text(encoding="utf-8"))["ids"]
    if ids != [r["id"] for r in rows]:
        raise SystemExit(f"層 {n}: 1,584 行の特徴の並びがデータと違う")
    pick = [i for i, r in enumerate(rows) if r["split"] == "validation" and r["route"] != "UNKNOWN"]
    z = np.load(d / "features_F2.npz")
    last, mean = z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)
    hd = HEADS[src](n)
    reps = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
    out = []
    for (seed, classes, th, p) in heads:
        name = f"heads_mlp_seed{seed}.pt"
        val = head_probs(load_pt(hd / name), float(reps[name]["temperature"]), last, mean)["reject"]
        out.append((seed, classes, {**th, "REJECT": quantile(val, 0.02)}, p))
    return out


def hook_for(src: str):
    return hook933("val", 0.02, {}) if src == SRC933 else hook1584


def has_heads(n: int, src: str) -> bool:
    return (HEADS[src](n) / "evaluate_report.json").exists()


def cached(loader):
    """評価セットの h を、層ごとに 1 回だけ読む"""
    memo = {}

    def load(n: int):
        if n not in memo:
            memo[n] = loader(n)
        return memo[n]
    return load


def measure(n: int, src: str, ts: dict, loader):
    """流れ全体（P17 で採った構成の決まり）で、層 n・ヘッド src・BERT ts の KPI と内訳を出す"""
    if not has_heads(n, src) or loader(n) is None:
        return None
    final = STEPS[-1]
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=hook_for(src), steps=[final], sources=[src], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=make_rules, amb_chat_ok=True)
    key = f"{n}|{src}|{final[0]}"
    return blob["kpi"][key], blob["detail"][key]


def bert_part(ts: dict, rows: list[dict]) -> dict:
    """BERT だけ（確率だけで数える）。フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上"""
    fit = list(ts["labels_fit"])
    y = np.array([ts["labels_fit"][i] for i in fit])
    ids = [r["id"] for r in rows]
    cat = np.array([r["category"] for r in rows])
    grp = np.array([r.get("os_group") or "" for r in rows])
    known = np.array([r["action"] != "UNKNOWN" for r in rows])
    isH = cat == "H"
    out = collections.defaultdict(list)
    for name in sorted(k for k in ts["models"] if k.startswith("bert_s")):
        m = ts["models"][name]
        hi = hi_rule(name, np.array([m["cv"][i] for i in fit])[y == 0])
        p = np.array([m["pred"][i] for i in ids])
        out["hi"].append(hi)
        out["H_flag"].append(float(np.mean(p[isH] >= BERT_LO)))
        out["H_refuse"].append(float(np.mean(p[isH] >= hi)))
        for g in sorted(set(grp[isH])):
            out[f"H_flag_{g}"].append(float(np.mean(p[isH & (grp == g)] >= BERT_LO)))
            out[f"H_refuse_{g}"].append(float(np.mean(p[isH & (grp == g)] >= hi)))
        out["known_flag"].append(float(np.mean(p[known] >= BERT_LO)))
        for c in "KNF":
            out[f"{c}_flag"].append(float(np.mean(p[cat == c] >= BERT_LO)))
        out["known_refuse_n"].append(int((p[known] >= hi).sum()))
        out["G_flag"].append(float(np.mean(p[cat == "G"] >= BERT_LO)))
        out["auc_known"].append(auc(p[isH], p[known]))
        out["accuracy_flag"].append(float(np.mean((p >= BERT_LO) == isH)))
    return dict(out)


def head_part(n: int, src: str, rows: list[dict], last, mean, effect: dict) -> dict | None:
    """LM ヘッドだけ（規則・引数・BERT は外す）。行き先の正解率と、棄却・確信度で止めた割合"""
    if not has_heads(n, src):
        return None
    d = HEADS[src](n)
    reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
    heads = []
    for seed in SEEDS:
        name = f"heads_mlp_seed{seed}.pt"
        blob, rep = load_pt(d / name), reps[name]
        heads.append((seed, blob["metadata"]["action_classes"], {k: float(v) for k, v in rep["thresholds"].items()},
                      head_probs(blob, float(rep["temperature"]), last, mean)))
    heads = hook_for(src)(n, src, heads)
    act = np.array([r["action"] for r in rows])
    cat = np.array([r["category"] for r in rows])
    known, isH, isG = act != "UNKNOWN", cat == "H", cat == "G"
    out = collections.defaultdict(list)
    conf_pairs = collections.Counter()
    for seed, classes, th, p in heads:
        pred = np.array([classes[i] for i in p["action"].argmax(1)])
        conf = p["action"].max(1)
        tau = np.array([th["CHAT"] if c == "CHAT" else th["TOOL_write" if effect.get(c[5:]) == "write" else "TOOL_read"] for c in pred])
        rej = p["reject"] >= th["REJECT"]
        ok = ~rej & ~(conf < tau)                                    # ヘッドが通した（棄却でも確信度不足でもない）
        right = pred == act
        out["acc"].append(float(np.mean(right[known])))
        for c in "KNF":
            out[f"acc_{c}"].append(float(np.mean(right[cat == c])))
        for c in ORDER:
            out[f"recall_{J[c]}"].append(float(np.mean(right[known & (act == c)])))
        for t_, p_ in zip(act[known & ~right], pred[known & ~right]):
            conf_pairs[(t_, p_)] += 1
        out["cover"].append(float(np.mean(ok[known])))
        out["wrong_write"].append(int(np.isin(pred[known & ok & ~right], WRITE).sum()))
        for name, sel in (("H", isH), ("G", isG)):
            out[f"{name}_stop"].append(float(np.mean(~ok[sel])))
            out[f"{name}_pass_write"].append(float(np.mean(ok[sel] & np.isin(pred[sel], WRITE))))
            out[f"auc_{name}"].append(auc(p["reject"][sel], p["reject"][known]))
        out["taus"].append(float(th["REJECT"]))
    res = dict(out)
    res["confusions"] = [{"true": J[t_], "pred": J[p_], "rows": v / len(heads)} for (t_, p_), v in conf_pairs.most_common(8)]
    return res


def verdict(k: dict, n_rows: int, n_known: int) -> list[dict]:
    """検証の目標と比べる。点推定と、行数から出した 95% 信頼区間（Wilson）の両方で見る"""
    out = []
    for lab, key, op, target in TARGETS:
        v = k[key]["mean"]
        n = n_known if key == "over" else n_rows
        lo, hi = (float(x) for x in interval(v, n).split("〜"))
        point = v >= target if op == "以上" else v <= target
        sure = lo >= target if op == "以上" else hi <= target            # 区間ごと目標の内側にある
        out.append({"kpi": lab, "key": key, "value": v, "target": f"{target:g} {op}", "ci": f"{lo:.3f}〜{hi:.3f}", "rows": n,
                    "point": bool(point), "sure": bool(sure)})
    return out


def judge(ns: list[int]) -> None:
    tss = {}
    checks = {}
    for key, _, _ in VARIANTS:
        f = WORK / f"text_scores_{key}.json"
        if f.exists():
            raw = json.loads(f.read_text(encoding="utf-8"))
            if raw["models"]:
                tss[key] = by_mode(raw, MODES[0])
                checks[key] = raw.get("check", {})
    if "E" not in tss:
        raise SystemExit("BERT の結果が無い（先に text）")
    effect = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    load4 = cached(load_eval4)
    sets = [("v4", "評価セット v4（合否を測る）", read_jsonl(CH4 / "utterances.jsonl"), load4),
            ("v3", "評価セット v3（直す用。参考）", read_jsonl(CH3 / "utterances.jsonl"), cached(load_eval3))]
    configs = [x for x in CONFIGS if x[2] in tss]
    print("\n===== P18 judge: ヘッドと BERT を学習し直す =====")
    print("どの構成も: 引数 p17_arguments・規則 p17_rules・棄却の閾値は validation の 2%・BERT の断る閾値は学習側の 99.5% 点・フラグはすべての機能・P15 の数え方")
    print(f"あらかじめ決めた本命: {PRIMARY} / 合否は評価セット v4 で見る（v3 は直す用）")
    md = ["数字はヘッドの seed 3 つ × BERT の seed 3 つの平均（かっこは最小〜最大）。信頼区間は平均の割合と行数から出した目安（Wilson・95%）。"
          f"あらかじめ決めた本命は「{PRIMARY}」。**合否は評価セット v4 で見る**（評価セット v3 は P13〜P17 で手当てを作るのに使った）\n"]
    res = {"kpi": {}, "verdict": {}, "stops": {}, "severe": {}, "detail": {}, "bert": {}, "head": {}, "bert_check": checks}
    if checks.get("E"):
        line = "・".join(f"{k} {v['max_abs_diff']:.4f}" for k, v in checks["E"].items() if v.get("max_abs_diff") is not None)
        print(f"確かめ: 案 E を学習し直した確率と P13 のときの確率の差の最大（seed ごと）: {line}")
        md.append(f"確かめ: 案 E を学習し直した確率と、P13 のときの確率の差の最大（seed ごと）: {line}\n")

    for tag, title, rows, loader in sets:
        n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
        c = {k: sum(r["category"] == k for r in rows) for k in NOTES}
        print(f"\n################ {title}: {n_rows} 行（" + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + "）################")
        md.append(f"\n## {title}: {n_rows} 行（" + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + "）\n")
        for n in ns:
            keep = {}
            for cname, src, bert in configs:
                got = measure(n, src, tss[bert], loader)
                if got is not None:
                    keep[cname] = got
            if not keep:
                print(f"層 {n}: 測れる構成が無い（端末の h かヘッドが無い）")
                continue
            print(f"\n===== {tag}・層 {n} 個 =====")
            heads_ = [lab for lab, _, _ in KPI] + ["EtoE の成功の区間", "重大な誤りの割合の区間"]
            md += [f"\n### {tag}・層 {n} 個\n", "|構成|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
            for cname, (k, d) in keep.items():
                cells = [cell(k[key], nd) for _, key, nd in KPI] + [interval(k["e2e"]["mean"], n_rows), interval(k["severe"]["mean"], n_rows)]
                res["kpi"][f"{tag}|{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, "e2e_ci": cells[-2], "severe_ci": cells[-1]}
                mark = "**" if cname == PRIMARY else ""
                print(f"{cname:<18} " + "  ".join(f"{h}: {x}" for h, x in zip(heads_, cells)))
                md.append(f"|{mark}{cname}{mark}|" + "|".join(cells) + "|")
            # 合否（本命）
            if PRIMARY in keep:
                v = verdict(keep[PRIMARY][0], n_rows, n_known)
                res["verdict"][f"{tag}|{n}"] = v
                md += [f"\n本命を検証の目標と比べる（{tag}・層 {n} 個）:\n", "|KPI|目標|値|目標を満たすか（平均の値）|区間（行数）|区間ごと目標の内側か|", "|---|---|---|---|---|---|"]
                for x in v:
                    md.append(f"|{x['kpi']}|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|")
                print(f"  本命と目標: " + " / ".join(f"{x['kpi']} {x['value']:.4f}（目標 {x['target']}）{'○' if x['point'] else '×'}・区間 {x['ci']}{'○' if x['sure'] else '×'}" for x in v))
            # 対象の行を止めた段・重大な誤りの内訳
            md += [f"\n対象の依頼と雑談 {n_known} 行を止めた段（1 回あたりの行数）:\n", "|構成|規則|引数が足りない|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|検査|合計|",
                   "|---|---|---|---|---|---|---|---|---|"]
            for cname, (k, d) in keep.items():
                cnt = collections.Counter()
                for x in d["stop_rows"]:
                    cnt[x["stop"]] += x["share"]
                v = [cnt["聞き返し（規則）"], cnt["聞き返し（引数が足りない）"], cnt["聞き返し（棄却ヘッド）"], cnt["聞き返し（確信度）"], cnt["聞き返し（BERT のフラグ）"],
                     cnt["拒否（BERT が断る）"], cnt["聞き返し（検査）"] + cnt["拒否（検査）"]]
                res["stops"][f"{tag}|{n}|{cname}"] = v
                md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in v) + f"|{sum(v):.1f}|")
                print(f"  {cname[:12]:<12} 止めた段: 規則 {v[0]:.1f}・引数 {v[1]:.1f}・棄却 {v[2]:.1f}・確信度 {v[3]:.1f}・BERT フラグ {v[4]:.1f}・BERT 断る {v[5]:.1f}・検査 {v[6]:.1f}")
            md += ["\n重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|合計|", "|---|" + "---|" * (len(CAT) + 1)]
            for cname, (k, d) in keep.items():
                cnt = collections.Counter()
                for x in d["by_group"]:
                    if x["grade"] == "重大":
                        cnt[x["category"]] += x["rows"]
                res["severe"][f"{tag}|{n}|{cname}"] = dict(cnt)
                md.append(f"|{cname}|" + "|".join(f"{cnt[k_]:.1f}" for k_ in CAT) + f"|{sum(cnt.values()):.1f}|")
                print(f"  {cname[:12]:<12} 重大: " + "・".join(f"{CAT[k_]} {cnt[k_]:.1f}" for k_ in CAT))
            # 種類ごとの内訳（最初の構成と本命）
            show = [x for x in (configs[0][0], PRIMARY) if x in keep]
            md += [f"\n種類ごとの内訳（{tag}・層 {n} 個。割合）:\n", "|種類|行|" + "|".join(f"{x}: 正解／回復可能／通常／重大" for x in show) + "|", "|---|---|" + "---|" * len(show)]
            gkeys = []
            for r in rows:
                gk = (r["category"], (r.get("os_group") or r.get("g_kind")) if r["category"] in "HG" else None)
                if gk not in gkeys:
                    gkeys.append(gk)
            for cat_, grp in gkeys:
                sel = [r for r in rows if r["category"] == cat_ and (grp is None or (r.get("os_group") or r.get("g_kind")) == grp)]
                cells = []
                for x in show:
                    bg = keep[x][1]["by_group"]
                    tot = {g: sum(y["rows"] for y in bg if y["category"] == cat_ and (grp is None or y["group"] == grp) and y["grade"] == g)
                           for g in ("正解", "回復可能", "通常", "重大")}
                    cells.append("／".join(f"{tot[g] / len(sel):.3f}" for g in ("正解", "回復可能", "通常", "重大")))
                md.append(f"|{NOTES[cat_]}" + (f"・{grp}" if grp else "") + f"|{len(sel)}|" + "|".join(cells) + "|")
            for x in show:
                d = keep[x][1]
                res["detail"][f"{tag}|{n}|{x}"] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows", "flag_rows", "by_group", "stages")}
                md += [f"\n{x}のとき（{tag}・層 {n} 個）:\n", "|見るもの|行（出た割合）|", "|---|---|",
                       "|重大な誤り|" + " / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:30]) + "|",
                       "|通常の誤り|" + " / ".join(f"[{y['category']}]{y['utterance']} → {y['outcome']}（{y['share']:.2f}）" for y in d["normal_rows"][:30]) + "|",
                       "|止めた対象の行|" + " / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:40]) + "|"]
            if PRIMARY in keep:
                print("  本命で残る重大な誤り: " + " / ".join(f"[{y['category']}]{y['utterance']}→{y['executed']}（{y['share']:.2f}）" for y in keep[PRIMARY][1]["severe_rows"][:12]))

    # ---- 部品ごと（評価セット v4）
    rows4 = read_jsonl(CH4 / "utterances.jsonl")
    md.append("\n## 部品ごとの精度（評価セット v4。規則・引数・対象の語の確かめは外す）\n")
    bp = {key: bert_part(tss[key], rows4) for key in tss if all(r["id"] in next(iter(tss[key]["models"].values()))["pred"] for r in rows4)}
    if bp:
        labels = [("断る閾値（seed ごと）", "hi"), ("機能にない依頼にフラグ以上（再現率）", "H_flag"), ("　学習に同じ機能がある", "H_flag_seen"), ("　どの学習にも無い機能", "H_flag_unseen"),
                  ("機能にない依頼を断る閾値以上", "H_refuse"), ("　学習に同じ機能がある", "H_refuse_seen"), ("　どの学習にも無い機能", "H_refuse_unseen"),
                  ("対象の依頼と雑談に誤ってフラグ", "known_flag"), ("　機能の語を含む依頼", "K_flag"), ("　ふつうの依頼", "N_flag"), ("　雑談", "F_flag"),
                  ("対象を断った（行。seed ごと）", "known_refuse_n"), ("意図が不明にフラグ", "G_flag"), ("2 値の正解率（フラグ）", "accuracy_flag"),
                  ("AUC（機能にない依頼 ↔ 対象の依頼と雑談）", "auc_known")]
        print("\n===== 部品: BERT（評価セット v4。フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上）=====")
        md += ["### BERT（フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上）\n", "|指標|" + "|".join(f"案 {k}" for k in bp) + "|", "|---|" + "---|" * len(bp)]
        for lab, key in labels:
            cells = ["・".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in bp[k].get(key, [])) if key in ("hi", "known_refuse_n") else
                     (ms(bp[k][key]) if key in bp[k] else "—") for k in bp]
            print(f"{lab:<26} " + "  ".join(f"[案 {k}] {x}" for k, x in zip(bp, cells)))
            md.append(f"|{lab}|" + "|".join(cells) + "|")
        res["bert"] = bp
    for n in ns:
        loaded = load4(n)
        if loaded is None:
            continue
        rows, last, mean = loaded
        hp = {src: head_part(n, src, rows, last, mean, effect) for src in (SRC933, SRC1584)}
        hp = {k: v for k, v in hp.items() if v}
        if not hp:
            continue
        labels = [("行き先の正解率（対象の依頼と雑談）", "acc", 3), ("　機能の語を含む依頼", "acc_K", 3), ("　ふつうの依頼", "acc_N", 3), ("　雑談", "acc_F", 3)] + \
                 [(f"　再現率: {J[c_]}", f"recall_{J[c_]}", 3) for c_ in ORDER] + \
                 [("対象の依頼と雑談を通した割合", "cover", 3), ("通したうち、別の操作系と判定（行）", "wrong_write", 1),
                  ("機能にない依頼を止めた（棄却・確信度）", "H_stop", 3), ("　操作系として通した", "H_pass_write", 3), ("　棄却の得点の AUC", "auc_H", 3),
                  ("意図が不明を止めた（棄却・確信度）", "G_stop", 3), ("　操作系として通した", "G_pass_write", 3), ("　棄却の得点の AUC", "auc_G", 3)]
        print(f"\n===== 部品: LM ヘッド・層 {n} 個（評価セット v4）=====")
        md += [f"\n### LM ヘッド・層 {n} 個（mlp・seed 3 つ。棄却の閾値は validation の 2%）\n", "|指標|" + "|".join(hp) + "|", "|---|" + "---|" * len(hp)]
        for lab, key, nd in labels:
            cells = [ms(hp[s][key], nd) for s in hp]
            print(f"{lab:<28} " + "  ".join(f"[{s[:8]}] {x}" for s, x in zip(hp, cells)))
            md.append(f"|{lab}|" + "|".join(cells) + "|")
        md.append("|棄却の閾値（seed ごと）|" + "|".join("・".join(f"{t:.3f}" for t in hp[s]["taus"]) for s in hp) + "|")
        for s in hp:
            md.append(f"\n{s} の取り違えの多い組（正解 → 判定。1 つのヘッドあたりの行数）: " + " / ".join(f"{x['true']} → {x['pred']} {x['rows']:.1f}" for x in hp[s]["confusions"]))
        res["head"][str(n)] = hp
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "retrain.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "retrain.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（retrain.md・retrain.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "heads", "text", "judge"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in BERT_SEEDS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P18 {a.step} =====", flush=True)
    if a.step == "encode":
        encode()
    elif a.step == "device":
        device(ns)
    elif a.step == "heads":
        heads(ns, a.force)
    elif a.step == "text":
        text(a.device, tuple(int(x) for x in a.seeds.split(",") if x))
    else:
        judge(ns)


if __name__ == "__main__":
    main()
