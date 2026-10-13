# -*- coding: utf-8 -*-
"""P7 手順 2: 機能にない依頼の止め方を 2 通り試し、新しい評価セット（challenge_v2）で比べる。

設計: DoC/設計書_P7_機能にない依頼の止め方.md の 4〜6 節。

|方法|中身|
|---|---|
|ヘッドの学習データを増やす|933 行に train_add_v2（168 行）を足した 1,101 行で、ヘッドを学習し直す（グラフは変えない）|
|外の部品で止める|発話の文だけを読む 2 値の分類器（機能にない依頼か、それ以外か）を別に作り、止めたらヘッドの判定より先に「確認へ回す」。日本語 BERT と、比べる下限の文字 n-gram|

|段|何をするか|venv|
|---|---|---|
|`encode`|足す 168 行と評価セット v2 の 200 行をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、上の行を層 17・26・35 のグラフに流し、`h` を集める|EdgeSLM_Android|
|`heads`|P5 の 933 行の `h` に足す行の `h` を繋ぎ、ヘッドを学習・評価・交差検証（EdgeSLM_MultiHead_Multi のコードをそのまま呼ぶ）|EdgeSLM_MultiHead_Multi|
|`text`|文だけの分類器（n-gram・BERT）を学習し、学習側の交差検証・評価セットの確率を保存。EdgeSLM_MultiHead_Multi の `l1.clear.train` の学習関数をそのまま呼ぶ|EdgeSLM_MultiHead_Multi（GPU）|
|`judge`|評価セット v2 で、ヘッドだけ（933 行・1,101 行）／文の分類器だけ／組み合わせ を並べる。numpy だけ|EdgeSLM_Android|

    python -m app.p7_block encode
    python -m app.p7_block device --layers 17,26,35
    python -m app.p7_block heads --layers 17,26,35
    python -m app.p7_block text
    python -m app.p7_block judge --layers 17,26,35
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p5_retrain import P3, WORK as P5, decide
from .p6_seeds import SEEDS, auc, head_probs, load_pt
from .p7_data import CH2, NOTES, TR2, read_jsonl

CH1 = PROJECT_ROOT / "data" / "challenge_v1" / "utterances.jsonl"
WORK = PROJECT_ROOT / "work" / "p7"
RES = PROJECT_ROOT / "results" / "P7"
T = 64
COMMON = {"TOOL_write": 0.5, "TOOL_read": 0.3, "CHAT": 0.4}       # そろえて比べるときの確信度の閾値（P6 と同じ）
MATCH_OVER = 0.05                                                  # 同じ手間: 行き先のある行の 5% を確認へ回す
FINE_GRID = [round(x, 3) for x in np.linspace(0.005, 0.995, 199)] + [1.01]
BERT_SEEDS = (0, 1)
SAFE = ("REJECT", "CLARIFY", "BLOCK")
HEADS = {"933 行（P5）": lambda n: P5 / f"L{n:02d}" / "heads", "1,101 行（P7）": lambda n: WORK / f"L{n:02d}" / "heads"}


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
    for kind, rows in (("train", read_jsonl(TR2 / "added.jsonl")), ("eval", read_jsonl(CH2 / "utterances.jsonl"))):
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
    for kind in ("train", "eval"):
        ids = json.loads((WORK / f"token_ids_{kind}.json").read_text(encoding="utf-8"))
        keys = list(ids)
        for n in D.graphs(ns):
            f = WORK / f"device_{kind}_L{n}.json"
            if f.exists() and [r["id"] for r in json.loads(f.read_text(encoding="utf-8"))["rows"]] == keys:
                print(f"層 {n}・{kind}: 済み（{f.name}）", flush=True)
                continue
            got = []
            for k, s in enumerate(range(0, len(keys), D.BATCH)):
                part = keys[s:s + D.BATCH]
                res = D.run_job({**D.base_job(f"p7_{kind}_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
                rs = res.get("results", [])
                bad = [(i, r.get("error")) for i, r in zip(part, rs) if not r.get("ok")]
                if len(rs) != len(part) or bad:
                    raise SystemExit(f"層 {n}・{kind}・{k}: 通らない行 {bad[:3]}（{len(rs)}/{len(part)}）")
                got += [{"id": i, "last": r["h_last"], "mean": r["h_mean"], "run_ms": r["run_ms"]} for i, r in zip(part, rs)]
                print(f"  層 {n} 個・{kind} {k + 1}: {len(part)} 行・NPU 中央値 {np.median([r['run_ms'] for r in rs]):.1f}ms", flush=True)
            f.write_text(json.dumps({"rows": got}), encoding="utf-8")
            print(f"層 {n}・{kind}: {len(got)} 行 → {f}", flush=True)


# ---------------------------------------------------------------- heads

def features(n: int) -> None:
    src = P5 / f"L{n:02d}" / "features"
    z = np.load(src / "features_F2.npz")
    meta = json.loads((src / "features_F2.meta.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in read_jsonl(TR2 / "utterances.jsonl")]
    k = len(meta["ids"])
    if ids[:k] != meta["ids"]:
        raise SystemExit(f"層 {n}: 合わせた学習データの先頭 {k} 行が、P5 の特徴の並びと違う")
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
         "meta": {"source": f"device (Xperia 5 III, 層 {n} 個のまとめ読み T=64, QNN HTP)。先頭 {k} 行は P2・P5、残りは P7", "n_layers_kept": n}}
    (d / "features_F2.meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"層 {n}: 特徴 {last.shape}（933 行＋{len(ids) - k} 行）", flush=True)


def heads(ns: list[int], force: bool) -> None:
    use_multi_library()
    from edge_router.heads import crossval, evaluate, train
    from .p1_compare import summarize_crossval, summarize_evaluate

    def run(module, argv: list[str]) -> None:
        print(f"\n===== {module.__name__} =====", flush=True)
        sys.argv = [module.__name__] + argv
        module.main()

    for n in ns:
        base = WORK / f"L{n:02d}"
        if not (WORK / f"device_train_L{n}.json").exists():
            print(f"層 {n}: 足した行の端末の h が無い（先に device）。飛ばす", flush=True)
            continue
        print(f"\n########## 層 {n} 個（933 行＋足した行）##########", flush=True)
        features(n)
        if (base / "crossval" / "crossval_report.json").exists() and not force:
            print(f"層 {n}: ヘッドは作成済み（作り直すなら --force）", flush=True)
            continue
        common = ["--data", str(TR2 / "utterances.jsonl"), "--features", str(base / "features")]
        run(train, common + ["--out", str(base / "heads"), "--force"])
        run(evaluate, common + ["--run", str(base / "heads")])
        run(crossval, common + ["--out", str(base / "crossval")])
    print(f"\n{'学習データ':<22} {'交差検証 mlp':>16} {'全基準':>6} {'誤実行':>6} {'棄却再現':>8} {'test s7':>8}")
    rows = []
    for n in ns:
        for src, where in HEADS.items():
            base = where(n).parent
            cv = summarize_crossval(base / "crossval" / "crossval_report.json")
            ev = summarize_evaluate(base / "heads" / "evaluate_report.json")
            if not cv:
                continue
            m, t = cv["mlp"], ev.get("mlp_s7_test_accuracy")
            rows.append({"n_layers": n, "data": src, "crossval_mlp": m, "test_mlp_s7": t})
            print(f"層 {n:>2}・{src:<16} {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} {m['configs_all_ok']}/{m['configs']:<4} "
                  f"{m['wrong_writes']:>6} {m['rejection_recall_mean']:>8.3f} {t if t is None else f'{t:.4f}':>8}", flush=True)
    print("（933 行と 1,101 行は測っている行が違うので、数字をそのまま比べない）")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "heads_compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- text（文だけの分類器）

def text(device_name: str) -> None:
    """ラベル 1 = 機能にない依頼（add_type=OS）、0 = それ以外（機能にある依頼・雑談・意図が不明）。入力は発話の文だけ"""
    use_multi_library()
    from edge_router.heads.crossval import make_folds
    from edge_router.l1.clear.train import fit_predict_bert, fit_predict_ngram
    from edge_router.l1.decide import load_config
    cfg = load_config()["model"]
    rows = read_jsonl(TR2 / "utterances.jsonl")
    fit_rows = [r for r in rows if r["split"] in ("train", "validation")]
    test_rows = [r for r in rows if r["split"] == "test"]
    pred_rows = test_rows + read_jsonl(CH2 / "utterances.jsonl") + read_jsonl(CH1)
    y = np.array([int(r.get("add_type") == "OS") for r in fit_rows])
    texts = [r["utterance"] for r in fit_rows]
    fold = make_folds(fit_rows, 5)
    folds = np.array([int(fold[r["id"]].replace("fold", "")) for r in fit_rows])
    print(f"学習 {len(fit_rows)} 行（機能にない依頼 {int(y.sum())}・それ以外 {int((1 - y).sum())}）/ test {len(test_rows)} 行 / "
          f"評価セット v2・v1 {len(pred_rows) - len(test_rows)} 行 / BERT: {cfg['name']}・{cfg['epochs']} epoch", flush=True)
    out = {"labels_fit": {r["id"]: int(v) for r, v in zip(fit_rows, y)}, "models": {}, "bert": cfg}
    f = WORK / "text_scores.json"
    if f.exists():
        out["models"] = json.loads(f.read_text(encoding="utf-8")).get("models", {})
    jobs = [("ngram", fit_predict_ngram, {})] + [(f"bert_s{s}", fit_predict_bert, {"cfg": cfg, "device": device_name, "seed": cfg["seed"] + s})
                                                 for s in BERT_SEEDS]
    for name, fit, kw in jobs:
        if name in out["models"]:
            print(f"{name}: 済み", flush=True)
            continue
        cv = np.zeros(len(fit_rows))
        for k in range(5):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            cv[te] = fit([texts[i] for i in tr], y[tr], [texts[i] for i in te], **kw)
            print(f"  {name}: 交差検証 {k + 1}/5", flush=True)
        pred = fit(texts, y, [r["utterance"] for r in pred_rows], **kw)
        out["models"][name] = {"cv": {r["id"]: float(p) for r, p in zip(fit_rows, cv)},
                               "pred": {r["id"]: float(p) for r, p in zip(pred_rows, pred)}}
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        neg = cv[y == 0]
        print(f"{name}: 学習側の交差検証 AUC {auc(cv[y == 1], neg):.3f} / それ以外の行の確率の最大 {neg.max():.3f}・99% 点 {np.quantile(neg, 0.99):.3f}", flush=True)
    print(f"保存しました: {f}")


# ---------------------------------------------------------------- judge

def load_eval(n: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = read_jsonl(CH2 / "utterances.jsonl")
    got = {r["id"]: r for r in json.loads((WORK / f"device_eval_L{n}.json").read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in rows if r["id"] in got]
    return (rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64))


def blocked(out: list[dict], mask) -> list[dict]:
    """外の部品が止めた行は、ヘッドの判定より先に「確認へ回す」にする"""
    if mask is None:
        return out
    res = []
    for r, b in zip(out, mask):
        if b:
            r = {**r, "final": "BLOCK", "kind": "確認へ", "ok": r["expected"] == "UNKNOWN"}
        res.append(r)
    return res


def score(out: list[dict], rows: list[dict]) -> dict:
    def rate(sel, f):
        v = [f(o) for o, r in zip(out, rows) if sel(r)]
        return float(np.mean(v)) if v else float("nan")
    known = lambda r: r["action"] != "UNKNOWN"
    H = lambda r: r["category"] == "H"
    safe = lambda o: o["kind"] == "確認へ"
    return {"accuracy_all": rate(lambda r: True, lambda o: o["ok"]),
            "accuracy_known": rate(known, lambda o: o["kind"] == "正解"),
            "over_clarify_known": rate(known, safe),
            "wrong_write": int(sum(o["kind"] == "誤実行" for o in out)),
            "os_recall": rate(H, safe), "os_wrong_write": int(sum(o["kind"] == "誤実行" for o, r in zip(out, rows) if H(r))),
            "os_recall_seen": rate(lambda r: H(r) and r["os_function_in_train"], safe),
            "os_recall_unseen": rate(lambda r: H(r) and not r["os_function_in_train"], safe),
            "amb_recall": rate(lambda r: r["category"] == "G", safe),
            "k_accuracy": rate(lambda r: r["category"] == "K", lambda o: o["kind"] == "正解"),
            "k_over": rate(lambda r: r["category"] == "K", safe),
            "f_accuracy": rate(lambda r: r["category"] == "F", lambda o: o["kind"] == "正解")}


def at_matched(rows, meta, p, mask) -> dict:
    """同じ手間: 行き先のある行を確認へ回す割合が MATCH_OVER に一番近い棄却の閾値（確信度の閾値は COMMON）"""
    base = decide(rows, {**meta, "thresholds": {**COMMON, "REJECT": 1.01}}, p)        # 棄却を使わないときの判定
    best = None
    for tau in FINE_GRID:
        m = p["reject"] >= tau
        m = m if mask is None else (m | mask)
        s = score(blocked(base, m), rows)
        key = (abs(s["over_clarify_known"] - MATCH_OVER), -tau)
        if best is None or key < best[0]:
            best = (key, {**s, "tau": tau})
    return best[1]


def agg(items: list[dict]) -> dict:
    keys = [k for k in items[0] if k != "tau"]
    return {k: {"mean": float(np.nanmean([x[k] for x in items])), "min": float(np.nanmin([x[k] for x in items])),
                "max": float(np.nanmax([x[k] for x in items]))} for k in keys}


COLS = [("全体の正答率", "accuracy_all", 3), ("行き先のある行の正解率", "accuracy_known", 3), ("同・確認へ回した割合", "over_clarify_known", 3),
        ("機能にない依頼を回せた割合", "os_recall", 3), ("　学習に同じ機能がある", "os_recall_seen", 3), ("　学習に同じ機能が無い", "os_recall_unseen", 3),
        ("機能にない依頼の誤実行（60 行中）", "os_wrong_write", 1), ("意図が不明を回せた割合", "amb_recall", 3),
        ("機能の語を含む依頼（K）の正解率", "k_accuracy", 3), ("　K を確認へ回した割合", "k_over", 3), ("誤実行（全体）", "wrong_write", 1)]


def cell(a: dict, key: str, nd: int) -> str:
    v = a[key]
    return f"{v['mean']:.{nd}f}（{v['min']:.{nd}f}〜{v['max']:.{nd}f}）"


def table(title: str, cols: list[tuple[str, dict]], md: list[str]) -> None:
    print(f"\n===== {title} =====")
    md += [f"\n### {title}\n", "|指標|" + "|".join(c for c, _ in cols) + "|", "|---|" + "---|" * len(cols)]
    for label, key, nd in COLS:
        cells = [cell(a, key, nd) for _, a in cols]
        print(f"{label:<24} " + "  ".join(f"[{c}] {v}" for (c, _), v in zip(cols, cells)))
        md.append(f"|{label}|" + "|".join(cells) + "|")


def judge(ns: list[int]) -> None:
    tools = {t["name"]: t["effect"] for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    ts = json.loads((WORK / "text_scores.json").read_text(encoding="utf-8")) if (WORK / "text_scores.json").exists() else None
    md, res = [], {"head_only": {}, "text": {}, "combined": {}, "overlap": {}}
    ev = {}
    for n in ns:
        if (WORK / f"device_eval_L{n}.json").exists():
            ev[n] = load_eval(n)
    if not ev:
        raise SystemExit("評価セット v2 の端末の h が無い（先に device）")
    rows0 = next(iter(ev.values()))[0]
    c = {k: sum(r["category"] == k for r in rows0) for k in NOTES}
    print(f"\n===== P7 judge: 評価セット v2（{len(rows0)} 行 = " + "・".join(f"{NOTES[k]} {v}" for k, v in c.items() if v) + "）=====")

    def head_set(n: int, src: str) -> list:
        d = HEADS[src](n)
        if not (d / "evaluate_report.json").exists():
            return []
        reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
        rows, last, mean = ev[n]
        out = []
        for seed in SEEDS:
            name = f"heads_mlp_seed{seed}.pt"
            blob, rep = load_pt(d / name), reps[name]
            meta = {"action_classes": blob["metadata"]["action_classes"], "tool_effect": tools,
                    "thresholds": {k: float(v) for k, v in rep["thresholds"].items()}}
            out.append((seed, meta, head_probs(blob, float(rep["temperature"]), last, mean)))
        return out

    # ---- A. ヘッドだけ
    sets = {(n, src): head_set(n, src) for n in ev for src in HEADS}
    for mode, f in (("validation で決めた閾値", lambda rows, m, p, mask: score(blocked(decide(rows, m, p), mask), rows)),
                    (f"同じ手間（行き先のある行の {MATCH_OVER:.0%} を確認へ）", at_matched)):
        cols = []
        for (n, src), hs in sets.items():
            if hs:
                a = agg([f(ev[n][0], m, p, None) for _, m, p in hs])
                res["head_only"][f"{n}|{src}|{mode}"] = a
                cols.append((f"層 {n}・{src}", a))
        table(f"A. ヘッドだけ・{mode}（mlp・seed 3 つの平均と最小〜最大）", cols, md)

    # ---- B. 文だけの分類器
    if ts:
        fit_ids = list(ts["labels_fit"])
        y = np.array([ts["labels_fit"][i] for i in fit_ids])
        rows = rows0
        isH = np.array([r["category"] == "H" for r in rows])
        print("\n===== B. 文だけの分類器（機能にない依頼か、それ以外か）=====")
        md += ["\n### B. 文だけの分類器（閾値は学習側の交差検証で決め、評価セットにはそのまま当てる）\n",
               "|分類器|閾値の決め方|閾値|機能にない依頼を止めた割合|学習に同じ機能がある|無い|止めてしまった行（K・N・F・G）|学習側 AUC|評価セット AUC|",
               "|---|---|---|---|---|---|---|---|---|"]
        thr = {}
        for name, m in ts["models"].items():
            cv = np.array([m["cv"][i] for i in fit_ids])
            p = np.array([m["pred"][r["id"]] for r in rows])
            neg = cv[y == 0]
            thr[name] = {"学習側で 1 件も止めない": float(neg.max()) + 1e-4, "学習側で 1% 止める": float(np.quantile(neg, 0.99, method="higher"))}
            res["text"][name] = {"cv_auc": auc(cv[y == 1], neg), "eval_auc": auc(p[isH], p[~isH]), "thresholds": {}}
            for tname, t in thr[name].items():
                mask = p >= t
                seen = np.array([r.get("os_function_in_train", False) for r in rows])
                stop = {k: int(sum(mask[i] for i, r in enumerate(rows) if r["category"] == k)) for k in ("K", "N", "F", "G")}
                x = {"threshold": t, "os_recall": float(mask[isH].mean()), "os_recall_seen": float(mask[isH & seen].mean()),
                     "os_recall_unseen": float(mask[isH & ~seen].mean()), "stopped": stop,
                     "stopped_rows": [r["utterance"] for i, r in enumerate(rows) if mask[i] and r["category"] != "H"],
                     "missed_rows": [r["utterance"] for i, r in enumerate(rows) if not mask[i] and r["category"] == "H"]}
                res["text"][name]["thresholds"][tname] = x
                st = "・".join(f"{k} {v}" for k, v in stop.items())
                print(f"{name:<8} {tname}（{t:.3f}）: 止めた {x['os_recall']:.3f}（同じ機能あり {x['os_recall_seen']:.3f}・無し {x['os_recall_unseen']:.3f}）/ "
                      f"止めてしまった {st} / AUC 学習側 {res['text'][name]['cv_auc']:.3f}・評価 {res['text'][name]['eval_auc']:.3f}")
                md.append(f"|{name}|{tname}|{t:.3f}|{x['os_recall']:.3f}|{x['os_recall_seen']:.3f}|{x['os_recall_unseen']:.3f}|{st}|"
                          f"{res['text'][name]['cv_auc']:.3f}|{res['text'][name]['eval_auc']:.3f}|")

        # ---- C. 組み合わせ（1,101 行のヘッド＋文の分類器）
        src = "1,101 行（P7）"
        for tname in ("学習側で 1 件も止めない", "学習側で 1% 止める"):
            for mode, f in (("validation で決めた閾値", lambda rows, m, p, mask: score(blocked(decide(rows, m, p), mask), rows)),
                            (f"同じ手間（{MATCH_OVER:.0%}）", at_matched)):
                cols = []
                for n in ev:
                    hs = sets[(n, src)]
                    if not hs:
                        continue
                    for label, names in (("ヘッドだけ", [None]), ("＋n-gram", ["ngram"]), ("＋BERT", [k for k in ts["models"] if k.startswith("bert")])):
                        items = []
                        for nm in names:
                            mask = None if nm is None else np.array([ts["models"][nm]["pred"][r["id"]] for r in ev[n][0]]) >= thr[nm][tname]
                            items += [f(ev[n][0], m, p, mask) for _, m, p in hs]
                        if items:
                            a = agg(items)
                            res["combined"][f"{n}|{label}|{tname}|{mode}"] = a
                            cols.append((f"層 {n} {label}", a))
                table(f"C. 組み合わせ（1,101 行のヘッド）・文の分類器の閾値「{tname}」・{mode}", cols, md)

        # ---- D. 取りこぼしの重なり（同じ手間のヘッドだけで回せなかった機能にない依頼を、BERT が止めるか）
        print("\n===== D. 取りこぼしの重なり（同じ手間のとき。機能にない依頼 60 行）=====")
        md += ["\n### D. 取りこぼしの重なり（同じ手間のとき。機能にない依頼 60 行。seed の平均）\n",
               "|層|ヘッドが回せなかった行|うち BERT（1% 止める）が止めた行|うち n-gram が止めた行|どちらも止められなかった行|", "|---|---|---|---|---|"]
        for n in ev:
            hs = sets[(n, src)]
            if not hs:
                continue
            rows = ev[n][0]
            miss, by_b, by_n, both = [], [], [], []
            for _, m, p in hs:
                s = at_matched(rows, m, p, None)
                base = decide(rows, {**m, "thresholds": {**COMMON, "REJECT": s["tau"]}}, p)
                missed = np.array([r["category"] == "H" and o["kind"] != "確認へ" for o, r in zip(base, rows)])
                bm = np.mean([np.array([ts["models"][k]["pred"][r["id"]] for r in rows]) >= thr[k]["学習側で 1% 止める"]
                              for k in ts["models"] if k.startswith("bert")], axis=0)
                nm = np.array([ts["models"]["ngram"]["pred"][r["id"]] for r in rows]) >= thr["ngram"]["学習側で 1% 止める"]
                miss.append(missed.sum()); by_b.append(float(bm[missed].sum())); by_n.append(int(nm[missed].sum()))
                both.append(float(((1 - bm) * missed).sum()))
            res["overlap"][str(n)] = {"head_missed": float(np.mean(miss)), "bert_caught": float(np.mean(by_b)),
                                      "ngram_caught": float(np.mean(by_n)), "left": float(np.mean(both))}
            o = res["overlap"][str(n)]
            print(f"層 {n}: ヘッドが回せなかった {o['head_missed']:.1f} 行 → BERT が止めた {o['bert_caught']:.1f}・n-gram が止めた {o['ngram_caught']:.1f} / "
                  f"BERT でも残る {o['left']:.1f}")
            md.append(f"|層 {n}|{o['head_missed']:.1f}|{o['bert_caught']:.1f}|{o['ngram_caught']:.1f}|{o['left']:.1f}|")
    else:
        print("\n文の分類器の結果（work/p7/text_scores.json）が無い。ヘッドだけの表を出した")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "block_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "block_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（block_compare.md・block_compare.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "heads", "text", "judge"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P7 {a.step} =====", flush=True)
    if a.step == "encode":
        encode()
    elif a.step == "device":
        device(ns)
    elif a.step == "heads":
        heads(ns, a.force)
    elif a.step == "text":
        text(a.device)
    else:
        judge(ns)


if __name__ == "__main__":
    main()
