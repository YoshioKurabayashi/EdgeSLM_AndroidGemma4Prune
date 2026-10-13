# -*- coding: utf-8 -*-
"""P13 手順 2: 大きい評価セット（challenge_v3・800 行）で、いまの構成と、重大な誤りを減らす 2 つの手当てを測る。

設計: DoC/設計書_P13_評価セットを増やし重大な誤りを減らす.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

|構成|BERT の学習データ|規則|
|---|---|---|
|いまの構成（P12）|案 C（1,101 行のうち 882 行＋対象側の文 90 行）|EdgeSLM_MultiHead_Multi の規則のまま|
|＋規則を補う|案 C|`p13_rules`（言い直し・取り消し・言いよどみ）を重ねる|
|＋BERT に機能を足す|案 E（C ＋ 機能にない依頼 108 行 = 36 機能）|そのまま|
|**＋両方（本命）**|案 E|`p13_rules`|

どの構成も、ヘッドは 933 行（P5）・引数は P9 の補った処理・棄却の閾値は validation の 2%（P10）・
BERT の断る閾値は学習側の 99.5% 点・フラグはすべての機能（P12）。**本命は測る前に決めた**（設計書 6 節）。

|段|何をするか|venv|
|---|---|---|
|`encode`|評価セット v3 をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、評価セット v3 を層 17・26・35 のグラフに流し、`h` を集める|EdgeSLM_Android|
|`text`|BERT（案 C・案 E × seed 3 つ）を学習し、学習側の交差検証と評価セット（v3・v2・v1・test）の確率を保存|EdgeSLM_MultiHead_Multi（GPU）|
|`judge`|評価セット v3 で 4 つの構成の KPI と、種類ごとの内訳を出す。numpy と PyYAML だけ|どちらでも|

    python -m app.p13_eval encode
    python -m app.p13_eval device --layers 17,26,35
    python -m app.p13_eval text
    python -m app.p13_eval judge --layers 17,26,35      # results/P13/eval_v3.md・eval_v3.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from .common import HF_DIR, PROJECT_ROOT, use_multi_library
from .p5_retrain import P3
from .p6_seeds import auc
from .p7_data import CH1, CH2, TR2, read_jsonl
from .p8_gate import BERT_LO, STEPS, run
from .p9_arguments import make_resolve
from .p10_reject import PRIMARY as REJECT_RULE, SRC, make_hook
from .p11_bert import MODES, by_mode
from .p11_data import ADD as BERT_ADD1
from .p13_data import BERT_ADD2, CH3, NOTES
from .p13_rules import make_rules

WORK = PROJECT_ROOT / "work" / "p13"
RES = PROJECT_ROOT / "results" / "P13"
P12_JSON = PROJECT_ROOT / "results" / "P12" / "block_rule.json"
T = 64
BERT_SEEDS = (0, 1, 2)
VARIANTS = [("C", "いまの BERT（P11 の案 C）", False), ("E", "機能にない依頼 36 機能を足した BERT", True)]
CONFIGS = [("いまの構成（P12）", "C", False), ("＋規則を補う", "C", True), ("＋BERT に機能を足す", "E", False), ("＋両方（本命）", "E", True)]
PRIMARY = "＋両方（本命）"
hi_rule = lambda name, neg: float(np.quantile(neg, 0.995, method="higher"))       # P12 の本命: 学習側の 99.5% 点
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("重大な誤りの割合", "severe", 4), ("不要な聞き返し", "over", 3),
       ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3), ("対象を断った（件）", "known_refused_n", 1)]


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
    rows = read_jsonl(CH3 / "utterances.jsonl")
    ids = {r["id"]: enc(r) for r in rows}
    over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
    if over:
        raise SystemExit(f"{T} トークンを超える行がある: {over}")
    ln = [len(v) for v in ids.values()]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids_eval.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"評価セット v3: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)}（グラフは {T} まで）")


def device(ns: list[int]) -> None:
    from . import p2_device as D
    D.ensure_app()
    ids = json.loads((WORK / "token_ids_eval.json").read_text(encoding="utf-8"))
    keys = list(ids)
    parts = WORK / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for n in D.graphs(ns):
        f = WORK / f"device_eval_L{n}.json"
        if f.exists() and [r["id"] for r in json.loads(f.read_text(encoding="utf-8"))["rows"]] == keys:
            print(f"層 {n}: 済み（{f.name}）", flush=True)
            continue
        got, t0 = [], time.time()
        for k, s in enumerate(range(0, len(keys), D.BATCH)):
            part = keys[s:s + D.BATCH]
            pf = parts / f"L{n}_{k:02d}.json"                       # 途中で止まっても、済んだ分は残す
            if pf.exists():
                rows = json.loads(pf.read_text(encoding="utf-8"))["rows"]
                if [r["id"] for r in rows] == part:
                    got += rows
                    continue
            res = D.run_job({**D.base_job(f"p13_eval_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
            rs = res.get("results", [])
            bad = [(i, r.get("error")) for i, r in zip(part, rs) if not r.get("ok")]
            if len(rs) != len(part) or bad:
                raise SystemExit(f"層 {n}・{k}: 通らない行 {bad[:3]}（{len(rs)}/{len(part)}）")
            rows = [{"id": i, "last": r["h_last"], "mean": r["h_mean"], "run_ms": r["run_ms"]} for i, r in zip(part, rs)]
            pf.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            got += rows
            print(f"  層 {n} 個 {k + 1}: {len(part)} 行・NPU 中央値 {np.median([r['run_ms'] for r in rs]):.1f}ms（通算 {time.time() - t0:.0f} 秒）", flush=True)
        f.write_text(json.dumps({"rows": got}), encoding="utf-8")
        print(f"層 {n}: {len(got)} 行 → {f}", flush=True)


def load_eval3(n: int):
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH3 / "utterances.jsonl") if r["id"] in got]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


# ---------------------------------------------------------------- text（GPU）

def fit_set(with_os: bool) -> tuple[list[list[dict]], dict]:
    """学習の行（組ごと）とラベル（1 = 機能にない依頼）。もとの 882 行・対象側の 90 行・（案 E だけ）機能にない依頼 108 行"""
    base = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] in ("train", "validation")]
    parts = [base, read_jsonl(BERT_ADD1 / "added.jsonl")] + ([read_jsonl(BERT_ADD2 / "added.jsonl")] if with_os else [])
    y = {r["id"]: int(r.get("add_type") == "OS") for part in parts for r in part}
    return parts, y


def text(device_name: str, seeds: tuple) -> None:
    use_multi_library()
    from edge_router.heads.crossval import make_folds
    from edge_router.l1.clear.train import fit_predict_bert
    from edge_router.l1.decide import load_config
    cfg = load_config()["model"]
    pred_rows = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] == "test"] + read_jsonl(CH3 / "utterances.jsonl") + \
        read_jsonl(CH2 / "utterances.jsonl") + read_jsonl(CH1)
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"BERT: {cfg['name']}・{cfg['epochs']} epoch・seed {[cfg['seed'] + s for s in seeds]} / 確率を出す行 {len(pred_rows)}（test・評価セット v3・v2・v1）", flush=True)
    for key, title, with_os in VARIANTS:
        parts, y = fit_set(with_os)
        fit_rows = [r for part in parts for r in part]
        fold = {}
        for part in parts:                                           # 組ごとに分ける（もとの 882 行の分け方は、どの案でも同じ）
            fold.update(make_folds(part, 5))
        folds = np.array([int(fold[r["id"]].replace("fold", "")) for r in fit_rows])
        yy = np.array([y[r["id"]] for r in fit_rows])
        texts = [r["utterance"] for r in fit_rows]
        f = WORK / f"text_scores_{key}.json"
        out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"variant": key, "title": title, "labels_fit": y, "models": {}, "bert": cfg}
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
            cv = np.zeros(len(fit_rows))
            for k in range(5):
                tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
                cv[te] = fit_predict_bert([texts[i] for i in tr], yy[tr], [texts[i] for i in te], **kw)
                print(f"  {name}: 交差検証 {k + 1}/5", flush=True)
            pred = fit_predict_bert(texts, yy, [r["utterance"] for r in pred_rows], **kw)
            out["models"][name] = {"cv": {r["id"]: float(p) for r, p in zip(fit_rows, cv)},
                                   "pred": {r["id"]: float(p) for r, p in zip(pred_rows, pred)}}
            f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            neg = cv[yy == 0]
            print(f"{name}: 学習側の交差検証 AUC {auc(cv[yy == 1], neg):.3f} / それ以外の行でフラグ（{BERT_LO} 以上）{float((neg >= BERT_LO).mean()):.3f}・"
                  f"99.5% 点 {np.quantile(neg, 0.995, method='higher'):.3f} / {time.time() - t0:.0f} 秒", flush=True)
        print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge（numpy だけ）

def interval(p: float, n: int) -> str:
    """割合 p・n 行のときの 95% 信頼区間の目安（Wilson）"""
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{max(0.0, c - h):.3f}〜{min(1.0, c + h):.3f}"


def cell(v: dict, nd: int = 3) -> str:
    return f"{v['mean']:.{nd}f}" if v["min"] == v["max"] else f"{v['mean']:.{nd}f}（{v['min']:.{nd}f}〜{v['max']:.{nd}f}）"


def text_side(ts: dict, ch3: list[dict], base_ids: dict) -> dict:
    fit_ids = list(ts["labels_fit"])
    y = np.array([ts["labels_fit"][i] for i in fit_ids])
    known = [r["id"] for r in ch3 if r["action"] != "UNKNOWN"]
    out = collections_list()
    for name in sorted(ts["models"]):
        m = ts["models"][name]
        cv = np.array([m["cv"][i] for i in fit_ids])
        hi = hi_rule(name, cv[y == 0])
        out["hi"].append(hi)
        out["cv_flag_known"].append(float(np.mean([m["cv"][i] >= BERT_LO for i in base_ids["known"]])))
        out["cv_os_flag"].append(float(np.mean([m["cv"][i] >= BERT_LO for i in base_ids["os"]])))
        out["v3_flag_known"].append(float(np.mean([m["pred"][i] >= BERT_LO for i in known])))
        out["v3_refuse_known"].append(int(sum(m["pred"][i] >= hi for i in known)))
        for g in ("seen", "new_seen", "unseen"):
            ids = [r["id"] for r in ch3 if r["category"] == "H" and r.get("os_group") == g]
            out[f"v3_os_flag_{g}"].append(float(np.mean([m["pred"][i] >= BERT_LO for i in ids])))
            out[f"v3_os_refuse_{g}"].append(float(np.mean([m["pred"][i] >= hi for i in ids])))
    return out


def collections_list() -> dict:
    import collections
    return collections.defaultdict(list)


def agg(xs: list) -> dict:
    return {"mean": float(np.mean(xs)), "min": float(min(xs)), "max": float(max(xs))}


def judge(ns: list[int]) -> None:
    tss = {}
    for key, _, _ in VARIANTS:
        f = WORK / f"text_scores_{key}.json"
        if f.exists():
            ts = json.loads(f.read_text(encoding="utf-8"))
            if ts["models"]:
                tss[key] = by_mode(ts, MODES[0])
    if "C" not in tss:
        raise SystemExit("BERT の結果が無い（先に text）")
    ch3 = read_jsonl(CH3 / "utterances.jsonl")
    final = STEPS[-1]
    base = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] in ("train", "validation")]
    base_ids = {"known": [r["id"] for r in base if r["action"] != "UNKNOWN"], "os": [r["id"] for r in base if r.get("add_type") == "OS"]}
    n_rows = len(ch3)
    c = {k: sum(r["category"] == k for r in ch3) for k in NOTES}
    users = sum(r.get("author") == "user" for r in ch3)
    print(f"\n===== P13 judge: 評価セット v3（{n_rows} 行 = " + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。うち利用者が足した行 {users}）=====")
    print(f"ヘッド {SRC}・引数は補った処理・棄却の閾値は {REJECT_RULE}・BERT の断る閾値は学習側の 99.5% 点・フラグはすべての機能")
    print(f"あらかじめ決めた本命: {PRIMARY}")
    md = [f"評価セット v3: {n_rows} 行（" + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。利用者が足した行 {users}）。"
          f"数字はヘッドの seed 3 つ × BERT の seed の平均（かっこは最小〜最大）。信頼区間は平均の割合と行数から出した目安（Wilson・95%）\n"]
    res = {"rows": c, "users": users, "text": {}, "kpi": {}, "detail": {}}

    # ---- 1. BERT の側（文だけ）
    print("\n===== 1. BERT の側（確率だけで数えたもの。フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上）=====")
    labels = [("断る閾値（seed ごと）", "hi"), ("学習側: 対象にフラグ", "cv_flag_known"), ("学習側: 機能にない依頼にフラグ以上", "cv_os_flag"),
              ("v3: 対象 " + str(sum(r["action"] != "UNKNOWN" for r in ch3)) + " 行にフラグ", "v3_flag_known"), ("v3: 対象を断った（行。seed ごと）", "v3_refuse_known"),
              ("v3: 機能にない依頼 seen にフラグ以上", "v3_os_flag_seen"), ("　断る閾値以上", "v3_os_refuse_seen"),
              ("v3: new_seen にフラグ以上", "v3_os_flag_new_seen"), ("　断る閾値以上", "v3_os_refuse_new_seen"),
              ("v3: unseen にフラグ以上", "v3_os_flag_unseen"), ("　断る閾値以上", "v3_os_refuse_unseen")]
    tsd = {k: text_side(tss[k], ch3, base_ids) for k in tss}
    md += ["\n### 1. BERT の側（確率だけで数えたもの。フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上）\n",
           "|指標|" + "|".join(f"案 {k}" for k in tss) + "|", "|---|" + "---|" * len(tss)]
    for lab, key in labels:
        cells = ["・".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in tsd[k][key]) if key in ("hi", "v3_refuse_known") else cell(agg(tsd[k][key]))
                 for k in tss]
        print(f"{lab:<28} " + "  ".join(f"[案 {k}] {x}" for k, x in zip(tss, cells)))
        md.append(f"|{lab}|" + "|".join(cells) + "|")
    res["text"] = {k: dict(v) for k, v in tsd.items()}

    # ---- 2. KPI（評価セット v3）
    def kpi_of(bert: str, with_rules: bool, loader):
        blob = run(ns, make_resolve=make_resolve, out=None, heads_hook=make_hook("val", 0.02, {}), steps=[final], sources=[SRC], quiet=True,
                   ts=tss[bert], watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=make_rules if with_rules else None)
        return {n: (blob["kpi"][f"{n}|{SRC}|{final[0]}"], blob["detail"][f"{n}|{SRC}|{final[0]}"]) for n in ns if f"{n}|{SRC}|{final[0]}" in blob["kpi"]}

    configs = [x for x in CONFIGS if x[1] in tss]
    out = {name: kpi_of(bert, wr, load_eval3) for name, bert, wr in configs}
    if not out[configs[0][0]]:
        raise SystemExit("評価セット v3 の端末の h が無い（先に device）")
    for n in ns:
        if n not in out[configs[0][0]]:
            continue
        print(f"\n===== 2. 評価セット v3 の KPI・層 {n} 個 =====")
        heads = [lab for lab, _, _ in KPI] + ["EtoE の成功の区間", "重大な誤りの割合の区間"]
        md += [f"\n### 2. 評価セット v3 の KPI・層 {n} 個\n", "|構成|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
        for name, _, _ in configs:
            k, d = out[name][n]
            cells = [cell(k[key], nd) for _, key, nd in KPI] + [interval(k["e2e"]["mean"], n_rows), interval(k["severe"]["mean"], n_rows)]
            res["kpi"][f"{n}|{name}"] = {**{key: k[key] for _, key, _ in KPI}, "e2e_ci": cells[-2], "severe_ci": cells[-1]}
            res["detail"][f"{n}|{name}"] = d
            mark = "**" if name == PRIMARY else ""
            print(f"{name:<16} " + "  ".join(f"{h}: {x}" for h, x in zip(heads, cells)))
            md.append(f"|{mark}{name}{mark}|" + "|".join(cells) + "|")
        # 種類ごとの内訳（1 回あたりの行数 → 割合）
        md += [f"\n種類ごとの内訳（層 {n} 個。割合。正解 = 機能にない依頼は断った・意図が不明は聞き返した・それ以外は正しく処理）:\n",
               "|種類|行|" + "|".join(f"{name}: 正解／回復可能／通常／重大" for name, _, _ in configs) + "|", "|---|---|" + "---|" * len(configs)]
        groups = [("H", "seen"), ("H", "new_seen"), ("H", "unseen")] + [("G", k) for k in ("途中で切れる", "言い直しと取り消し", "指示語", "曖昧")] + \
                 [("K", None), ("N", None), ("F", None)]
        for cat, grp in groups:
            sel = [r for r in ch3 if r["category"] == cat and (grp is None or (r.get("os_group") or r.get("g_kind")) == grp)]
            if not sel:
                continue
            cells = []
            for name, _, _ in configs:
                bg = out[name][n][1]["by_group"]
                tot = {g: sum(x["rows"] for x in bg if x["category"] == cat and (grp is None or x["group"] == grp) and x["grade"] == g)
                       for g in ("正解", "回復可能", "通常", "重大")}
                cells.append("／".join(f"{tot[g] / len(sel):.3f}" for g in ("正解", "回復可能", "通常", "重大")))
            label = f"{NOTES[cat]}" + (f"・{grp}" if grp else "")
            print(f"  {label:<24} {len(sel):>3} 行 " + "  ".join(f"[{name[:8]}] {x}" for (name, _, _), x in zip(configs, cells)))
            md.append(f"|{label}|{len(sel)}|" + "|".join(cells) + "|")
        for name in (configs[0][0], configs[-1][0]):
            d = out[name][n][1]
            md += [f"\n{name}のとき（層 {n} 個）:\n", "|見るもの|行（出た割合）|", "|---|---|",
                   "|重大な誤り|" + " / ".join(f"[{x['category']}]{x['utterance']} → {x['executed']}（{x['share']:.2f}）" for x in d["severe_rows"][:30]) + "|",
                   "|通常の誤り|" + " / ".join(f"[{x['category']}]{x['utterance']} → {x['outcome']}（{x['share']:.2f}）" for x in d["normal_rows"][:30]) + "|",
                   "|BERT のフラグで聞き返した対象|" + " / ".join(f"{x['utterance']}（{x['tool']}・{x['share']:.2f}）" for x in d["flag_rows"][:20]) + "|"]
            print(f"  {name}・重大な誤り: " + " / ".join(f"[{x['category']}]{x['utterance']}→{x['executed']}（{x['share']:.2f}）" for x in d["severe_rows"][:14]))

    # ---- 3. 確かめ: 評価セット v2 で、学習し直した案 C が P12 の本命とどのくらい違うか（BERT は学習し直すたびに少し変わる）
    v2 = kpi_of("C", False, None)
    if v2 and P12_JSON.exists():
        p12 = json.loads(P12_JSON.read_text(encoding="utf-8"))["kpi"]
        md += ["\n### 3. 確かめ: 評価セット v2（200 行）・いまの構成。P12 の数字 → 学習し直した BERT\n", "|層|EtoE の成功|重大な誤り（件）|不要な聞き返し|通常の誤り|", "|---|---|---|---|---|"]
        print("\n===== 3. 確かめ: 評価セット v2・いまの構成（P12 の数字 → 学習し直した BERT）=====")
        for n in v2:
            ref, k = p12.get(f"C|{n}|99.5% 点|すべての機能"), v2[n][0]
            if ref:
                row = [f"{ref[key]['mean']:.{nd}f} → {k[key]['mean']:.{nd}f}" for key, nd in (("e2e", 3), ("severe_n", 1), ("over", 3), ("normal", 3))]
                res.setdefault("v2_check", {})[str(n)] = {key: [ref[key]["mean"], k[key]["mean"]] for key in ("e2e", "severe_n", "over", "normal")}
                print(f"層 {n}: EtoE {row[0]} / 重大 {row[1]} / 不要な聞き返し {row[2]} / 通常 {row[3]}")
                md.append(f"|層 {n}|" + "|".join(row) + "|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "eval_v3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "eval_v3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（eval_v3.md・eval_v3.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "text", "judge"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in BERT_SEEDS))
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P13 {a.step} =====", flush=True)
    if a.step == "encode":
        encode()
    elif a.step == "device":
        device(ns)
    elif a.step == "text":
        text(a.device, tuple(int(x) for x in a.seeds.split(",") if x))
    else:
        judge(ns)


if __name__ == "__main__":
    main()
