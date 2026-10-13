# -*- coding: utf-8 -*-
"""P11 手順 2: 文の分類器（BERT）の学習データを 4 通りに変えて学習し、フラグの立ち方と KPI を比べる。

設計: DoC/設計書_P11_BERTのフラグが対象の依頼に立つ問題.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

|案|BERT の学習データ|
|---|---|
|A|いまのまま（P7。1,101 行のうち train・validation の 882 行。機能にない依頼 = 1）|
|B|シートヒーター・ハンドルヒーターの 6 行を、対象側（0）へ移す|
|C|対象側の文 90 行（`p11_data`）を足す。ラベルは A と同じ|
|D|B と C の両方|

**あらかじめ決めた本命は C**（いまのツールの定義にはシートヒーターの引数が無いので、ラベルは「機能にない依頼」のままが定義と合う）。
B・D は感度を見るために並べる。評価セットの結果を見て本命を選び直さない。

ヘッドは変えない（933 行・P5）。流れは P10 の本命（引数は P9 の補った処理・棄却の閾値は validation の 2%）。
BERT は seed を変えて 3 つ学習し、「1 つずつ使う（実機に載せる形）」と「3 つの確率を平均する（seed の揺れの大きさを見る参考）」を並べる。

|段|何をするか|venv|
|---|---|---|
|`text`|4 案 × seed 3 つの BERT を学習し、学習側の交差検証・評価セットの確率を保存（A の seed 0・1 は P7 の結果を写す）|EdgeSLM_MultiHead_Multi（GPU）|
|`judge`|フラグの立ち方（学習側の交差検証・test・評価セット v1・v2）と、評価セット v2 の KPI を並べる。numpy と PyYAML だけ|どちらでも|

    python -m app.p11_bert text
    python -m app.p11_bert judge --layers 17,26,35      # results/P11/bert_variants.md・bert_variants.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .common import PROJECT_ROOT, use_multi_library
from .p6_seeds import auc
from .p7_block import WORK as P7
from .p7_data import CH1, CH2, TR2, read_jsonl
from .p8_gate import BERT_LO, STEPS, run
from .p9_arguments import make_resolve
from .p10_reject import PRIMARY as REJECT_RULE, SRC, make_hook
from .p11_data import ADD, RELABEL

WORK = PROJECT_ROOT / "work" / "p11"
RES = PROJECT_ROOT / "results" / "P11"
P10_JSON = PROJECT_ROOT / "results" / "P10" / "reject_rule.json"
VARIANTS = [("A", "いまのまま", dict(relabel=False, add=False)),
            ("B", "ヒーター 6 行を対象側へ", dict(relabel=True, add=False)),
            ("C", "対象側の文を足す", dict(relabel=False, add=True)),
            ("D", "B＋C", dict(relabel=True, add=True))]
PRIMARY = "C"
BERT_SEEDS = (0, 1, 2)
WRITE = ("TOOL_control_hvac", "TOOL_control_media", "TOOL_set_navigation")
MODES = ("1 つずつ", "seed の平均")


def fit_set(v: dict) -> tuple[list[dict], list[dict], dict]:
    """案ごとの学習の行とラベル（1 = 機能にない依頼）。もとの 882 行（base）と、足す行（extra）を分けて返す"""
    base = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] in ("train", "validation")]
    extra = read_jsonl(ADD / "added.jsonl") if v["add"] else []
    y = {r["id"]: int(r.get("add_type") == "OS" and not (v["relabel"] and r.get("os_function") in RELABEL)) for r in base}
    y.update({r["id"]: 0 for r in extra})
    return base, extra, y


# ---------------------------------------------------------------- text（GPU）

def text(device_name: str, seeds: tuple) -> None:
    use_multi_library()
    from edge_router.heads.crossval import make_folds
    from edge_router.l1.clear.train import fit_predict_bert
    from edge_router.l1.decide import load_config
    cfg = load_config()["model"]
    rows = read_jsonl(TR2 / "utterances.jsonl")
    pred_rows = [r for r in rows if r["split"] == "test"] + read_jsonl(CH2 / "utterances.jsonl") + read_jsonl(CH1)
    p7 = json.loads((P7 / "text_scores.json").read_text(encoding="utf-8")) if (P7 / "text_scores.json").exists() else None
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"BERT: {cfg['name']}・{cfg['epochs']} epoch・seed {[cfg['seed'] + s for s in seeds]} / 確率を出す行 {len(pred_rows)}（test・評価セット v2・v1）", flush=True)
    for key, title, v in VARIANTS:
        base, extra, y = fit_set(v)
        fit_rows = base + extra
        fold = make_folds(base, 5)                                 # もとの 882 行の分け方は、どの案でも同じにする
        if extra:
            fold = {**fold, **make_folds(extra, 5)}
        folds = np.array([int(fold[r["id"]].replace("fold", "")) for r in fit_rows])
        yy = np.array([y[r["id"]] for r in fit_rows])
        texts = [r["utterance"] for r in fit_rows]
        f = WORK / f"text_scores_{key}.json"
        out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"variant": key, "title": title, "labels_fit": y, "models": {},
                                                                              "bert": cfg, "n_base": len(base)}
        if out["labels_fit"] != y:
            raise SystemExit(f"案 {key}: 保存済みの結果とラベルが違う（データを変えたなら {f.name} を別の名前にしてから実行）")
        print(f"\n##### 案 {key}（{title}）: 学習 {len(fit_rows)} 行（機能にない依頼 {int(yy.sum())}・それ以外 {int((1 - yy).sum())}）#####", flush=True)
        if key == "A" and p7 and p7["labels_fit"] == y:
            for s in seeds:
                name = f"bert_s{s}"
                if name in p7["models"] and name not in out["models"]:
                    out["models"][name] = p7["models"][name]
                    print(f"{name}: P7 の結果を写した", flush=True)
            f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        for s in seeds:
            name = f"bert_s{s}"
            if name in out["models"]:
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
                  f"99% 点 {np.quantile(neg, 0.99):.3f} / {time.time() - t0:.0f} 秒", flush=True)
        print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge（numpy だけ）

def load_variants() -> dict:
    out = {}
    for key, title, v in VARIANTS:
        f = WORK / f"text_scores_{key}.json"
        if f.exists():
            ts = json.loads(f.read_text(encoding="utf-8"))
            if ts["models"]:
                out[key] = ts
    if "A" not in out and (P7 / "text_scores.json").exists():           # text をまだ実行していないとき: P7 の結果だけで A を出す
        p7 = json.loads((P7 / "text_scores.json").read_text(encoding="utf-8"))
        out["A"] = {"labels_fit": p7["labels_fit"], "models": {k: m for k, m in p7["models"].items() if k.startswith("bert_s")}}
    return out


def by_mode(ts: dict, mode: str, only: tuple | None = None) -> dict:
    names = sorted(k for k in ts["models"] if k.startswith("bert_s") and (only is None or k in only))
    if mode == MODES[0]:
        return {"labels_fit": ts["labels_fit"], "models": {k: ts["models"][k] for k in names}}
    mean = {part: {i: float(np.mean([ts["models"][k][part][i] for k in names])) for i in ts["models"][names[0]][part]} for part in ("cv", "pred")}
    return {"labels_fit": ts["labels_fit"], "models": {"bert_mean": mean}}


def groups() -> dict:
    """フラグの立ち方を数える行の組。学習側は、どの案でも同じ「もとの 882 行」で数える"""
    rows = read_jsonl(TR2 / "utterances.jsonl")
    base = [r for r in rows if r["split"] in ("train", "validation")]
    test = [r for r in rows if r["split"] == "test"]
    ch1, ch2 = read_jsonl(CH1), read_jsonl(CH2 / "utterances.jsonl")
    ids = lambda rs, f: [r["id"] for r in rs if f(r)]
    known = lambda r: r["action"] not in ("UNKNOWN", "MULTI")
    write = lambda r: r["action"] in WRITE
    heat = lambda r: r.get("os_function") in RELABEL
    os_tr = lambda r: r.get("add_type") == "OS" and not heat(r)
    return {"cv": {"write": ids(base, write), "hvac": ids(base, lambda r: r["action"] == WRITE[0]), "media": ids(base, lambda r: r["action"] == WRITE[1]),
                   "nav": ids(base, lambda r: r["action"] == WRITE[2]), "known": ids(base, known), "os": ids(base, os_tr), "heat": ids(base, heat),
                   "neg": ids(base, lambda r: r.get("add_type") != "OS")},
            "test": {"write": ids(test, write), "known": ids(test, known), "os": ids(test, os_tr)},
            "ch1": {"write": ids(ch1, write), "known": ids(ch1, known), "os": ids(ch1, lambda r: r["category"] == "H")},
            "ch2": {"write": ids(ch2, write), "known": ids(ch2, known), "os": ids(ch2, lambda r: r["category"] == "H" and not heat(r)),
                    "heat": ids(ch2, lambda r: r["category"] == "H" and heat(r))}}


def text_stats(ts: dict, g: dict) -> dict:
    fit_ids = list(ts["labels_fit"])
    y = np.array([ts["labels_fit"][i] for i in fit_ids])
    items = []
    for m in ts["models"].values():
        hi = float(np.quantile(np.array([m["cv"][i] for i in fit_ids])[y == 0], 0.99, method="higher"))     # p8_gate と同じ決め方
        def rate(part, ids, thr):
            return float(np.mean(np.array([m[part][i] for i in ids]) >= thr)) if ids else float("nan")
        x = {"hi": hi, "cv_auc": auc(np.array([m["cv"][i] for i in g["cv"]["os"]]), np.array([m["cv"][i] for i in g["cv"]["neg"]]))}
        for k in ("write", "hvac", "media", "nav", "known"):
            x[f"cv_flag_{k}"] = rate("cv", g["cv"][k], BERT_LO)
        x.update({"cv_os_flag": rate("cv", g["cv"]["os"], BERT_LO), "cv_os_block": rate("cv", g["cv"]["os"], hi),
                  "cv_heat_flag": rate("cv", g["cv"]["heat"], BERT_LO)})
        for s in ("test", "ch1", "ch2"):
            x.update({f"{s}_flag_write": rate("pred", g[s]["write"], BERT_LO), f"{s}_flag_known": rate("pred", g[s]["known"], BERT_LO),
                      f"{s}_block_known": rate("pred", g[s]["known"], hi),
                      f"{s}_os_flag": rate("pred", g[s]["os"], BERT_LO), f"{s}_os_block": rate("pred", g[s]["os"], hi)})
        x["ch2_heat_flag"] = rate("pred", g["ch2"]["heat"], BERT_LO)
        items.append(x)
    return {k: {"mean": float(np.mean([x[k] for x in items])), "min": float(min(x[k] for x in items)), "max": float(max(x[k] for x in items))}
            for k in items[0]}


T1 = [("断る閾値（学習側で 1% 止める値）", "hi"), ("対象の操作系にフラグが立つ割合", "cv_flag_write"), ("　空調", "cv_flag_hvac"), ("　メディア", "cv_flag_media"),
      ("　ナビ", "cv_flag_nav"), ("対象の依頼・雑談すべてにフラグが立つ割合", "cv_flag_known"), ("機能にない依頼にフラグ以上が出る割合", "cv_os_flag"),
      ("　断る閾値以上", "cv_os_block"), ("シートヒーター・ハンドルヒーター 6 行にフラグ以上", "cv_heat_flag"), ("AUC（機能にない依頼 ↔ それ以外）", "cv_auc")]
T2 = [("test: 対象の操作系にフラグ", "test_flag_write"), ("test: 機能にない依頼にフラグ以上", "test_os_flag"),
      ("評価セット v1: 対象の操作系にフラグ", "ch1_flag_write"), ("評価セット v1: 対象すべてにフラグ", "ch1_flag_known"),
      ("評価セット v1: 機能にない依頼にフラグ以上", "ch1_os_flag"), ("評価セット v1: 同・断る閾値以上", "ch1_os_block"),
      ("評価セット v2: 対象の操作系にフラグ", "ch2_flag_write"), ("評価セット v2: 対象すべてにフラグ", "ch2_flag_known"),
      ("評価セット v2: 対象を断る閾値以上", "ch2_block_known"),
      ("評価セット v2: 機能にない依頼にフラグ以上", "ch2_os_flag"), ("評価セット v2: 同・断る閾値以上", "ch2_os_block"),
      ("評価セット v2: シートヒーター・ハンドルヒーター 4 行にフラグ以上", "ch2_heat_flag")]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("1 回目で正しく処理", "first", 3),
       ("機能にない依頼を断った", "os_refuse", 3), ("同・聞き返した", "os_clarify", 3), ("同・操作として実行（件）", "os_severe_n", 1),
       ("対象を断った（件）", "known_refused_n", 1)]
FLAG_STAGE = "対象の依頼・雑談|聞き返し（BERT のフラグ）"


def cell(v: dict, nd: int = 3) -> str:
    return f"{v['mean']:.{nd}f}" if v["min"] == v["max"] else f"{v['mean']:.{nd}f}（{v['min']:.{nd}f}〜{v['max']:.{nd}f}）"


def judge(ns: list[int]) -> None:
    tss = load_variants()
    if not tss:
        raise SystemExit("BERT の結果が無い（work/p7/text_scores.json か、先に text）")
    g = groups()
    ch2 = read_jsonl(CH2 / "utterances.jsonl")
    watch = set(g["ch2"]["heat"])
    final = STEPS[-1]
    title = {k: t for k, t, _ in VARIANTS}
    cols = [(k, mode) for k, _, _ in VARIANTS if k in tss for mode in MODES]
    name = lambda k, mode: f"{k} {title[k]}・{mode}"
    seeds = {k: sorted(m for m in tss[k]["models"] if m.startswith("bert_s")) for k in tss}
    print(f"\n===== P11 judge: BERT の学習データ 4 案（ヘッド {SRC}・引数は補った処理・棄却の閾値は {REJECT_RULE}）=====")
    print("あらかじめ決めた本命: 案 " + PRIMARY + f"（{title[PRIMARY]}）・1 つずつ")
    print("ある結果: " + " / ".join(f"案 {k}: {len(seeds[k])} seed" for k in tss) + ("" if len(tss) == len(VARIANTS) else "（無い案は text を実行すると出る）"))
    n_of = {s: {k: len(v) for k, v in d.items()} for s, d in g.items()}
    md = [f"数える行: 学習側 = もとの 882 行（対象の操作系 {n_of['cv']['write']}・対象の依頼と雑談 {n_of['cv']['known']}・機能にない依頼 {n_of['cv']['os']}）/ "
          f"test（操作系 {n_of['test']['write']}・機能にない依頼 {n_of['test']['os']}）/ 評価セット v1（操作系 {n_of['ch1']['write']}・対象 {n_of['ch1']['known']}・"
          f"機能にない依頼 {n_of['ch1']['os']}）/ 評価セット v2（操作系 {n_of['ch2']['write']}・対象 {n_of['ch2']['known']}・機能にない依頼 {n_of['ch2']['os']}・"
          f"ヒーター {n_of['ch2']['heat']}）。フラグ = 確率 {BERT_LO} 以上。機能にない依頼は、どの案でもシートヒーター・ハンドルヒーターを除いて数える",
          "seed: " + "・".join(f"案 {k} = {len(seeds[k])} つ" for k in tss) + "。「1 つずつ」は seed ごとの平均（最小〜最大）、「seed の平均」は確率を平均した 1 つ\n"]
    print(md[0])
    res = {"text": {}, "kpi": {}, "detail": {}, "seeds": seeds, "rows": n_of}

    # ---- 1・2. フラグの立ち方
    stats = {(k, mode): text_stats(by_mode(tss[k], mode), g) for k, mode in cols}
    for (k, mode), s in stats.items():
        res["text"][f"{k}|{mode}"] = s
    for head, table in (("1. 学習側の交差検証（学習に使っていない fold の確率。もとの 882 行で数える）", T1),
                        ("2. 学習に使っていない行（test・評価セット v1・v2）", T2)):
        print(f"\n===== {head} =====")
        md += [f"\n### {head}\n", "|指標|" + "|".join(name(k, m) for k, m in cols) + "|", "|---|" + "---|" * len(cols)]
        for lab, key in table:
            cells = [cell(stats[c][key]) for c in cols]
            print(f"{lab:<30} " + "  ".join(f"[{k}・{m[:4]}] {x}" for (k, m), x in zip(cols, cells)))
            md.append(f"|{lab}|" + "|".join(cells) + "|")

    # ---- 3. KPI（評価セット v2。P10 の本命の流れ）
    def kpi_of(ts: dict) -> dict:
        blob = run(ns, make_resolve=make_resolve, out=None, heads_hook=make_hook("val", 0.02, {}), steps=[final], sources=[SRC], quiet=True,
                   ts=ts, watch=watch)
        return {n: (blob["kpi"][f"{n}|{SRC}|{final[0]}"], blob["detail"][f"{n}|{SRC}|{final[0]}"]) for n in ns if f"{n}|{SRC}|{final[0]}" in blob["kpi"]}

    # 確かめ: A の seed 0・1 だけで測ると、P10 の本命の数字と同じになるか
    if P10_JSON.exists() and {"bert_s0", "bert_s1"} <= set(seeds["A"]):
        p10 = json.loads(P10_JSON.read_text(encoding="utf-8"))
        chk = kpi_of(by_mode(tss["A"], MODES[0], ("bert_s0", "bert_s1")))
        diff = [f"層 {n}・{k_}" for n in chk for _, k_, _ in KPI if abs(chk[n][0][k_]["mean"] - p10[f"{n}|{REJECT_RULE}"]["kpi"][k_]["mean"]) > 1e-9]
        line = "確かめ（案 A・seed 0 と 1 だけ ↔ P10 の本命）: " + ("同じ数字" if not diff else f"違う {diff}")
        print("\n" + line)
        md.append("\n" + line)
        res["check_p10"] = not diff
        if diff:
            raise SystemExit("P10 と数字が合わない。流れのどこかが変わっている")
    out = {c: kpi_of(by_mode(tss[c[0]], c[1])) for c in cols}
    for n in ns:
        if n not in out[cols[0]]:
            continue
        print(f"\n===== 3. 評価セット v2 の KPI・層 {n} 個 =====")
        heads_ = [lab for lab, _, _ in KPI] + ["対象を BERT のフラグで聞き返した（行）"]
        md += [f"\n### 3. 評価セット v2 の KPI・層 {n} 個（ヘッドの seed 3 つ × BERT の平均。かっこは最小〜最大）\n",
               "|BERT の学習データ|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
        for c in cols:
            k, d = out[c][n]
            cells = [cell(k[key], nd) for _, key, nd in KPI] + [f"{d['stages'].get(FLAG_STAGE, 0.0):.1f}"]
            res["kpi"][f"{n}|{c[0]}|{c[1]}"] = {**{key: k[key] for _, key, _ in KPI}, "flag_known_rows": d["stages"].get(FLAG_STAGE, 0.0)}
            res["detail"][f"{n}|{c[0]}|{c[1]}"] = d
            mark = "**" if (c[0] == PRIMARY and c[1] == MODES[0]) else ""
            print(f"{name(*c):<28} " + "  ".join(f"{h}: {x}" for h, x in zip(heads_, cells)))
            md.append(f"|{mark}{name(*c)}{mark}|" + "|".join(cells) + "|")
        md += ["\nBERT のフラグで聞き返した対象の行（1 つずつ。出た割合）:\n", "|案|発話（実行しようとしたツール・割合）|", "|---|---|"]
        for k_ in tss:
            fr = out[(k_, MODES[0])][n][1]["flag_rows"]
            md.append(f"|{k_}|" + " / ".join(f"{x['utterance']}（{x['tool']}・{x['share']:.2f}）" for x in fr[:14]) + "|")
            print(f"  案 {k_} のフラグで聞き返した対象の行: " + " / ".join(f"{x['utterance']}（{x['share']:.2f}）" for x in fr[:10]))
        md += [f"\nシートヒーター・ハンドルヒーターの {len(watch)} 行の行き先（1 つずつ。行数）:\n", "|案|" + "|".join(r["utterance"] for r in ch2 if r["id"] in watch) + "|",
               "|---|" + "---|" * len(watch)]
        for k_ in tss:
            w = out[(k_, MODES[0])][n][1]["watch"]
            cells = [" / ".join(f"{x['outcome']} {x['share']:.2f}" for x in w if x["utterance"] == r["utterance"]) for r in ch2 if r["id"] in watch]
            md.append(f"|{k_}|" + "|".join(cells) + "|")
            print(f"  案 {k_} のヒーター {len(watch)} 行: " + " ｜ ".join(cells))
        md += ["\n残った重大な誤り（1 つずつ。出た割合）:\n", "|案|発話 → 実行したツール（割合）|", "|---|---|"]
        for k_ in tss:
            sr = out[(k_, MODES[0])][n][1]["severe_rows"]
            md.append(f"|{k_}|" + " / ".join(f"{x['utterance']} → {x['executed']}（{x['share']:.2f}）" for x in sr[:12]) + "|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "bert_variants.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "bert_variants.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（bert_variants.md・bert_variants.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["text", "judge"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in BERT_SEEDS))
    a = ap.parse_args()
    print(f"\n===== P11 {a.step} =====", flush=True)
    if a.step == "text":
        text(a.device, tuple(int(x) for x in a.seeds.split(",") if x))
    else:
        judge([int(x) for x in a.layers.replace(" ", ",").split(",") if x])


if __name__ == "__main__":
    main()
