# -*- coding: utf-8 -*-
"""P20 手順 2: P19 で決めた構成の合否を、新しい評価セット（challenge_v5・600 行）で測る。

設計: DoC/設計書_P20_決めた構成の合否.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

**本命（測る前に決めた）= P19 で決めた構成・層 26・端末に載せる BERT 1 つ。**
ヘッドは学習し直さない（P18 の 1,584 行）。BERT は案 F を同じ手順で学習し直す（P18 では確率だけを保存し、モデルを保存していなかったため。
評価セット v5 の確率を出すのと、端末に載せるためにモデルを保存するのを兼ねる）。学習し直した確率が P18 のときと同じかを確かめる。

|段|何をするか|venv|
|---|---|---|
|`encode`|評価セット v5 をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、評価セット v5 を層 17・26・35 のグラフに流し、`h` を集める（途中で止めても済んだ分は残る）|EdgeSLM_Android|
|`text`|BERT（案 F × seed 3 つ）を全体で学習し、評価セット v5 などの確率を保存。モデルも保存する（`work/p20/bert_F/`）。交差検証は P18 の結果を使う|EdgeSLM_MultiHead_Multi（GPU）|
|`judge`|評価セット v5 で、本命と比べる相手の KPI・合否・内訳・部品ごとの精度を出す。numpy と PyYAML だけ|どちらでも|

    python -m app.p20_verdict encode
    python -m app.p20_verdict device --layers 17,26,35
    python -m app.p20_verdict text
    python -m app.p20_verdict judge --layers 17,26,35      # results/P20/verdict.md・verdict.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p5_retrain import P3
from .p7_data import CH1, CH2, TR2, read_jsonl
from .p8_gate import BERT_LO
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import cell, interval
from .p16_parts import J, ORDER, ms
from .p17_rules import make_rules as rules_p17
from .p18_data import CH4
from .p18_retrain import SRC1584, WORK as P18, bert_part, cached, fit_set, head_part, verdict
from .p19_config import bert_check, measure
from .p19_rules import make_rules as rules_p19
from .p20_data import CH5, NOTES

WORK = PROJECT_ROOT / "work" / "p20"
RES = PROJECT_ROOT / "results" / "P20"
T = 64
BERT_SEEDS = (0, 1, 2)
PRIMARY_LAYER = 26
# （名前, BERT の seed の選び方, 棄却の閾値, 規則）。"one" = 端末に載せる 1 つ・"ok" = 決まりで残った seed・"all" = 3 つとも
CONFIGS = [("決めた構成・BERT 1 つ（本命）", "one", 0.02, rules_p19),
           ("決めた構成・残った seed の平均", "ok", 0.02, rules_p19),
           ("比べる相手: S5 なし", "ok", 0.02, rules_p17),
           ("比べる相手: P18 の本命（S5 なし・BERT は seed 3 つ）", "all", 0.02, rules_p17)]
PRIMARY = CONFIGS[0][0]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("重大な誤りの割合", "severe", 4), ("不要な聞き返し", "over", 3),
       ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3), ("対象を断った（件）", "known_refused_n", 1)]
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
    rows = read_jsonl(CH5 / "utterances.jsonl")
    ids = {r["id"]: enc(r) for r in rows}
    over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
    if over:
        raise SystemExit(f"{T} トークンを超える行がある: {over}")
    ln = [len(v) for v in ids.values()]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids_eval.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"評価セット v5: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)}（グラフは {T} まで）")


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
            res = D.run_job({**D.base_job(f"p20_eval_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
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


def load_eval5(n: int):
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH5 / "utterances.jsonl") if r["id"] in got]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


# ---------------------------------------------------------------- text（GPU）

def fit_predict_keep(train_texts, y, predict_texts, cfg, device, seed, save_dir):
    """EdgeSLM_MultiHead_Multi の `l1.clear.train.fit_predict_bert` と同じ手順（乱数の種・並べ替え・最適化・推論）。
    違いは、最後にモデルとトークナイザを `save_dir` に保存することだけ（元の関数は確率だけを返し、モデルを残さない）"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["name"], num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    y = np.asarray(y)
    order = np.arange(len(train_texts))
    model.train()
    for _ in range(cfg["epochs"]):
        np.random.shuffle(order)
        for s in range(0, len(order), cfg["batch_size"]):
            idx = order[s:s + cfg["batch_size"]]
            batch = tok([train_texts[i] for i in idx], padding=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt").to(device)
            loss = torch.nn.functional.cross_entropy(model(**batch).logits, torch.tensor(y[idx], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(predict_texts), 64):
            batch = tok(predict_texts[s:s + 64], padding=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt").to(device)
            out.extend(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().tolist())
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tok.save_pretrained(str(save_dir))
    return np.array(out)


def text(device_name: str, seeds: tuple) -> None:
    use_multi_library()
    from edge_router.l1.decide import load_config
    cfg = load_config()["model"]
    old = json.loads((P18 / "text_scores_F.json").read_text(encoding="utf-8"))
    parts, y = fit_set(True)
    if old["labels_fit"] != y:
        raise SystemExit("P18 の案 F と、学習の行・ラベルが違う（データが変わっている）")
    fit_rows = [r for part in parts for r in part]
    yy = np.array([y[r["id"]] for r in fit_rows])
    texts = [r["utterance"] for r in fit_rows]
    pred_rows = read_jsonl(CH5 / "utterances.jsonl") + [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] == "test"] + \
        read_jsonl(CH4 / "utterances.jsonl") + read_jsonl(CH3 / "utterances.jsonl") + read_jsonl(CH2 / "utterances.jsonl") + read_jsonl(CH1)
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / "text_scores_F.json"
    out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"variant": "F", "labels_fit": y, "models": {}, "bert": cfg, "check": {}}
    if out["labels_fit"] != y:
        raise SystemExit(f"保存済みの結果とラベルが違う（データを変えたなら {f.name} を別の名前にしてから実行）")
    print(f"BERT 案 F: 学習 {len(fit_rows)} 行（機能にない依頼 {int(yy.sum())}）・seed {[cfg['seed'] + s for s in seeds]} / 確率を出す行 {len(pred_rows)}（評価セット v5・test・v4〜v1）", flush=True)
    for s in seeds:
        name = f"bert_s{s}"
        save_dir = WORK / "bert_F" / name
        m = out["models"].get(name)
        if m and all(r["id"] in m["pred"] for r in pred_rows) and (save_dir / "config.json").exists():
            print(f"{name}: 済み", flush=True)
            continue
        if name not in old["models"]:
            raise SystemExit(f"P18 の案 F に {name} が無い（交差検証の結果が要る）")
        t0 = time.time()
        pred = fit_predict_keep(texts, yy, [r["utterance"] for r in pred_rows], cfg, device_name, cfg["seed"] + s, save_dir)
        prev = old["models"][name]
        diff = [abs(float(p) - prev["pred"][r["id"]]) for r, p in zip(pred_rows, pred) if r["id"] in prev["pred"]]
        out["models"][name] = {"cv": prev["cv"], "pred": {r["id"]: float(p) for r, p in zip(pred_rows, pred)}}
        out["check"][name] = {"rows": len(diff), "max_abs_diff": float(max(diff))}
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {time.time() - t0:.0f} 秒 / P18 のときの確率との差の最大 {max(diff):.4f}（{len(diff)} 行）/ モデルを保存 {save_dir}", flush=True)
        if max(diff) > 1e-3:
            print(f"  ★{name}: P18 のときと確率が違う。判定は続けるが、P18・P19 の数字とそのまま比べられない", flush=True)
    print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge（numpy だけ）

def judge(ns: list[int]) -> None:
    f = WORK / "text_scores_F.json"
    if not f.exists():
        raise SystemExit("BERT の結果が無い（先に text）")
    raw = json.loads(f.read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = {"all": by_mode(raw, MODES[0]), "ok": by_mode(raw, MODES[0], only=tuple(chk["ok"])), "one": by_mode(raw, MODES[0], only=(chk["one"],))}
    effect = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    rows = read_jsonl(CH5 / "utterances.jsonl")
    loader = cached(load_eval5)
    n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
    c = {k: sum(r["category"] == k for r in rows) for k in NOTES}
    users = sum(r.get("author") == "user" for r in rows)
    print(f"\n===== P20 judge: 評価セット v5（{n_rows} 行 = " + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。うち利用者が足した行 {users}）=====")
    print(f"あらかじめ決めた本命: {PRIMARY}・層 {PRIMARY_LAYER}")
    md = [f"評価セット v5: {n_rows} 行（" + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。利用者が足した行 {users}）。"
          f"**あらかじめ決めた本命は「{PRIMARY}」・層 {PRIMARY_LAYER}**。数字はヘッドの seed 3 つ（× BERT の seed）の平均（かっこは最小〜最大）。信頼区間は Wilson・95%\n"]
    res = {"bert_check": chk, "bert_diff": raw.get("check", {}), "kpi": {}, "verdict": {}, "stops": {}, "severe": {}, "detail": {}, "bert": {}, "head": {}}
    line = "・".join(f"{k} {v['max_abs_diff']:.4f}" for k, v in raw.get("check", {}).items())
    print(f"確かめ: 学習し直した BERT の確率と P18 のときの確率の差の最大: {line}")
    md += [f"確かめ: 学習し直した BERT（案 F）の確率と、P18 のときの確率の差の最大（seed ごと）: {line}\n",
           "BERT の使うモデル（P19 の決まり）: " + "・".join(f"{k} {'使う' if v['ok'] else '使わない'}（交差検証 {v['cv_flag']:.3f}→確かめ用の行 {v['pool_flag']:.3f}）"
                                                  for k, v in chk["seeds"].items()) + f"。端末に載せる 1 つ = {chk['one']}\n"]
    print(md[-1].strip())

    for n in ns:
        keep = {}
        for cname, which, fpr, rules in CONFIGS:
            got = measure(n, ts[which], loader, fpr, rules)
            if got is not None:
                keep[cname] = got
        if not keep:
            print(f"層 {n}: 測れない（端末の h かヘッドが無い）")
            continue
        print(f"\n===== 層 {n} 個 =====")
        heads_ = [lab for lab, _, _ in KPI] + ["EtoE の成功の区間", "重大な誤りの割合の区間"]
        md += [f"\n## 層 {n} 個" + ("（本命の層）" if n == PRIMARY_LAYER else "") + "\n", "|構成|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
        for cname, (k, d) in keep.items():
            cells = [cell(k[key], nd) for _, key, nd in KPI] + [interval(k["e2e"]["mean"], n_rows), interval(k["severe"]["mean"], n_rows)]
            res["kpi"][f"{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, "e2e_ci": cells[-2], "severe_ci": cells[-1]}
            mark = "**" if cname == PRIMARY else ""
            print(f"{cname:<30} " + "  ".join(f"{h}: {x}" for h, x in zip(heads_, cells)))
            md.append(f"|{mark}{cname}{mark}|" + "|".join(cells) + "|")
        if PRIMARY in keep:
            v = verdict(keep[PRIMARY][0], n_rows, n_known)
            res["verdict"][str(n)] = v
            md += [f"\n本命を検証の目標と比べる（層 {n} 個）:\n", "|KPI|目標|値|目標を満たすか（平均の値）|区間（行数）|区間ごと目標の内側か|", "|---|---|---|---|---|---|"]
            for x in v:
                md.append(f"|{x['kpi']}|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|")
            ok = all(x["point"] for x in v)
            print(f"  本命と目標: " + " / ".join(f"{x['kpi']} {x['value']:.4f}（目標 {x['target']}）{'○' if x['point'] else '×'}・区間 {x['ci']}{'○' if x['sure'] else '×'}" for x in v))
            print(f"  → 層 {n}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}（平均の値）")
            md.append(f"\n**層 {n}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}（平均の値）**")
        md += [f"\n対象の依頼と雑談 {n_known} 行を止めた段（1 回あたりの行数）:\n", "|構成|規則|引数が足りない|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|検査|合計|", "|---|---|---|---|---|---|---|---|---|"]
        for cname, (k, d) in keep.items():
            cnt = collections.Counter()
            for x in d["stop_rows"]:
                cnt[x["stop"]] += x["share"]
            v = [cnt["聞き返し（規則）"], cnt["聞き返し（引数が足りない）"], cnt["聞き返し（棄却ヘッド）"], cnt["聞き返し（確信度）"], cnt["聞き返し（BERT のフラグ）"],
                 cnt["拒否（BERT が断る）"], cnt["聞き返し（検査）"] + cnt["拒否（検査）"]]
            res["stops"][f"{n}|{cname}"] = v
            md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in v) + f"|{sum(v):.1f}|")
            print(f"  {cname[:16]:<16} 止めた段: 規則 {v[0]:.1f}・引数 {v[1]:.1f}・棄却 {v[2]:.1f}・確信度 {v[3]:.1f}・BERT フラグ {v[4]:.1f}・BERT 断る {v[5]:.1f}・検査 {v[6]:.1f}")
        md += ["\n重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|合計|", "|---|" + "---|" * (len(CAT) + 1)]
        for cname, (k, d) in keep.items():
            cnt = collections.Counter()
            for x in d["by_group"]:
                if x["grade"] == "重大":
                    cnt[x["category"]] += x["rows"]
            res["severe"][f"{n}|{cname}"] = dict(cnt)
            md.append(f"|{cname}|" + "|".join(f"{cnt[k_]:.1f}" for k_ in CAT) + f"|{sum(cnt.values()):.1f}|")
            print(f"  {cname[:16]:<16} 重大: " + "・".join(f"{CAT[k_]} {cnt[k_]:.1f}" for k_ in CAT))
        if PRIMARY in keep:
            d = keep[PRIMARY][1]
            # 種類ごとの内訳
            md += [f"\n本命の種類ごとの内訳（層 {n} 個。割合）:\n", "|種類|行|正解／回復可能／通常／重大|", "|---|---|---|"]
            gkeys = []
            for r in rows:
                gk = (r["category"], (r.get("os_group") or r.get("g_kind")) if r["category"] in "HG" else None)
                if gk not in gkeys:
                    gkeys.append(gk)
            for cat_, grp in gkeys:
                sel = [r for r in rows if r["category"] == cat_ and (grp is None or (r.get("os_group") or r.get("g_kind")) == grp)]
                tot = {g: sum(y["rows"] for y in d["by_group"] if y["category"] == cat_ and (grp is None or y["group"] == grp) and y["grade"] == g)
                       for g in ("正解", "回復可能", "通常", "重大")}
                md.append(f"|{NOTES[cat_]}" + (f"・{grp}" if grp else "") + f"|{len(sel)}|" + "／".join(f"{tot[g] / len(sel):.3f}" for g in ("正解", "回復可能", "通常", "重大")) + "|")
            # S5 が止めた対象の行 = 本命で規則が止めた行のうち、S5 なしの構成では規則で止まっていない行
            base = {x["utterance"] for x in keep[CONFIGS[2][0]][1]["stop_rows"] if x["stop"] == "聞き返し（規則）"} if CONFIGS[2][0] in keep else set()
            s5 = [x for x in d["stop_rows"] if x["stop"] == "聞き返し（規則）" and x["utterance"] not in base]
            res["detail"][str(n)] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows", "by_group", "stages")}
            res["detail"][str(n)]["s5_stopped_known"] = s5
            md += [f"\n本命で見るもの（層 {n} 個）:\n", "|見るもの|行（出た割合）|", "|---|---|",
                   "|重大な誤り|" + " / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:30]) + "|",
                   "|通常の誤り|" + " / ".join(f"[{y['category']}]{y['utterance']} → {y['outcome']}（{y['share']:.2f}）" for y in d["normal_rows"][:30]) + "|",
                   "|止めた対象の行|" + " / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:50]) + "|",
                   "|うち、規則 S5 が止めた対象の行|" + " / ".join(f"{y['utterance']}（{y['share']:.2f}）" for y in s5) + "|"]
            print("  本命で残る重大な誤り: " + " / ".join(f"[{y['category']}]{y['utterance']}→{y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:12]))
            print("  S5 が止めた対象の行: " + " / ".join(f"{y['utterance']}（{y['share']:.2f}）" for y in s5))

    # ---- 部品ごと（評価セット v5）
    md.append("\n## 部品ごとの精度（評価セット v5。規則・引数は外す）\n")
    bp = bert_part(ts["all"], rows)
    labels = [("断る閾値", "hi"), ("機能にない依頼にフラグ以上（再現率）", "H_flag"), ("　学習に同じ機能がある", "H_flag_seen"), ("　どの学習にも無い機能", "H_flag_unseen"),
              ("機能にない依頼を断る閾値以上", "H_refuse"), ("対象の依頼と雑談に誤ってフラグ", "known_flag"), ("　機能の語を含む依頼", "K_flag"), ("　ふつうの依頼", "N_flag"), ("　雑談", "F_flag"),
              ("対象を断った（行）", "known_refuse_n"), ("意図が不明にフラグ", "G_flag"), ("AUC（機能にない依頼 ↔ 対象の依頼と雑談）", "auc_known")]
    names = sorted(k for k in ts["all"]["models"])
    print("\n===== 部品: BERT 案 F（評価セット v5・seed ごと）=====")
    md += ["### BERT 案 F（seed ごと。フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上）\n", "|指標|" + "|".join(f"{nm}" + ("（載せる 1 つ）" if nm == chk["one"] else "" if chk["seeds"][nm]["ok"] else "（使わない）") for nm in names) + "|",
           "|---|" + "---|" * len(names)]
    for lab, key in labels:
        cells = [f"{x:.3f}" if isinstance(x, float) else str(x) for x in bp.get(key, [])]
        print(f"{lab:<26} " + "  ".join(cells))
        md.append(f"|{lab}|" + "|".join(cells) + "|")
    res["bert"] = bp
    for n in ns:
        loaded = loader(n)
        if loaded is None:
            continue
        rs, last, mean = loaded
        hp = head_part(n, SRC1584, rs, last, mean, effect)
        if not hp:
            continue
        labels = [("行き先の正解率（対象の依頼と雑談）", "acc", 3), ("　機能の語を含む依頼", "acc_K", 3), ("　ふつうの依頼", "acc_N", 3), ("　雑談", "acc_F", 3)] + \
                 [(f"　再現率: {J[c_]}", f"recall_{J[c_]}", 3) for c_ in ORDER] + \
                 [("対象の依頼と雑談を通した割合", "cover", 3), ("通したうち、別の操作系と判定（行）", "wrong_write", 1),
                  ("機能にない依頼を止めた（棄却・確信度）", "H_stop", 3), ("　操作系として通した", "H_pass_write", 3), ("　棄却の得点の AUC", "auc_H", 3),
                  ("意図が不明を止めた（棄却・確信度）", "G_stop", 3), ("　操作系として通した", "G_pass_write", 3), ("　棄却の得点の AUC", "auc_G", 3)]
        print(f"\n===== 部品: LM ヘッド 1,584 行・層 {n} 個（評価セット v5）=====")
        md += [f"\n### LM ヘッド 1,584 行・層 {n} 個（mlp・seed 3 つ。棄却の閾値は validation の 2%）\n", "|指標|値|", "|---|---|"]
        for lab, key, nd in labels:
            print(f"{lab:<28} {ms(hp[key], nd)}")
            md.append(f"|{lab}|{ms(hp[key], nd)}|")
        md.append(f"\n取り違えの多い組（正解 → 判定。1 つのヘッドあたりの行数）: " + " / ".join(f"{x['true']} → {x['pred']} {x['rows']:.1f}" for x in hp["confusions"]))
        res["head"][str(n)] = hp
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "verdict.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（verdict.md・verdict.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "text", "judge"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in BERT_SEEDS))
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P20 {a.step} =====", flush=True)
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
