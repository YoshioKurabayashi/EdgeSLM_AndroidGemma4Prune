# -*- coding: utf-8 -*-
"""P21 手順 3: 規則と引数を直した構成の合否を、新しい評価セット（challenge_v6・600 行）で測る。

設計: DoC/設計書_P21_規則と引数の誤りを直す.md の 7 節 ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

**本命（測る前に決めた）= P19 で決めた構成 ＋ P21 の直し（`p21_rules`・`p21_arguments`）・層 26・端末に載せる BERT 1 つ。**
ヘッドも BERT も学習し直さない。BERT は P20 で保存したモデル（`work/p20/bert_F/`）で、評価セット v6 の確率を出すだけ。
保存したモデルが P20 のときと同じ確率を出すかを、評価セット v5 の行で確かめる。

|段|何をするか|venv|
|---|---|---|
|`encode`|評価セット v6 をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、評価セット v6 を層 17・26・35 のグラフに流し、`h` を集める（途中で止めても済んだ分は残る）|EdgeSLM_Android|
|`text`|保存した BERT（案 F × seed 3 つ）で、評価セット v6 の確率を出す（学習はしない）|EdgeSLM_MultiHead_Multi（GPU。無ければ CPU）|
|`judge`|評価セット v6 で、本命と比べる相手の KPI・合否・内訳・部品ごとの精度を出す。numpy と PyYAML だけ|どちらでも|

    python -m app.p21_verdict encode
    python -m app.p21_verdict device --layers 17,26,35
    python -m app.p21_verdict text
    python -m app.p21_verdict judge --layers 17,26,35      # results/P21/verdict.md・verdict.json
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT
from .p5_retrain import P3
from .p7_data import TR2, read_jsonl
from .p11_bert import MODES, by_mode
from .p13_eval import cell, interval
from .p16_parts import J, ORDER, ms
from .p17_arguments import make_resolve as args_p17
from .p18_retrain import SRC1584, bert_part, cached, head_part, verdict
from .p19_config import bert_check
from .p19_rules import make_rules as rules_p19
from .p20_data import CH5
from .p20_verdict import WORK as P20
from .p21_arguments import make_resolve as args_p21
from .p18_data import md5_of
from .p21_data import CH6, NOTES
from .p21_fix import measure
from .p21_rules import make_rules as rules_p21

WORK = PROJECT_ROOT / "work" / "p21"
RES = PROJECT_ROOT / "results" / "P21"
T = 64
BERT_SEEDS = (0, 1, 2)
PRIMARY_LAYER = 26
# （名前, BERT の seed の選び方, 引数, 規則）。"one" = 端末に載せる 1 つ・"ok" = 決まりで残った seed。棄却の閾値はどれも validation の 2%
CONFIGS = [("直した構成・BERT 1 つ（本命）", "one", args_p21, rules_p21),
           ("比べる相手: 直す前（P20 の構成）", "one", args_p17, rules_p19),
           ("比べる相手: 規則だけ直す", "one", args_p17, rules_p21),
           ("比べる相手: 引数だけ直す", "one", args_p21, rules_p19),
           ("直した構成・残った seed の平均", "ok", args_p21, rules_p21)]
PRIMARY, BEFORE = CONFIGS[0][0], CONFIGS[1][0]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("重大な誤りの割合", "severe", 4), ("不要な聞き返し", "over", 3),
       ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3), ("対象を断った（件）", "known_refused_n", 1)]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}
STOPS = ["聞き返し（規則）", "聞き返し（引数が足りない）", "聞き返し（棄却ヘッド）", "聞き返し（確信度）", "聞き返し（BERT のフラグ）", "拒否（BERT が断る）"]


def layers_of(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", ",").split(",") if x]


# ---------------------------------------------------------------- encode・device（P20 と同じ手順。行と保存先だけ違う）

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
    rows = read_jsonl(CH6 / "utterances.jsonl")
    ids = {r["id"]: enc(r) for r in rows}
    over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
    if over:
        raise SystemExit(f"{T} トークンを超える行がある: {over}")
    ln = [len(v) for v in ids.values()]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids_eval.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"評価セット v6: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)}（グラフは {T} まで）")


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
            res = D.run_job({**D.base_job(f"p21_eval_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
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


def load_eval6(n: int):
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH6 / "utterances.jsonl") if r["id"] in got]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


# ---------------------------------------------------------------- text（保存したモデルで確率を出すだけ。学習しない）

def predict_saved(save_dir, texts: list[str], max_length: int, device_name: str) -> np.ndarray:
    """`p20_verdict.fit_predict_keep` の推論の部分と同じ（64 行ずつ・softmax の 2 つ目）"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(save_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(device_name)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 64):
            batch = tok(texts[s:s + 64], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device_name)
            out.extend(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().tolist())
    return np.array(out)


def text(device_name: str, seeds: tuple) -> None:
    src = P20 / "text_scores_F.json"
    if not src.exists():
        raise SystemExit(f"P20 の BERT の結果が無い（{src}。先に scripts\\run_p20.ps1）")
    raw = json.loads(src.read_text(encoding="utf-8"))
    if device_name == "cuda":
        import torch
        if not torch.cuda.is_available():
            device_name = "cpu"
            print("GPU が使えないので CPU で出す（学習はしないので数分で終わる）", flush=True)
    rows5, rows6 = read_jsonl(CH5 / "utterances.jsonl"), read_jsonl(CH6 / "utterances.jsonl")
    out = {**raw, "check": {}, "source": "P20 の結果に、保存したモデルで出した評価セット v6 の確率を足したもの"}
    print(f"BERT 案 F（P20 で保存したモデル）: 確率を出す行 {len(rows6)}（評価セット v6）＋ 確かめ {len(rows5)}（評価セット v5）/ {device_name}", flush=True)
    for s in seeds:
        name = f"bert_s{s}"
        save_dir = P20 / "bert_F" / name
        if name not in raw["models"] or not (save_dir / "config.json").exists():
            raise SystemExit(f"{name}: P20 で保存したモデルが無い（{save_dir}。先に scripts\\run_p20.ps1）")
        t0 = time.time()
        pred = predict_saved(save_dir, [r["utterance"] for r in rows5 + rows6], raw["bert"]["max_length"], device_name)
        prev = raw["models"][name]["pred"]
        diff = [abs(float(p) - prev[r["id"]]) for r, p in zip(rows5, pred) if r["id"] in prev]
        out["models"][name] = {"cv": raw["models"][name]["cv"], "pred": {**prev, **{r["id"]: float(p) for r, p in zip(rows6, pred[len(rows5):])}}}
        out["check"][name] = {"rows": len(diff), "max_abs_diff": float(max(diff))}
        print(f"{name}: {time.time() - t0:.0f} 秒 / P20 のときの確率との差の最大 {max(diff):.4f}（評価セット v5 の {len(diff)} 行）", flush=True)
        if max(diff) > 1e-3:
            raise SystemExit(f"{name}: 保存したモデルが P20 のときと違う確率を出す（差 {max(diff):.4f}）。モデルが入れ替わっていないか確かめる")
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / "text_scores_F.json"
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge（numpy だけ）

def share_by(items: list[dict], stops: tuple | None = None) -> collections.Counter:
    c = collections.Counter()
    for x in items:
        if stops is None or x["stop"] in stops:
            c[x["utterance"]] += x["share"]
    return c


def judge(ns: list[int]) -> None:
    f = WORK / "text_scores_F.json"
    if not f.exists():
        raise SystemExit("BERT の結果が無い（先に text）")
    raw = json.loads(f.read_text(encoding="utf-8"))
    rows = read_jsonl(CH6 / "utterances.jsonl")
    for name, m in raw["models"].items():
        miss = [r["id"] for r in rows if r["id"] not in m["pred"]]
        if miss:
            raise SystemExit(f"{name}: BERT の確率が無い行がある（{len(miss)} 行。評価セット v6 を変えたなら、text をもう一度）")
    chk = bert_check(raw)
    ts = {"ok": by_mode(raw, MODES[0], only=tuple(chk["ok"])), "one": by_mode(raw, MODES[0], only=(chk["one"],)), "all": by_mode(raw, MODES[0])}
    effect = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    loader = cached(load_eval6)
    n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
    c = {k: sum(r["category"] == k for r in rows) for k in NOTES}
    users = sum(r.get("author") == "user" for r in rows)
    print(f"\n===== P21 judge: 評価セット v6（{n_rows} 行 = " + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。うち利用者が足した行 {users}）=====")
    print(f"あらかじめ決めた本命: {PRIMARY}・層 {PRIMARY_LAYER}")
    md = [f"評価セット v6: {n_rows} 行（" + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。利用者が足した行 {users}）。"
          f"**あらかじめ決めた本命は「{PRIMARY}」・層 {PRIMARY_LAYER}**。数字はヘッドの seed 3 つ（× BERT の seed）の平均（かっこは最小〜最大）。信頼区間は Wilson・95%\n"]
    res = {"bert_check": chk, "bert_diff": raw.get("check", {}), "kpi": {}, "verdict": {}, "stops": {}, "severe": {}, "detail": {}, "changed": {}, "bert": {}, "head": {}}
    here = PROJECT_ROOT / "app"
    sums = {"評価セット v6": md5_of(CH6 / "utterances.jsonl"), "p21_rules.py": md5_of(here / "p21_rules.py"), "p21_arguments.py": md5_of(here / "p21_arguments.py")}
    res["md5"] = sums
    print("測ったもの（md5）: " + "・".join(f"{k} {v}" for k, v in sums.items()))
    md.append("測ったもの（md5。改行が LF のとき、設計書 7 節の値と同じになる）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "\n")
    line = "・".join(f"{k} {v['max_abs_diff']:.4f}" for k, v in raw.get("check", {}).items())
    print(f"確かめ: 保存した BERT の確率と P20 のときの確率の差の最大（評価セット v5 の行）: {line}")
    md += [f"確かめ: 保存した BERT（案 F）が出す確率と、P20 のときの確率の差の最大（評価セット v5 の行・seed ごと）: {line}\n",
           "BERT の使うモデル（P19 の決まり）: " + "・".join(f"{k} {'使う' if v['ok'] else '使わない'}（交差検証 {v['cv_flag']:.3f}→確かめ用の行 {v['pool_flag']:.3f}）"
                                                  for k, v in chk["seeds"].items()) + f"。端末に載せる 1 つ = {chk['one']}\n"]
    print(md[-1].strip())

    for n in ns:
        keep = {}
        for cname, which, mr, mk in CONFIGS:
            got = measure(n, ts[which], loader, mr, mk)
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
            print(f"{cname:<24} " + "  ".join(f"{h}: {x}" for h, x in zip(heads_, cells)))
            md.append(f"|{mark}{cname}{mark}|" + "|".join(cells) + "|")
        v = verdict(keep[PRIMARY][0], n_rows, n_known)
        res["verdict"][str(n)] = v
        md += [f"\n本命を検証の目標と比べる（層 {n} 個）:\n", "|KPI|目標|値|目標を満たすか（平均の値）|区間（行数）|区間ごと目標の内側か|", "|---|---|---|---|---|---|"]
        for x in v:
            md.append(f"|{x['kpi']}|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|")
        ok = all(x["point"] for x in v)
        print("  本命と目標: " + " / ".join(f"{x['kpi']} {x['value']:.4f}（目標 {x['target']}）{'○' if x['point'] else '×'}・区間 {x['ci']}{'○' if x['sure'] else '×'}" for x in v))
        print(f"  → 層 {n}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}（平均の値）")
        md.append(f"\n**層 {n}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}（平均の値）**")

        md += [f"\n対象の依頼と雑談 {n_known} 行を止めた段（1 回あたりの行数）:\n", "|構成|規則|引数が足りない|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|検査|合計|", "|---|---|---|---|---|---|---|---|---|"]
        for cname, (k, d) in keep.items():
            cnt = collections.Counter()
            for x in d["stop_rows"]:
                cnt[x["stop"]] += x["share"]
            vv = [cnt[s_] for s_ in STOPS] + [cnt["聞き返し（検査）"] + cnt["拒否（検査）"]]
            res["stops"][f"{n}|{cname}"] = vv
            md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in vv) + f"|{sum(vv):.1f}|")
            print(f"  {cname[:14]:<14} 止めた段: 規則 {vv[0]:.1f}・引数 {vv[1]:.1f}・棄却 {vv[2]:.1f}・確信度 {vv[3]:.1f}・BERT フラグ {vv[4]:.1f}・BERT 断る {vv[5]:.1f}・検査 {vv[6]:.1f}")
        md += ["\n重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|合計|", "|---|" + "---|" * (len(CAT) + 1)]
        for cname, (k, d) in keep.items():
            cnt = collections.Counter()
            for x in d["by_group"]:
                if x["grade"] == "重大":
                    cnt[x["category"]] += x["rows"]
            res["severe"][f"{n}|{cname}"] = dict(cnt)
            md.append(f"|{cname}|" + "|".join(f"{cnt[k_]:.1f}" for k_ in CAT) + f"|{sum(cnt.values()):.1f}|")
            print(f"  {cname[:14]:<14} 重大: " + "・".join(f"{CAT[k_]} {cnt[k_]:.1f}" for k_ in CAT))

        d = keep[PRIMARY][1]
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

        # 直しで変わった行 = 直す前（P20 の構成）と本命で、止まり方・重大な誤りが変わった行
        b = keep[BEFORE][1]
        ra = ("聞き返し（規則）", "聞き返し（引数が足りない）")
        sb, sa = share_by(b["stop_rows"], ra), share_by(d["stop_rows"], ra)
        freed = [[u, round(sb[u] - sa[u], 2)] for u in sb if sb[u] - sa[u] > 1e-9]
        newly = [[u, round(sa[u] - sb[u], 2)] for u in sa if sa[u] - sb[u] > 1e-9]
        vb, va = share_by(b["severe_rows"]), share_by(d["severe_rows"])
        fixed = [[u, round(vb[u] - va[u], 2)] for u in vb if vb[u] - va[u] > 1e-9]
        worse = [[u, round(va[u] - vb[u], 2)] for u in va if va[u] - vb[u] > 1e-9]
        res["changed"][str(n)] = {"freed": freed, "newly_stopped": newly, "severe_fixed": fixed, "severe_new": worse}
        res["detail"][str(n)] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows", "by_group", "stages")}
        fmt = lambda items: " / ".join(f"{u}（{s:.2f}）" for u, s in items) if items else "なし"
        md += [f"\n直しで変わった行（直す前 → 本命。層 {n} 個。かっこは変わった割合）:\n", "|見るもの|行|", "|---|---|",
               f"|規則・引数で止まらなくなった対象の行（{sum(s for _, s in freed):.1f}）|{fmt(freed)}|",
               f"|規則・引数で新しく止まった対象の行（{sum(s for _, s in newly):.1f}）|{fmt(newly)}|",
               f"|重大な誤りでなくなった行（{sum(s for _, s in fixed):.1f}）|{fmt(fixed)}|",
               f"|新しく重大な誤りになった行（{sum(s for _, s in worse):.1f}）|{fmt(worse)}|"]
        md += [f"\n本命で見るもの（層 {n} 個）:\n", "|見るもの|行（出た割合）|", "|---|---|",
               "|重大な誤り|" + " / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:30]) + "|",
               "|通常の誤り|" + " / ".join(f"[{y['category']}]{y['utterance']} → {y['outcome']}（{y['share']:.2f}）" for y in d["normal_rows"][:30]) + "|",
               "|止めた対象の行|" + " / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:60]) + "|"]
        print("  本命で残る重大な誤り: " + " / ".join(f"[{y['category']}]{y['utterance']}→{y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:12]))
        print(f"  直しで止まらなくなった対象の行: {fmt(freed)}")
        print(f"  直しで新しく止まった対象の行: {fmt(newly)} / 新しく重大な誤りになった行: {fmt(worse)}")

    # ---- 部品ごと（評価セット v6）
    md.append("\n## 部品ごとの精度（評価セット v6。規則・引数は外す）\n")
    bp = bert_part(ts["all"], rows)
    labels = [("断る閾値", "hi"), ("機能にない依頼にフラグ以上（再現率）", "H_flag"), ("　学習に同じ機能がある", "H_flag_seen"), ("　どの学習にも無い機能", "H_flag_unseen"),
              ("機能にない依頼を断る閾値以上", "H_refuse"), ("対象の依頼と雑談に誤ってフラグ", "known_flag"), ("　機能の語を含む依頼", "K_flag"), ("　ふつうの依頼", "N_flag"), ("　雑談", "F_flag"),
              ("対象を断った（行）", "known_refuse_n"), ("意図が不明にフラグ", "G_flag"), ("AUC（機能にない依頼 ↔ 対象の依頼と雑談）", "auc_known")]
    names = sorted(k for k in ts["all"]["models"])
    print("\n===== 部品: BERT 案 F（評価セット v6・seed ごと）=====")
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
        print(f"\n===== 部品: LM ヘッド 1,584 行・層 {n} 個（評価セット v6）=====")
        md += [f"\n### LM ヘッド 1,584 行・層 {n} 個（mlp・seed 3 つ。棄却の閾値は validation の 2%）\n", "|指標|値|", "|---|---|"]
        for lab, key, nd in labels:
            print(f"{lab:<28} {ms(hp[key], nd)}")
            md.append(f"|{lab}|{ms(hp[key], nd)}|")
        md.append("\n取り違えの多い組（正解 → 判定。1 つのヘッドあたりの行数）: " + " / ".join(f"{x['true']} → {x['pred']} {x['rows']:.1f}" for x in hp["confusions"]))
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
    print(f"\n===== P21 {a.step} =====", flush=True)
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
