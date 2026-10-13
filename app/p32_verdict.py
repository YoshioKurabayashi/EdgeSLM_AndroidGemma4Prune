# -*- coding: utf-8 -*-
"""P32 手順 2: P31 の仕上げを入れた構成の合否を、新しい評価セット（challenge_v9・600 行）で測る。

設計: DoC/設計書_P32_新しい評価セットv9で合否.md ｜ 測る前の記録: results/P32/preregistration.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

**本命（測る前に決めた）= ヘッド 1,656 行・P31 の規則と引数（範囲外は断る）・S1 で選んだ seed（validation でヘッドが通す割合が一番大きい seed）・BERT bert_s0。**
比べる相手: P28 の構成（P21 の規則・P26 の引数・絞らない）／いまのアプリ（ヘッド 1,584 行・P23 の引数・seed 7）。合否は値まで見た KPI・載せる 1 つの seed で決める（3 つの平均も出す）。

|段|何をするか|venv|
|---|---|---|
|`encode`|評価セット v9 をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、v9 を層 17・26 のグラフに流し、`h` を集める|EdgeSLM_Android|
|`text`|保存した BERT（案 F）で v9 の確率を出す（v8 の行で P28 のときと同じ確率かを確かめる）|EdgeSLM_MultiHead_Multi|
|`judge`|合否・止めた段・機能にない依頼の形ごとの扱い・見ていない対象の言い方を断った割合|どちらでも|

    python -m app.p32_verdict encode
    python -m app.p32_verdict device --layers 17,26
    python -m app.p32_verdict text
    python -m app.p32_verdict judge --layers 17,26      # results/P32/verdict.md・verdict.json
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT, use_multi_library
from . import p18_retrain as R
from .p5_retrain import P3
from .p7_data import TR2, read_jsonl
from .p8_gate import STEPS, run
from .p11_bert import MODES, by_mode
from .p13_eval import cell, hi_rule, interval
from .p18_data import md5_of
from .p18_retrain import SRC1584, cached, has_heads, verdict
from .p19_config import bert_check
from .p21_rules import make_rules as rules_p21
from .p21_verdict import KPI, STOPS, predict_saved, share_by
from .p22_fix import value_kpi
from .p23_arguments import make_resolve as args_p23
from .p26_arguments import make_resolve as args_p26
from .p28_data import CH8
from .p28_verdict import SRC1656, WORK as P28
from .p31_arguments import make_resolve as args_p31
from .p31_check import pick_seed
from .p31_rules import make_rules as rules_p31
from .p31_scope import READ, evidence
from .p32_data import CH9, labeled

WORK = PROJECT_ROOT / "work" / "p32"
RES = PROJECT_ROOT / "results" / "P32"
T = 64
BERT_SEEDS = (0, 1, 2)
FIXED = {"p21_rules.py": "8d74b163767c81df036d74142d9fc7ad", "p23_arguments.py": "f3e6afe7c86c98a191115a505b37f681", "p26_arguments.py": "d88f09926c802fa8f945a31c3bfc4b64",
         "p29_scope.py": "a7a13896e6785a0b1700728d86b6225f", "p30_scope.py": "30768b9bb46e933d5b6514ca0b50e206", "p30_words.py": "9e19baa3a66c4b7b1e994e518a269a73",
         "p31_scope.py": "cbab5b13536d717f4336cd65c65c3dc6", "p31_arguments.py": "411e0c9686019b6068a4d49102035922", "p31_rules.py": "17013f0b070da1991649694f0f648752"}
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}
GRADES = ("正解", "回復可能", "通常", "重大")
VKPI = [("違う値での実行（行）", "wrong_exec", 1), ("値まで見た EtoE の成功", "e2e_value", 3), ("値まで見た重大な誤り（件）", "severe_value_n", 1), ("値まで見た重大な誤りの割合", "severe_value", 4)]


def layers_of(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", ",").split(",") if x]


def configs(n: int, s1: int) -> list:
    """（名前, ヘッド, 引数, 規則, seed）。seed None = 3 つの平均"""
    return [(f"本命: P31・seed {s1}（S1）", SRC1656, args_p31, rules_p31, s1),
            ("本命の構成・seed 3 つの平均", SRC1656, args_p31, rules_p31, None),
            (f"比べる相手 1: P28 の構成・seed {s1}", SRC1656, args_p26, rules_p21, s1),
            ("比べる相手 1: P28 の構成・seed 3 つの平均", SRC1656, args_p26, rules_p21, None),
            ("比べる相手 2: いまのアプリ（1,584 行・P23・seed 7）", SRC1584, args_p23, rules_p21, 7)]


# ---------------------------------------------------------------- encode・device・text

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
    rows = read_jsonl(CH9 / "utterances.jsonl")
    ids = {r["id"]: enc(r) for r in rows}
    over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
    if over:
        raise SystemExit(f"{T} トークンを超える行がある: {over}")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids_eval.json").write_text(json.dumps(ids), encoding="utf-8")
    ln = [len(v) for v in ids.values()]
    print(f"評価セット v9: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)} / md5 {md5_of(CH9 / 'utterances.jsonl')}")


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
            pf = parts / f"L{n}_{k:02d}.json"
            if pf.exists():
                rows = json.loads(pf.read_text(encoding="utf-8"))["rows"]
                if [r["id"] for r in rows] == part:
                    got += rows
                    continue
            res = D.run_job({**D.base_job(f"p32_eval_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
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


def load_eval9(n: int, by_form: bool = False):
    """by_form = True なら、機能にない依頼の学習に無い機能を「unseen操作」「unseen問い合わせ」に分けた os_group で返す（内訳を数えるため）"""
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH9 / "utterances.jsonl") if r["id"] in got]
    if by_form:
        rows = [{**r, "os_group": f"unseen{r['os_form']}"} if r.get("os_group") == "unseen" else r for r in rows]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


def text(device_name: str, seeds: tuple) -> None:
    src = P28 / "text_scores_F.json"
    if not src.exists():
        raise SystemExit(f"P28 の BERT の結果が無い（{src}）")
    raw = json.loads(src.read_text(encoding="utf-8"))
    if device_name == "cuda":
        import torch
        if not torch.cuda.is_available():
            device_name = "cpu"
    rows8, rows9 = read_jsonl(CH8 / "utterances.jsonl"), read_jsonl(CH9 / "utterances.jsonl")
    out = {**raw, "check": {}, "source": "P28 の結果に、保存したモデルで出した評価セット v9 の確率を足したもの"}
    for s in seeds:
        name = f"bert_s{s}"
        save_dir = PROJECT_ROOT / "work" / "p20" / "bert_F" / name
        t0 = time.time()
        pred = predict_saved(save_dir, [r["utterance"] for r in rows8 + rows9], raw["bert"]["max_length"], device_name)
        prev = raw["models"][name]["pred"]
        diff = [abs(float(p) - prev[r["id"]]) for r, p in zip(rows8, pred) if r["id"] in prev]
        out["models"][name] = {"cv": raw["models"][name]["cv"], "pred": {**prev, **{r["id"]: float(p) for r, p in zip(rows9, pred[len(rows8):])}}}
        out["check"][name] = {"rows": len(diff), "max_abs_diff": float(max(diff))}
        print(f"{name}: {time.time() - t0:.0f} 秒 / P28 のときの確率との差の最大 {max(diff):.4f}（評価セット v8 の {len(diff)} 行）", flush=True)
        if max(diff) > 1e-3:
            raise SystemExit(f"{name}: 保存したモデルが P28 のときと違う確率を出す（差 {max(diff):.4f}）")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "text_scores_F.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"保存しました: {WORK / 'text_scores_F.json'}", flush=True)


# ---------------------------------------------------------------- judge

def measure(n: int, src: str, ts: dict, loader, make_resolve, make_rules, seed):
    if not has_heads(n, src) or loader(n) is None:
        return None
    base = R.hook_for(src)
    hook = base if seed is None else (lambda n_, s_, heads: [h for h in base(n_, s_, heads) if h[0] == seed])
    final = STEPS[-1]
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=hook, steps=[final], sources=[src], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=make_rules, amb_chat_ok=True)
    key = f"{n}|{src}|{final[0]}"
    return blob["kpi"][key], blob["detail"][key]


def judge(ns: list[int]) -> None:
    f = WORK / "text_scores_F.json"
    if not f.exists():
        raise SystemExit("BERT の結果が無い（先に text）")
    raw = json.loads(f.read_text(encoding="utf-8"))
    rows = read_jsonl(CH9 / "utterances.jsonl")
    use_multi_library()
    from edge_router.dispatch import arguments
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    lab = labeled(rows)
    here = PROJECT_ROOT / "app"
    sums = {"評価セット v9": md5_of(CH9 / "utterances.jsonl"), **{k: md5_of(here / k) for k in FIXED}}
    changed = [k for k, v in FIXED.items() if sums[k] != v]
    n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
    c = {k: sum(r["category"] == k for r in rows) for k in CAT}
    head = f"評価セット v9（{n_rows} 行 = " + "・".join(f"{CAT[k]} {v}" for k, v in c.items()) + f"。機能にない依頼のうち学習に無い機能は操作 40・問い合わせ 40）"
    print(f"\n===== P32 judge: {head} =====")
    md = ["# P32 合否（評価セット v9）\n", f"{head}。**本命（測る前に決めた）= ヘッド 1,656 行・P31 の規則と引数（範囲外は断る）・S1 の seed**。合否は値まで見た KPI・載せる 1 つの seed で決める。区間は Wilson・95%\n",
          "測ったもの（md5。改行が LF のとき）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "\n",
          ("**測る前に固定したファイルが変わっている: " + "・".join(changed) + "**\n") if changed else "測る前に固定したファイル（規則・引数・語の一覧）は、記録した md5 と同じ（変えていない）\n",
          "確かめ: BERT の確率と P28 のときの確率の差の最大（v8 の行）: " + "・".join(f"{k} {v['max_abs_diff']:.4f}" for k, v in raw.get("check", {}).items()) + f"。BERT = {chk['one']}\n"]
    print("\n".join(md[2:]).strip())
    res = {"md5": sums, "fixed_changed": changed, "seed": {}, "kpi": {}, "verdict": {}, "stops": {}, "severe": {}, "forms": {}, "detail": {}, "lexicon": {}, "bert": {}}
    # 見ていない対象の言い方を断る割合（正しい機能のとき・ヘッドに関係なく）
    kn = [r for r in rows if r["action"][5:] in READ]
    miss = [r["utterance"] for r in kn if not evidence(r["action"][5:], r["utterance"])]
    res["lexicon"] = {"read_rows": len(kn), "unexplained": miss}
    md.append(f"## 見ていない対象の言い方を断る割合（正しい機能のとき）\n\n対象の読み出しの行 {len(kn)} 行のうち、機能の定義で説明がつかない（＝断る）行: **{len(miss)} 行（{len(miss) / len(kn) * 100:.1f}%）**" + (f"（{' / '.join(miss)}）" if miss else "") + "\n")
    print(md[-1].strip())
    # BERT（形ごと）
    s0 = ts["models"][chk["one"]]["pred"] if "models" in ts else raw["models"][chk["one"]]["pred"]
    for form in ("seen", "操作", "問い合わせ"):
        sel = [r for r in rows if r.get("os_form") == form]
        res["bert"][form] = {"flag": float(np.mean([raw["models"][chk["one"]]["pred"][r["id"]] >= 0.5 for r in sel])), "rows": len(sel)}
    md.append("BERT（" + chk["one"] + "）が機能にない依頼にフラグ（0.5 以上）を立てた割合: " + "・".join(f"{k} {v['flag']:.2f}（{v['rows']} 行）" for k, v in res["bert"].items()) + "\n")
    print(md[-1].strip())

    for n in ns:
        s1, stat = pick_seed(n)
        res["seed"][str(n)] = {"best": s1, "stat": stat}
        loader, loader_f = cached(lambda n_: load_eval9(n_)), cached(lambda n_: load_eval9(n_, True))
        keep = {}
        for cname, src, make, mrules, sd in configs(n, s1):
            got = measure(n, src, ts, loader, make, mrules, sd)
            if got is not None:
                k, d = got
                keep[cname] = (k, d, value_kpi(k, d, lab, make(arguments), state, n_rows), (src, make, mrules, sd))
        if not keep:
            print(f"層 {n}: 測れない")
            continue
        primary = configs(n, s1)[0][0]
        print(f"\n===== 層 {n} 個（S1 の seed = {s1}）=====")
        heads_ = [h for h, _, _ in KPI] + [h for h, _, _ in VKPI] + ["値まで見た EtoE の区間", "値まで見た重大な誤りの区間"]
        md += [f"\n## 層 {n} 個（S1 の seed = {s1}。validation でヘッドが通す割合: " + "・".join(f"{s}: {v['cover']:.3f}" for s, v in stat.items()) + "）\n",
               "|構成|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
        for cname, (k, d, v, _) in keep.items():
            cells = [cell(k[key], nd) for _, key, nd in KPI] + [f"{v[key]:.{nd}f}" for _, key, nd in VKPI] + [interval(v["e2e_value"], n_rows), interval(v["severe_value"], n_rows)]
            res["kpi"][f"{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, **{key: v[key] for _, key, _ in VKPI}, "wrong_rows": v["rows"]}
            mark = "**" if cname == primary else ""
            md.append(f"|{mark}{cname}{mark}|" + "|".join(cells) + "|")
            print(f"  {cname[:30]:<30} EtoE {v['e2e_value']:.3f}・重大 {v['severe_value_n']:.1f}（{v['severe_value'] * 100:.2f}%）・聞き返し {k['over']['mean']:.3f}・通常 {k['normal']['mean']:.3f}・対象を断った {k['known_refused_n']['mean']:.1f}", flush=True)
        for cname in [x for x in keep if not x.endswith("平均")]:
            k, d, v, _ = keep[cname]
            vv = verdict({**k, "e2e": {"mean": v["e2e_value"]}, "severe": {"mean": v["severe_value"]}}, n_rows, n_known)
            res["verdict"][f"{n}|{cname}"] = vv
            ok = all(x["point"] for x in vv)
            md += [f"\n「{cname}」と目標（層 {n}。EtoE と重大な誤りは値まで見たもの）:\n", "|KPI|目標|値|目標を満たすか|区間（行数）|区間ごと内側か|", "|---|---|---|---|---|---|"]
            md += [f"|{x['kpi']}|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|" for x in vv]
            md.append(f"\n**層 {n}・{cname}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}**")
            print(f"  {cname[:24]} と目標: " + " / ".join(f"{x['kpi']} {x['value']:.4f}{'○' if x['point'] else '×'}" for x in vv) + f" → {'すべて満たす' if ok else '同時には満たさない'}")
        md += [f"\n止めた段（対象の依頼と雑談 {n_known} 行。1 回あたりの行数。「断った（範囲外）」は P31 で機能の定義で説明がつかず断った行）:\n",
               "|構成|規則|引数が足りない|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|断った（範囲外）|", "|---|---|---|---|---|---|---|---|"]
        for cname, (k, d, v, _) in keep.items():
            cnt = collections.Counter()
            for x in d["stop_rows"]:
                cnt[x["stop"]] += x["share"]
            vals = [cnt[s_] for s_ in STOPS] + [cnt["拒否（検査）"]]
            res["stops"][f"{n}|{cname}"] = vals
            md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in vals) + "|")
        md += ["\n値まで見た重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|違う値での実行|", "|---|" + "---|" * (len(CAT) + 1)]
        for cname, (k, d, v, _) in keep.items():
            cnt = collections.Counter()
            for x in d["severe_rows"]:
                cnt[x["category"]] += x["share"]
            res["severe"][f"{n}|{cname}"] = {**dict(cnt), "wrong_value": v["wrong_exec"]}
            md.append(f"|{cname}|" + "|".join(f"{cnt[k_]:.1f}" for k_ in CAT) + f"|{v['wrong_exec']:.1f}|")
        # 機能にない依頼の形ごと・種類ごと（本命・比べる相手 1・2）
        md += [f"\n種類ごとの扱い（層 {n}。正解／回復可能／通常／重大。機能にない依頼は 断る＝正解・聞き返す＝回復可能・読み出しで実行＝通常・操作で実行＝重大）:\n",
               "|種類|行|" + "|".join(cn for cn in list(keep)[::2] if True) + "|", "|---|---|" + "---|" * len(list(keep)[::2])]
        show = [cn for cn in keep if not cn.endswith("平均")]
        md[-2] = "|種類|行|" + "|".join(show) + "|"
        md[-1] = "|---|---|" + "---|" * len(show)
        gsets = {}
        for cname in show:
            src, make, mrules, sd = keep[cname][3]
            k2, d2 = measure(n, src, ts, loader_f, make, mrules, sd)
            g = collections.Counter()
            for x in d2["by_group"]:
                key = {"unseen操作": "H 学習に無い機能・操作", "unseen問い合わせ": "H 学習に無い機能・問い合わせ", "seen": "H 学習にある機能"}.get(x["group"], x["category"]) if x["category"] == "H" else x["category"]
                g[(key, x["grade"])] += x["rows"]
            gsets[cname] = g
        res["forms"][str(n)] = {cn: {f"{a}|{b}": w for (a, b), w in g.items()} for cn, g in gsets.items()}
        for key, nrow in (("H 学習にある機能", 70), ("H 学習に無い機能・操作", 40), ("H 学習に無い機能・問い合わせ", 40), ("G", 100), ("K", 120), ("N", 180), ("F", 50)):
            md.append(f"|{key}|{nrow}|" + "|".join("／".join(f"{gsets[cn][(key, g)]:.1f}" for g in GRADES) for cn in show) + "|")
        print("  学習に無い機能の扱い（本命・正解／回復可能／通常／重大）: 操作 " + "／".join(f"{gsets[primary][('H 学習に無い機能・操作', g)]:.1f}" for g in GRADES)
              + "・問い合わせ " + "／".join(f"{gsets[primary][('H 学習に無い機能・問い合わせ', g)]:.1f}" for g in GRADES))
        d = keep[primary][1]
        res["detail"][str(n)] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows", "by_group", "stages")}
        md += [f"\n本命で見るもの（層 {n}）:\n", "|見るもの|行（出た割合）|", "|---|---|",
               "|重大な誤り（ツールの取り違え）|" + (" / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"]) or "なし") + "|",
               "|違う値での実行|" + (" / ".join(f"{x['utterance']} → {x['got']}（正解 {x['label']}・{x['share']:.2f}）" for x in keep[primary][2]["rows"]) or "なし") + "|",
               "|通常の誤り|" + (" / ".join(f"[{y['category']}]{y['utterance']} → {y['outcome']}（{y['share']:.2f}）" for y in d["normal_rows"]) or "なし") + "|",
               "|止めた・断った対象の行|" + (" / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"]) or "なし") + "|"]
        print("  本命の重大な誤り: " + (" / ".join(f"[{y['category']}]{y['utterance']}→{y['executed']}" for y in d["severe_rows"]) or "なし"))
        print("  本命の違う値: " + (" / ".join(f"{x['utterance']}→{x['got']}" for x in keep[primary][2]["rows"]) or "なし"))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "verdict.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（verdict.md・verdict.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "text", "judge"])
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in BERT_SEEDS))
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P32 {a.step} =====", flush=True)
    {"encode": lambda: encode(), "device": lambda: device(ns), "text": lambda: text(a.device, tuple(int(x) for x in a.seeds.split(",") if x)), "judge": lambda: judge(ns)}[a.step]()


if __name__ == "__main__":
    main()
