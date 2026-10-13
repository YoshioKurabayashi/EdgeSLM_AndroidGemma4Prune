# -*- coding: utf-8 -*-
"""P27 手順 2: 歯止め（P25・P26）を入れた流れ全体の合否を、新しい評価セット（challenge_v7・600 行）で測る。

設計: DoC/設計書_P27_流れ全体を新しい行で測る.md の 4 節 ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md ｜ 測る前の記録: results/P27/preregistration.md

**本命（測る前に決めた）= P21 の規則 ＋ P26 の引数（歯止め G1〜G10）・端末に載せる BERT 1 つ。比べる相手 = P23 の引数（いまのアプリ）。**
ヘッドも BERT も学習し直さない（ヘッドは 1,584 行・seed 3 つ、BERT は P20 で保存した案 F）。
合否は**値まで見た KPI**（違う値での実行を、EtoE の成功から引き、重大な誤りに足す。利用者の決定・2026-10-07）で見る。

|段|何をするか|venv|
|---|---|---|
|`encode`|評価セット v7 をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、評価セット v7 を層 17・26 のグラフに流し、`h` を集める（途中で止めても済んだ分は残る）|EdgeSLM_Android|
|`text`|保存した BERT（案 F × seed 3 つ）で、評価セット v7 の確率を出す（学習はしない。評価セット v6 の行で P21 のときと同じ確率かを確かめる）|EdgeSLM_MultiHead_Multi（GPU。無ければ CPU）|
|`judge`|本命と比べる相手の KPI（値まで見たもの）・合否・止めた段・重大な誤りの内訳・歯止めで変わった行・部品ごとの精度。numpy と PyYAML だけ|どちらでも|

    python -m app.p27_verdict encode
    python -m app.p27_verdict device --layers 17,26
    python -m app.p27_verdict text
    python -m app.p27_verdict judge --layers 17,26      # results/P27/verdict.md・verdict.json
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p5_retrain import P3
from .p7_data import TR2, read_jsonl
from .p11_bert import MODES, by_mode
from .p13_eval import cell, interval
from .p16_parts import J, ORDER, ms
from .p18_data import md5_of
from .p18_retrain import SRC1584, bert_part, cached, head_part, verdict
from .p19_config import bert_check
from .p21_data import CH6, NOTES
from .p21_fix import measure
from .p21_rules import make_rules as rules_p21
from .p21_verdict import KPI, STOPS, predict_saved, share_by
from .p21_verdict import WORK as P21
from .p22_fix import show, value_kpi
from .p22_labels import match
from .p23_arguments import make_resolve as args_p23
from .p25_arguments import make_resolve as args_p25
from .p26_arguments import make_resolve as args_p26
from .p27_data import CH7, labeled

WORK = PROJECT_ROOT / "work" / "p27"
RES = PROJECT_ROOT / "results" / "P27"
T = 64
BERT_SEEDS = (0, 1, 2)
# 測る前に固定したもの（results/P27/preregistration.md）。変わっていたら結果に書く
FIXED = {"p21_rules.py": "8d74b163767c81df036d74142d9fc7ad", "p23_arguments.py": "f3e6afe7c86c98a191115a505b37f681",
         "p25_arguments.py": "00c7ac1fd843664a364c765b1b5ab73d", "p26_arguments.py": "d88f09926c802fa8f945a31c3bfc4b64"}
# （名前, BERT の seed の選び方, 引数）。規則はどれも P21・棄却の閾値は validation の 2%
CONFIGS = [("P26 の引数（本命）", "one", args_p26),
           ("比べる相手: P23 の引数（いまのアプリ）", "one", args_p23),
           ("P25 の引数（歯止め 1 段目）", "one", args_p25),
           ("P26 の引数・残った seed の平均", "ok", args_p26)]
PRIMARY, BEFORE = CONFIGS[0][0], CONFIGS[1][0]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}
VKPI = [("違う値での実行（行）", "wrong_exec", 1), ("値まで見た EtoE の成功", "e2e_value", 3), ("値まで見た重大な誤り（件）", "severe_value_n", 1),
        ("値まで見た重大な誤りの割合", "severe_value", 4), ("値まで見た 1 回目で正しく処理", "first_value", 3)]


def layers_of(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", ",").split(",") if x]


# ---------------------------------------------------------------- encode・device（P21 と同じ手順。行と保存先だけ違う）

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
    rows = read_jsonl(CH7 / "utterances.jsonl")
    ids = {r["id"]: enc(r) for r in rows}
    over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
    if over:
        raise SystemExit(f"{T} トークンを超える行がある: {over}")
    ln = [len(v) for v in ids.values()]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids_eval.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"評価セット v7: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)}（グラフは {T} まで）/ md5 {md5_of(CH7 / 'utterances.jsonl')}")


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
            res = D.run_job({**D.base_job(f"p27_eval_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
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


def load_eval7(n: int):
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH7 / "utterances.jsonl") if r["id"] in got]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


# ---------------------------------------------------------------- text（保存したモデルで確率を出すだけ。学習しない）

def text(device_name: str, seeds: tuple) -> None:
    src = P21 / "text_scores_F.json"
    if not src.exists():
        raise SystemExit(f"P21 の BERT の結果が無い（{src}。先に scripts\\run_p21.ps1）")
    raw = json.loads(src.read_text(encoding="utf-8"))
    if device_name == "cuda":
        import torch
        if not torch.cuda.is_available():
            device_name = "cpu"
            print("GPU が使えないので CPU で出す（学習はしないので数分で終わる）", flush=True)
    rows6, rows7 = read_jsonl(CH6 / "utterances.jsonl"), read_jsonl(CH7 / "utterances.jsonl")
    out = {**raw, "check": {}, "source": "P21 の結果に、保存したモデルで出した評価セット v7 の確率を足したもの"}
    print(f"BERT 案 F（P20 で保存したモデル）: 確率を出す行 {len(rows7)}（評価セット v7）＋ 確かめ {len(rows6)}（評価セット v6）/ {device_name}", flush=True)
    for s in seeds:
        name = f"bert_s{s}"
        save_dir = PROJECT_ROOT / "work" / "p20" / "bert_F" / name
        if name not in raw["models"] or not (save_dir / "config.json").exists():
            raise SystemExit(f"{name}: P20 で保存したモデルが無い（{save_dir}）")
        t0 = time.time()
        pred = predict_saved(save_dir, [r["utterance"] for r in rows6 + rows7], raw["bert"]["max_length"], device_name)
        prev = raw["models"][name]["pred"]
        diff = [abs(float(p) - prev[r["id"]]) for r, p in zip(rows6, pred) if r["id"] in prev]
        if not diff:
            raise SystemExit(f"{name}: P21 の結果に評価セット v6 の確率が無い（確かめられない）")
        out["models"][name] = {"cv": raw["models"][name]["cv"], "pred": {**prev, **{r["id"]: float(p) for r, p in zip(rows7, pred[len(rows6):])}}}
        out["check"][name] = {"rows": len(diff), "max_abs_diff": float(max(diff))}
        print(f"{name}: {time.time() - t0:.0f} 秒 / P21 のときの確率との差の最大 {max(diff):.4f}（評価セット v6 の {len(diff)} 行）", flush=True)
        if max(diff) > 1e-3:
            raise SystemExit(f"{name}: 保存したモデルが P21 のときと違う確率を出す（差 {max(diff):.4f}）。モデルが入れ替わっていないか確かめる")
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / "text_scores_F.json"
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge（numpy と PyYAML だけ）

def judge(ns: list[int]) -> None:
    f = WORK / "text_scores_F.json"
    if not f.exists():
        raise SystemExit("BERT の結果が無い（先に text）")
    raw = json.loads(f.read_text(encoding="utf-8"))
    rows = read_jsonl(CH7 / "utterances.jsonl")
    for name, m in raw["models"].items():
        miss = [r["id"] for r in rows if r["id"] not in m["pred"]]
        if miss:
            raise SystemExit(f"{name}: BERT の確率が無い行がある（{len(miss)} 行。評価セット v7 を変えたなら、text をもう一度）")
    use_multi_library()
    from edge_router.dispatch import arguments
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    lab = labeled(rows)
    res_fn = {name: make(arguments) for name, _, make in CONFIGS}
    chk = bert_check(raw)
    ts = {"ok": by_mode(raw, MODES[0], only=tuple(chk["ok"])), "one": by_mode(raw, MODES[0], only=(chk["one"],)), "all": by_mode(raw, MODES[0])}
    effect = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    loader = cached(load_eval7)
    n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
    c = {k: sum(r["category"] == k for r in rows) for k in NOTES}
    users = sum(r.get("author") == "user" for r in rows)
    here = PROJECT_ROOT / "app"
    sums = {"評価セット v7": md5_of(CH7 / "utterances.jsonl"), **{k: md5_of(here / k) for k in FIXED}}
    changed = [k for k, v in FIXED.items() if sums[k] != v]
    head = f"評価セット v7（{n_rows} 行 = " + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。利用者が足した行 {users}。正解の引数を付けた操作系の対象 {len(lab)} 行）"
    print(f"\n===== P27 judge: {head} =====")
    md = [f"{head}。**あらかじめ決めた本命は「{PRIMARY}」**。数字はヘッドの seed 3 つ（× BERT の seed）の平均（かっこは最小〜最大）。信頼区間は Wilson・95%。"
          "**合否は値まで見た KPI**（違う値での実行を EtoE の成功から引き、重大な誤りに足す）\n",
          "測ったもの（md5。改行が LF のとき）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "\n",
          ("**測る前に固定したファイルが変わっている: " + "・".join(changed) + "**\n") if changed else "測る前に固定した 4 つのファイル（規則・引数）は、記録した md5 と同じ（変えていない）\n"]
    print("\n".join(md[1:]).strip())
    res = {"md5": sums, "fixed_changed": changed, "bert_check": chk, "bert_diff": raw.get("check", {}), "kpi": {}, "verdict": {}, "stops": {}, "severe": {},
           "wrong_rows": {}, "guard": {}, "detail": {}, "bert": {}, "head": {}}
    line = "・".join(f"{k} {v['max_abs_diff']:.4f}" for k, v in raw.get("check", {}).items())
    md += [f"確かめ: 保存した BERT（案 F）が出す確率と、P21 のときの確率の差の最大（評価セット v6 の行・seed ごと）: {line}\n",
           "BERT の使うモデル（P19 の決まり）: " + "・".join(f"{k} {'使う' if v['ok'] else '使わない'}" for k, v in chk["seeds"].items()) + f"。端末に載せる 1 つ = {chk['one']}\n"]
    print(md[-2].strip() + "\n" + md[-1].strip())

    for n in ns:
        keep = {}
        for cname, which, mr in CONFIGS:
            got = measure(n, ts[which], loader, mr, rules_p21)
            if got is not None:
                k, d = got
                keep[cname] = (k, d, value_kpi(k, d, lab, res_fn[cname], state, n_rows))
        if not keep:
            print(f"層 {n}: 測れない（端末の h かヘッドが無い）")
            continue
        print(f"\n===== 層 {n} 個 =====")
        heads_ = [h for h, _, _ in KPI] + [h for h, _, _ in VKPI] + ["値まで見た EtoE の区間", "値まで見た重大な誤りの区間"]
        md += [f"\n## 層 {n} 個\n", "|構成|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
        for cname, (k, d, v) in keep.items():
            cells = [cell(k[key], nd) for _, key, nd in KPI] + [f"{v[key]:.{nd}f}" for _, key, nd in VKPI] + \
                    [interval(v["e2e_value"], n_rows), interval(v["severe_value"], n_rows)]
            res["kpi"][f"{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, **{key: v[key] for _, key, _ in VKPI}, "e2e_value_ci": cells[-2], "severe_value_ci": cells[-1]}
            mark = "**" if cname == PRIMARY else ""
            print(f"{cname[:20]:<20} " + "  ".join(f"{h[:12]}: {x}" for h, x in zip(heads_, cells)))
            md.append(f"|{mark}{cname}{mark}|" + "|".join(cells) + "|")
        for cname in (PRIMARY, BEFORE):
            if cname not in keep:
                continue
            k, d, v = keep[cname]
            kv = {**k, "e2e": {"mean": v["e2e_value"]}, "severe": {"mean": v["severe_value"]}}
            vv = verdict(kv, n_rows, n_known)
            res["verdict"][f"{n}|{cname}"] = vv
            md += [f"\n「{cname}」を検証の目標と比べる（層 {n} 個・値まで見た KPI）:\n", "|KPI|目標|値|目標を満たすか（平均の値）|区間（行数）|区間ごと目標の内側か|", "|---|---|---|---|---|---|"]
            for x in vv:
                md.append(f"|{x['kpi']}（値まで見た）|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|"
                          if x["key"] in ("e2e", "severe") else
                          f"|{x['kpi']}|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|")
            ok = all(x["point"] for x in vv)
            print(f"  {cname[:14]} と目標: " + " / ".join(f"{x['kpi']} {x['value']:.4f}（{x['target']}）{'○' if x['point'] else '×'}" for x in vv) + f" → {'すべて満たす' if ok else '同時には満たさない'}")
            md.append(f"\n**層 {n}・{cname}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}（平均の値）**")

        md += [f"\n対象の依頼と雑談 {n_known} 行を止めた段（1 回あたりの行数）:\n", "|構成|規則|引数が足りない（歯止めを含む）|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|検査|合計|", "|---|---|---|---|---|---|---|---|---|"]
        for cname, (k, d, v) in keep.items():
            cnt = collections.Counter()
            for x in d["stop_rows"]:
                cnt[x["stop"]] += x["share"]
            vals = [cnt[s_] for s_ in STOPS] + [cnt["聞き返し（検査）"] + cnt["拒否（検査）"]]
            res["stops"][f"{n}|{cname}"] = vals
            md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in vals) + f"|{sum(vals):.1f}|")
            print(f"  {cname[:14]:<14} 止めた段: " + "・".join(f"{s_.replace('聞き返し', '').replace('拒否', '')} {x:.1f}" for s_, x in zip(STOPS + ['検査'], vals)))
        md += ["\n値まで見た重大な誤りの内訳（行数。違う値での実行は、正しい機能を選んだ操作系の対象の行）:\n", "|構成|" + "|".join(CAT.values()) + "|違う値での実行|合計|", "|---|" + "---|" * (len(CAT) + 2)]
        for cname, (k, d, v) in keep.items():
            cnt = collections.Counter()
            for x in d["severe_rows"]:
                cnt[x["category"]] += x["share"]
            res["severe"][f"{n}|{cname}"] = {**dict(cnt), "wrong_value": v["wrong_exec"]}
            md.append(f"|{cname}|" + "|".join(f"{cnt[k_]:.1f}" for k_ in CAT) + f"|{v['wrong_exec']:.1f}|{sum(cnt.values()) + v['wrong_exec']:.1f}|")
            print(f"  {cname[:14]:<14} 重大: " + "・".join(f"{CAT[k_]} {cnt[k_]:.1f}" for k_ in CAT) + f"・違う値 {v['wrong_exec']:.1f}")
        md += ["\n違う値での実行（行。かっこは実行まで進んだ割合）:\n", "|構成|行|", "|---|---|"]
        for cname, (k, d, v) in keep.items():
            res["wrong_rows"][f"{n}|{cname}"] = v["rows"]
            md.append(f"|{cname}|" + (" / ".join(f"{x['utterance']} → {x['got']}（正解 {x['label']}・{x['share']:.2f}）" for x in v["rows"]) or "なし") + "|")

        # ---- 歯止めで変わった行（P23 → P26）
        b, d = keep[BEFORE][1], keep[PRIMARY][1]
        ra = ("聞き返し（引数が足りない）",)
        sb, sa = share_by(b["stop_rows"], ra), share_by(d["stop_rows"], ra)
        by_utt = {x["utterance"]: x for x in lab}
        newly = []
        for u in sa:
            s_ = sa[u] - sb[u]
            if s_ > 1e-9:
                x = by_utt.get(u)
                if x is None:
                    was = "正解の引数なし"
                else:
                    y = res_fn[BEFORE](x["tool"], None, u, state)
                    was = "P23 で合っていた" if match(x["tool"], x["expected"], y.complete, y.values) else f"P23 で違っていた（{show(y)}・正解 {x['label']}）"
                z = res_fn[PRIMARY](x["tool"], None, u, state) if x else None
                newly.append({"utterance": u, "share": round(s_, 2), "was": was, "why": (z.notes[0] if z is not None and z.notes else "")})
        freed = [[u, round(sb[u] - sa[u], 2)] for u in sb if sb[u] - sa[u] > 1e-9]
        vb, va = share_by(b["severe_rows"]), share_by(d["severe_rows"])
        fixed = [[u, round(vb[u] - va[u], 2)] for u in vb if vb[u] - va[u] > 1e-9]
        worse = [[u, round(va[u] - vb[u], 2)] for u in va if va[u] - vb[u] > 1e-9]
        wb = {x["utterance"]: x["share"] for x in keep[BEFORE][2]["rows"]}
        wa = {x["utterance"]: x["share"] for x in keep[PRIMARY][2]["rows"]}
        wfix = [[u, s_] for u, s_ in wb.items() if s_ - wa.get(u, 0) > 1e-9]
        res["guard"][str(n)] = {"newly_stopped": newly, "freed": freed, "severe_fixed": fixed, "severe_new": worse, "wrong_value_fixed": wfix}
        ok_lost = sum(x["share"] for x in newly if x["was"] == "P23 で合っていた")
        bad_stop = sum(x["share"] for x in newly if x["was"].startswith("P23 で違っていた"))
        fmt = lambda items: " / ".join(f"{u}（{s:.2f}）" for u, s in items) if items else "なし"
        md += [f"\n歯止めで変わった行（P23 → P26。層 {n} 個。かっこは変わった割合）:\n", "|見るもの|行|", "|---|---|",
               f"|引数の段で新しく止めた対象の行（{sum(x['share'] for x in newly):.1f}。うち P23 で値が合っていた {ok_lost:.1f}・違っていた {bad_stop:.1f}）|"
               + (" / ".join(f"{x['utterance']}（{x['share']:.2f}・{x['was']}・{x['why'].replace(' → 聞き返す', '')}）" for x in newly) or "なし") + "|",
               f"|違う値での実行でなくなった行（{sum(s for _, s in wfix):.1f}）|{fmt(wfix)}|",
               f"|ツールの取り違えの重大な誤りでなくなった行（{sum(s for _, s in fixed):.1f}）|{fmt(fixed)}|",
               f"|新しく重大な誤りになった行（{sum(s for _, s in worse):.1f}）|{fmt(worse)}|",
               f"|引数の段で止まらなくなった行（{sum(s for _, s in freed):.1f}）|{fmt(freed)}|"]
        print(f"  歯止めで新しく止めた対象の行 {sum(x['share'] for x in newly):.1f}（P23 で合っていた {ok_lost:.1f}・違っていた {bad_stop:.1f}）/ 違う値でなくなった {sum(s for _, s in wfix):.1f} / 取り違えの重大でなくなった {sum(s for _, s in fixed):.1f}")

        # ---- 本命の内訳
        d = keep[PRIMARY][1]
        md += [f"\n本命の種類ごとの内訳（層 {n} 個。割合。違う値での実行は「正解」に入っている＝値まで見ない数え方）:\n", "|種類|行|正解／回復可能／通常／重大|", "|---|---|---|"]
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
        res["detail"][str(n)] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows", "by_group", "stages")}
        md += [f"\n本命で見るもの（層 {n} 個）:\n", "|見るもの|行（出た割合）|", "|---|---|",
               "|重大な誤り（ツールの取り違え）|" + (" / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:30]) or "なし") + "|",
               "|通常の誤り|" + (" / ".join(f"[{y['category']}]{y['utterance']} → {y['outcome']}（{y['share']:.2f}）" for y in d["normal_rows"][:30]) or "なし") + "|",
               "|止めた対象の行|" + (" / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:80]) or "なし") + "|"]
        print("  本命で残る重大な誤り: " + " / ".join(f"[{y['category']}]{y['utterance']}→{y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:12]))
        print("  本命の違う値での実行: " + (" / ".join(f"{x['utterance']}→{x['got']}（{x['share']:.2f}）" for x in keep[PRIMARY][2]["rows"]) or "なし"))

    # ---- 部品ごと（評価セット v7）
    md.append("\n## 部品ごとの精度（評価セット v7。規則・引数は外す）\n")
    bp = bert_part(ts["all"], rows)
    labels = [("断る閾値", "hi"), ("機能にない依頼にフラグ以上（再現率）", "H_flag"), ("　学習に同じ機能がある", "H_flag_seen"), ("　どの学習にも無い機能", "H_flag_unseen"),
              ("機能にない依頼を断る閾値以上", "H_refuse"), ("対象の依頼と雑談に誤ってフラグ", "known_flag"), ("　機能の語を含む依頼", "K_flag"), ("　ふつうの依頼", "N_flag"), ("　雑談", "F_flag"),
              ("対象を断った（行）", "known_refuse_n"), ("意図が不明にフラグ", "G_flag"), ("AUC（機能にない依頼 ↔ 対象の依頼と雑談）", "auc_known")]
    names = sorted(k for k in ts["all"]["models"])
    print("\n===== 部品: BERT 案 F（評価セット v7・seed ごと）=====")
    md += ["### BERT 案 F（seed ごと。フラグ = 0.5 以上・断る = 学習側の 99.5% 点以上）\n", "|指標|" + "|".join(f"{nm}" + ("（載せる 1 つ）" if nm == chk["one"] else "" if chk["seeds"][nm]["ok"] else "（使わない）") for nm in names) + "|",
           "|---|" + "---|" * len(names)]
    for lab_, key in labels:
        cells = [f"{x:.3f}" if isinstance(x, float) else str(x) for x in bp.get(key, [])]
        print(f"{lab_:<26} " + "  ".join(cells))
        md.append(f"|{lab_}|" + "|".join(cells) + "|")
    res["bert"] = bp
    for n in ns:
        loaded = loader(n)
        if loaded is None:
            continue
        rs, last, mean = loaded
        hp = head_part(n, SRC1584, rs, last, mean, effect)
        if not hp:
            continue
        hl = [("行き先の正解率（対象の依頼と雑談）", "acc", 3), ("　機能の語を含む依頼", "acc_K", 3), ("　ふつうの依頼", "acc_N", 3), ("　雑談", "acc_F", 3)] + \
             [(f"　再現率: {J[c_]}", f"recall_{J[c_]}", 3) for c_ in ORDER] + \
             [("対象の依頼と雑談を通した割合", "cover", 3), ("通したうち、別の操作系と判定（行）", "wrong_write", 1),
              ("機能にない依頼を止めた（棄却・確信度）", "H_stop", 3), ("　操作系として通した", "H_pass_write", 3), ("　棄却の得点の AUC", "auc_H", 3),
              ("意図が不明を止めた（棄却・確信度）", "G_stop", 3), ("　操作系として通した", "G_pass_write", 3), ("　棄却の得点の AUC", "auc_G", 3)]
        print(f"\n===== 部品: LM ヘッド 1,584 行・層 {n} 個（評価セット v7）=====")
        md += [f"\n### LM ヘッド 1,584 行・層 {n} 個（mlp・seed 3 つ。棄却の閾値は validation の 2%）\n", "|指標|値|", "|---|---|"]
        for lab_, key, nd in hl:
            print(f"{lab_:<28} {ms(hp[key], nd)}")
            md.append(f"|{lab_}|{ms(hp[key], nd)}|")
        md.append("\n取り違えの多い組（正解 → 判定。1 つのヘッドあたりの行数）: " + " / ".join(f"{x['true']} → {x['pred']} {x['rows']:.1f}" for x in hp["confusions"]))
        res["head"][str(n)] = hp
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "verdict.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
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
    print(f"\n===== P27 {a.step} =====", flush=True)
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
