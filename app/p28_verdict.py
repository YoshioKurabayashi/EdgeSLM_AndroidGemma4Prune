# -*- coding: utf-8 -*-
"""P28 手順 2: 空調 ↔ メディアの区別を上げる行（train_add_v4・72 行）を足してヘッドを学習し直し、新しい評価セット（challenge_v8・600 行）で流れ全体の合否を測る。

設計: DoC/設計書_P28_空調とメディアの区別を学習で上げる.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md ｜ 測る前の記録: results/P28/preregistration.md

**本命（測る前に決めた）= ヘッド 1,656 行（P28）・P21 の規則・P26 の引数・端末に載せる BERT 1 つ。比べる相手 = ヘッド 1,584 行（P18）・同じ規則と引数。**
BERT（P20 で保存した案 F）・規則・引数・閾値の決め方は変えない（棄却 = validation の行き先のある行の 2%）。変えるのはヘッドだけ。
合否は値まで見た KPI（P27 と同じ）。評価セット v7・v6 は直す用（参考に、ヘッドだけ替えた差を出す）。

|段|何をするか|venv|
|---|---|---|
|`encode`|足す 72 行と評価セット v8 の 600 行をトークン列にする（714 行と同じ符号化かを確かめる）|EdgeSLM_Android|
|`device`|お試しアプリで、上の行を層 17・26 のグラフに流し、`h` を集める（途中で止めても済んだ分は残る）|EdgeSLM_Android|
|`heads`|P18 の 1,584 行の `h` に足す 72 行の `h` を繋ぎ、ヘッドを学習・評価（EdgeSLM_MultiHead_Multi のコードをそのまま呼ぶ）|EdgeSLM_MultiHead_Multi|
|`text`|保存した BERT で評価セット v8 の確率を出す（学習はしない。評価セット v7 の行で P27 のときと同じ確率かを確かめる）|EdgeSLM_MultiHead_Multi|
|`judge`|評価セット v8（合否）と v7・v6（直す用）で、ヘッド 1,584 行 ↔ 1,656 行の KPI（値まで見たもの）・合否・止めた段・変わった行・部品ごとの精度|どちらでも|

    python -m app.p28_verdict encode
    python -m app.p28_verdict device --layers 17,26
    python -m app.p28_verdict heads --layers 17,26
    python -m app.p28_verdict text
    python -m app.p28_verdict judge --layers 17,26      # results/P28/verdict.md・verdict.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time

import numpy as np

from .common import HF_DIR, MULTI_ROOT, PROJECT_ROOT, use_multi_library
from . import p18_retrain as R
from .p5_retrain import P3
from .p6_seeds import head_probs, load_pt
from .p7_data import TR2, read_jsonl
from .p8_gate import STEPS, run
from .p10_reject import quantile
from .p11_bert import MODES, by_mode
from .p13_eval import cell, hi_rule, interval
from .p16_parts import J, ORDER, ms
from .p18_data import md5_of
from .p18_retrain import HEADS, SRC1584, bert_part, cached, has_heads, head_part, verdict
from .p18_retrain import WORK as P18
from .p19_config import bert_check
from .p21_data import CH6, NOTES
from .p21_rules import make_rules as rules_p21
from .p21_verdict import KPI, STOPS, load_eval6, predict_saved, share_by
from .p22_fix import value_kpi
from .p22_labels import load_all
from .p23_arguments import make_resolve as args_p23
from .p26_arguments import make_resolve as args_p26
from .p27_data import CH7, labeled as labeled7
from .p27_verdict import WORK as P27, load_eval7
from .p28_data import CH8, TR4, labeled as labeled8

WORK = PROJECT_ROOT / "work" / "p28"
RES = PROJECT_ROOT / "results" / "P28"
T = 64
BERT_SEEDS = (0, 1, 2)
SRC1656 = "1,656 行（P28）"
HEADS[SRC1656] = lambda n: WORK / f"L{n:02d}" / "heads"
FIXED = {"p21_rules.py": "8d74b163767c81df036d74142d9fc7ad", "p23_arguments.py": "f3e6afe7c86c98a191115a505b37f681", "p26_arguments.py": "d88f09926c802fa8f945a31c3bfc4b64"}
# （名前, ヘッド, BERT の seed の選び方, 引数）。規則はどれも P21
CONFIGS = [("1,656 行のヘッド・P26 の引数（本命）", SRC1656, "one", args_p26),
           ("比べる相手: 1,584 行のヘッド・P26 の引数", SRC1584, "one", args_p26),
           ("いまのアプリ: 1,584 行のヘッド・P23 の引数", SRC1584, "one", args_p23),
           ("1,656 行のヘッド・P26・残った seed の平均", SRC1656, "ok", args_p26)]
PRIMARY, BEFORE = CONFIGS[0][0], CONFIGS[1][0]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}
VKPI = [("違う値での実行（行）", "wrong_exec", 1), ("値まで見た EtoE の成功", "e2e_value", 3), ("値まで見た重大な誤り（件）", "severe_value_n", 1),
        ("値まで見た重大な誤りの割合", "severe_value", 4)]


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
    for kind, rows in (("train", read_jsonl(TR4 / "added.jsonl")), ("eval", read_jsonl(CH8 / "utterances.jsonl"))):
        ids = {r["id"]: enc(r) for r in rows}
        over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
        if over:
            raise SystemExit(f"{kind}: {T} トークンを超える行がある: {over}")
        ln = [len(v) for v in ids.values()]
        (WORK / f"token_ids_{kind}.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"{kind}: {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最大 {max(ln)}（グラフは {T} まで）")
    print(f"md5: 学習 1,656 行 {md5_of(TR4 / 'utterances.jsonl')}・評価セット v8 {md5_of(CH8 / 'utterances.jsonl')}")


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
                pf = parts / f"{kind}_L{n}_{k:02d}.json"
                if pf.exists():
                    rows = json.loads(pf.read_text(encoding="utf-8"))["rows"]
                    if [r["id"] for r in rows] == part:
                        got += rows
                        continue
                res = D.run_job({**D.base_job(f"p28_{kind}_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
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


def load_eval8(n: int):
    f = WORK / f"device_eval_L{n}.json"
    if not f.exists():
        return None
    got = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in read_jsonl(CH8 / "utterances.jsonl") if r["id"] in got]
    return rows, np.array([got[r["id"]]["last"] for r in rows], np.float64), np.array([got[r["id"]]["mean"] for r in rows], np.float64)


# ---------------------------------------------------------------- heads

def features(n: int) -> None:
    src = P18 / f"L{n:02d}" / "features"
    z = np.load(src / "features_F2.npz")
    meta = json.loads((src / "features_F2.meta.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in read_jsonl(TR4 / "utterances.jsonl")]
    k = len(meta["ids"])
    if ids[:k] != meta["ids"]:
        raise SystemExit(f"層 {n}: 合わせた学習データの先頭 {k} 行が、P18 の特徴の並びと違う")
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
         "meta": {"source": f"device (Xperia 5 III, 層 {n} 個のまとめ読み T=64, QNN HTP)。先頭 {k} 行は P2・P5・P7・P18、残りは P28", "n_layers_kept": n}}
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
        print(f"\n########## 層 {n} 個（1,584 行＋足した 72 行）##########", flush=True)
        features(n)
        if (base / "heads" / "evaluate_report.json").exists() and not force:
            print(f"層 {n}: ヘッドは作成済み（作り直すなら --force）", flush=True)
            continue
        common = ["--data", str(TR4 / "utterances.jsonl"), "--features", str(base / "features")]
        call(train, common + ["--out", str(base / "heads"), "--force"])
        call(evaluate, common + ["--run", str(base / "heads")])
    for n in ns:
        for src in (SRC1584, SRC1656):
            if has_heads(n, src):
                t = summarize_evaluate(HEADS[src](n) / "evaluate_report.json").get("mlp_s7_test_accuracy")
                print(f"層 {n:>2}・{src:<14} test の正解率（mlp・seed 7）{'—' if t is None else f'{t:.4f}'}", flush=True)
    print("（1,584 行と 1,656 行は test の行が 18 行違うので、数字をそのまま比べない。比べるのは judge の評価セット v8）")


def hook1656(n: int, src: str, heads_: list) -> list:
    """1,656 行のヘッドの棄却の閾値を、validation の行き先のある行の 2% を誤って回す値にする（P10・P18 と同じ決め方）"""
    rows = read_jsonl(TR4 / "utterances.jsonl")
    d = WORK / f"L{n:02d}" / "features"
    ids = json.loads((d / "features_F2.meta.json").read_text(encoding="utf-8"))["ids"]
    if ids != [r["id"] for r in rows]:
        raise SystemExit(f"層 {n}: 1,656 行の特徴の並びがデータと違う")
    pick = [i for i, r in enumerate(rows) if r["split"] == "validation" and r["route"] != "UNKNOWN"]
    z = np.load(d / "features_F2.npz")
    last, mean = z["last__last"][pick].astype(np.float64), z["last__mean"][pick].astype(np.float64)
    hd = HEADS[src](n)
    reps = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
    out = []
    for (seed, classes, th, p) in heads_:
        name = f"heads_mlp_seed{seed}.pt"
        val = head_probs(load_pt(hd / name), float(reps[name]["temperature"]), last, mean)["reject"]
        out.append((seed, classes, {**th, "REJECT": quantile(val, 0.02)}, p))
    return out


_hook_for = R.hook_for
R.hook_for = lambda src: hook1656 if src == SRC1656 else _hook_for(src)        # head_part（部品ごとの精度）でも同じ閾値にする


def measure(n: int, src: str, ts: dict, loader, make_resolve):
    """`p21_fix.measure` と同じ（ヘッドを選べるようにしただけ）"""
    if not has_heads(n, src) or loader(n) is None:
        return None
    final = STEPS[-1]
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=R.hook_for(src), steps=[final], sources=[src], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=rules_p21, amb_chat_ok=True)
    key = f"{n}|{src}|{final[0]}"
    return blob["kpi"][key], blob["detail"][key]


# ---------------------------------------------------------------- text

def text(device_name: str, seeds: tuple) -> None:
    src = P27 / "text_scores_F.json"
    if not src.exists():
        raise SystemExit(f"P27 の BERT の結果が無い（{src}。先に scripts\\run_p27.ps1）")
    raw = json.loads(src.read_text(encoding="utf-8"))
    if device_name == "cuda":
        import torch
        if not torch.cuda.is_available():
            device_name = "cpu"
            print("GPU が使えないので CPU で出す（学習はしないので数分で終わる）", flush=True)
    rows7, rows8 = read_jsonl(CH7 / "utterances.jsonl"), read_jsonl(CH8 / "utterances.jsonl")
    out = {**raw, "check": {}, "source": "P27 の結果に、保存したモデルで出した評価セット v8 の確率を足したもの"}
    print(f"BERT 案 F（P20 で保存したモデル）: 確率を出す行 {len(rows8)}（評価セット v8）＋ 確かめ {len(rows7)}（評価セット v7）/ {device_name}", flush=True)
    for s in seeds:
        name = f"bert_s{s}"
        save_dir = PROJECT_ROOT / "work" / "p20" / "bert_F" / name
        if name not in raw["models"] or not (save_dir / "config.json").exists():
            raise SystemExit(f"{name}: P20 で保存したモデルが無い（{save_dir}）")
        t0 = time.time()
        pred = predict_saved(save_dir, [r["utterance"] for r in rows7 + rows8], raw["bert"]["max_length"], device_name)
        prev = raw["models"][name]["pred"]
        diff = [abs(float(p) - prev[r["id"]]) for r, p in zip(rows7, pred) if r["id"] in prev]
        if not diff:
            raise SystemExit(f"{name}: P27 の結果に評価セット v7 の確率が無い（確かめられない）")
        out["models"][name] = {"cv": raw["models"][name]["cv"], "pred": {**prev, **{r["id"]: float(p) for r, p in zip(rows8, pred[len(rows7):])}}}
        out["check"][name] = {"rows": len(diff), "max_abs_diff": float(max(diff))}
        print(f"{name}: {time.time() - t0:.0f} 秒 / P27 のときの確率との差の最大 {max(diff):.4f}（評価セット v7 の {len(diff)} 行）", flush=True)
        if max(diff) > 1e-3:
            raise SystemExit(f"{name}: 保存したモデルが P27 のときと違う確率を出す（差 {max(diff):.4f}）")
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / "text_scores_F.json"
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"保存しました: {f}", flush=True)


# ---------------------------------------------------------------- judge

def judge_set(tag: str, rows: list[dict], loader, lab: list[dict], ns: list[int], configs: list, ts: dict, res_fn: dict, state: dict,
              res: dict, md: list, full: bool) -> None:
    n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
    for n in ns:
        keep = {}
        for cname, src, which, make in configs:
            got = measure(n, src, ts[which], loader, make)
            if got is not None:
                k, d = got
                keep[cname] = (k, d, value_kpi(k, d, lab, res_fn[cname], state, n_rows))
        if PRIMARY not in keep or BEFORE not in keep:
            print(f"{tag}・層 {n}: 測れない（端末の h かヘッドが無い）")
            continue
        print(f"\n===== {tag}・層 {n} 個 =====")
        heads_ = [h for h, _, _ in KPI] + [h for h, _, _ in VKPI] + ["値まで見た EtoE の区間", "値まで見た重大な誤りの区間"]
        md += [f"\n## {tag}・層 {n} 個\n", "|構成|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
        for cname, (k, d, v) in keep.items():
            cells = [cell(k[key], nd) for _, key, nd in KPI] + [f"{v[key]:.{nd}f}" for _, key, nd in VKPI] + \
                    [interval(v["e2e_value"], n_rows), interval(v["severe_value"], n_rows)]
            res["kpi"][f"{tag}|{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, **{key: v[key] for _, key, _ in VKPI}, "wrong_rows": v["rows"]}
            mark = "**" if cname == PRIMARY else ""
            print(f"{cname[:22]:<22} " + "  ".join(f"{h[:10]}: {x}" for h, x in zip(heads_[:-2], cells[:-2]) if h[:2] in ('Et', '重大', '不要', '通常', '違う', '値ま')))
            md.append(f"|{mark}{cname}{mark}|" + "|".join(cells) + "|")
        if full:
            for cname in [c for c in (PRIMARY, BEFORE, CONFIGS[2][0]) if c in keep]:
                k, d, v = keep[cname]
                vv = verdict({**k, "e2e": {"mean": v["e2e_value"]}, "severe": {"mean": v["severe_value"]}}, n_rows, n_known)
                res["verdict"][f"{tag}|{n}|{cname}"] = vv
                md += [f"\n「{cname}」を検証の目標と比べる（{tag}・層 {n} 個。EtoE と重大な誤りは値まで見たもの）:\n",
                       "|KPI|目標|値|目標を満たすか（平均の値）|区間（行数）|区間ごと目標の内側か|", "|---|---|---|---|---|---|"]
                md += [f"|{x['kpi']}|{x['target']}|{x['value']:.4f}|{'○' if x['point'] else '×'}|{x['ci']}（{x['rows']} 行）|{'○' if x['sure'] else '×'}|" for x in vv]
                ok = all(x["point"] for x in vv)
                md.append(f"\n**{tag}・層 {n}・{cname}: 4 つの目標を{'すべて満たす' if ok else '同時には満たさない'}（平均の値）**")
                print(f"  {cname[:16]} と目標: " + " / ".join(f"{x['kpi']} {x['value']:.4f}{'○' if x['point'] else '×'}" for x in vv) + f" → {'すべて満たす' if ok else '同時には満たさない'}")
        md += [f"\n対象の依頼と雑談 {n_known} 行を止めた段（1 回あたりの行数）:\n", "|構成|規則|引数が足りない|棄却ヘッド|確信度|BERT のフラグ|BERT が断る|検査|合計|", "|---|---|---|---|---|---|---|---|---|"]
        for cname, (k, d, v) in keep.items():
            cnt = collections.Counter()
            for x in d["stop_rows"]:
                cnt[x["stop"]] += x["share"]
            vals = [cnt[s_] for s_ in STOPS] + [cnt["聞き返し（検査）"] + cnt["拒否（検査）"]]
            res["stops"][f"{tag}|{n}|{cname}"] = vals
            md.append(f"|{cname}|" + "|".join(f"{x:.1f}" for x in vals) + f"|{sum(vals):.1f}|")
            print(f"  {cname[:16]:<16} 止めた段: " + "・".join(f"{s_.replace('聞き返し', '').replace('拒否', '')} {x:.1f}" for s_, x in zip(STOPS + ['検査'], vals)))
        md += ["\n値まで見た重大な誤りの内訳（行数）:\n", "|構成|" + "|".join(CAT.values()) + "|違う値での実行|合計|", "|---|" + "---|" * (len(CAT) + 2)]
        for cname, (k, d, v) in keep.items():
            cnt = collections.Counter()
            for x in d["severe_rows"]:
                cnt[x["category"]] += x["share"]
            res["severe"][f"{tag}|{n}|{cname}"] = {**dict(cnt), "wrong_value": v["wrong_exec"]}
            md.append(f"|{cname}|" + "|".join(f"{cnt[k_]:.1f}" for k_ in CAT) + f"|{v['wrong_exec']:.1f}|{sum(cnt.values()) + v['wrong_exec']:.1f}|")
        # ---- ヘッドで変わった行（1,584 行 → 1,656 行。引数・規則・BERT は同じ）
        b, d = keep[BEFORE][1], keep[PRIMARY][1]
        fmt = lambda items: " / ".join(f"{u}（{s:.2f}）" for u, s in items) if items else "なし"
        diff = lambda xb, xa: [[u, round(xb[u] - xa[u], 2)] for u in xb if xb[u] - xa[u] > 1e-9]
        sb, sa = share_by(b["stop_rows"]), share_by(d["stop_rows"])
        vb, va = share_by(b["severe_rows"]), share_by(d["severe_rows"])
        nb, na = share_by(b["normal_rows"]), share_by(d["normal_rows"])
        ch = {"stop_freed": diff(sb, sa), "stop_new": diff(sa, sb), "severe_fixed": diff(vb, va), "severe_new": diff(va, vb), "normal_fixed": diff(nb, na), "normal_new": diff(na, nb)}
        res["changed"][f"{tag}|{n}"] = ch
        md += [f"\nヘッドで変わった行（1,584 行 → 1,656 行。{tag}・層 {n} 個。かっこは変わった割合）:\n", "|見るもの|行|", "|---|---|"]
        for lab_, key in (("止まらなくなった対象の行（不要な聞き返しが減った分）", "stop_freed"), ("新しく止まった対象の行", "stop_new"),
                          ("重大な誤りでなくなった行", "severe_fixed"), ("新しく重大な誤りになった行", "severe_new"),
                          ("通常の誤りでなくなった行", "normal_fixed"), ("新しく通常の誤りになった行", "normal_new")):
            md.append(f"|{lab_}（{sum(s for _, s in ch[key]):.1f}）|{fmt(ch[key])}|")
        print(f"  ヘッドで: 止まらなくなった {sum(s for _, s in ch['stop_freed']):.1f}・新しく止まった {sum(s for _, s in ch['stop_new']):.1f}・"
              f"重大でなくなった {sum(s for _, s in ch['severe_fixed']):.1f}・新しく重大 {sum(s for _, s in ch['severe_new']):.1f}・"
              f"通常でなくなった {sum(s for _, s in ch['normal_fixed']):.1f}・新しく通常 {sum(s for _, s in ch['normal_new']):.1f}")
        if full:
            d = keep[PRIMARY][1]
            res["detail"][f"{tag}|{n}"] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows", "by_group", "stages")}
            md += [f"\n本命で見るもの（{tag}・層 {n} 個）:\n", "|見るもの|行（出た割合）|", "|---|---|",
                   "|重大な誤り（ツールの取り違え）|" + (" / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:30]) or "なし") + "|",
                   "|違う値での実行|" + (" / ".join(f"{x['utterance']} → {x['got']}（正解 {x['label']}・{x['share']:.2f}）" for x in keep[PRIMARY][2]["rows"]) or "なし") + "|",
                   "|通常の誤り|" + (" / ".join(f"[{y['category']}]{y['utterance']} → {y['outcome']}（{y['share']:.2f}）" for y in d["normal_rows"][:30]) or "なし") + "|",
                   "|止めた対象の行|" + (" / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:80]) or "なし") + "|"]
            print("  本命で残る重大な誤り: " + (" / ".join(f"[{y['category']}]{y['utterance']}→{y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:10]) or "なし"))


def judge(ns: list[int]) -> None:
    f = WORK / "text_scores_F.json"
    if not f.exists():
        raise SystemExit("BERT の結果が無い（先に text）")
    raw = json.loads(f.read_text(encoding="utf-8"))
    rows8, rows7, rows6 = read_jsonl(CH8 / "utterances.jsonl"), read_jsonl(CH7 / "utterances.jsonl"), read_jsonl(CH6 / "utterances.jsonl")
    for name, m in raw["models"].items():
        miss = [r["id"] for r in rows8 + rows7 + rows6 if r["id"] not in m["pred"]]
        if miss:
            raise SystemExit(f"{name}: BERT の確率が無い行がある（{len(miss)} 行。評価セット v8 を変えたなら、text をもう一度）")
    use_multi_library()
    from edge_router.dispatch import arguments
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    res_fn = {name: make(arguments) for name, _, _, make in CONFIGS}
    chk = bert_check(raw)
    ts = {"ok": by_mode(raw, MODES[0], only=tuple(chk["ok"])), "one": by_mode(raw, MODES[0], only=(chk["one"],)), "all": by_mode(raw, MODES[0])}
    here = PROJECT_ROOT / "app"
    sums = {"学習 1,656 行": md5_of(TR4 / "utterances.jsonl"), "評価セット v8": md5_of(CH8 / "utterances.jsonl"), **{k: md5_of(here / k) for k in FIXED}}
    changed = [k for k, v in FIXED.items() if sums[k] != v]
    c = {k: sum(r["category"] == k for r in rows8) for k in NOTES}
    head = f"評価セット v8（{len(rows8)} 行 = " + "・".join(f"{NOTES[k]} {v}" for k, v in c.items()) + f"。正解の引数を付けた操作系の対象 {len(labeled8(rows8))} 行）"
    print(f"\n===== P28 judge: {head} =====")
    md = [f"# P28 ヘッドを学習し直した構成の合否（評価セット v8）\n", f"{head}。**あらかじめ決めた本命は「{PRIMARY}」**。数字はヘッドの seed 3 つの平均（かっこは最小〜最大）。"
          "区間は Wilson・95%。**合否は値まで見た KPI**（違う値での実行を EtoE の成功から引き、重大な誤りに足す）。評価セット v7・v6 は直す用（参考）\n",
          "測ったもの（md5。改行が LF のとき）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "\n",
          ("**測る前に固定したファイルが変わっている: " + "・".join(changed) + "**\n") if changed else "測る前に固定した規則・引数のファイルは、記録した md5 と同じ（変えていない）\n",
          "確かめ: 保存した BERT の確率と P27 のときの確率の差の最大（評価セット v7 の行）: " + "・".join(f"{k} {v['max_abs_diff']:.4f}" for k, v in raw.get("check", {}).items())
          + f"。端末に載せる BERT = {chk['one']}\n"]
    print("\n".join(md[2:]).strip())
    res = {"md5": sums, "fixed_changed": changed, "bert_check": chk, "kpi": {}, "verdict": {}, "stops": {}, "severe": {}, "changed": {}, "detail": {}, "head": {}, "bert": {}}
    judge_set("評価セット v8", rows8, cached(load_eval8), labeled8(rows8), ns, CONFIGS, ts, res_fn, state, res, md, True)
    md.append("\n# 参考: 直す用の評価セット（足す行は v7 の誤りを見て決めたので、v7 の数字は甘く出る）\n")
    judge_set("評価セット v7（直す用）", rows7, cached(load_eval7), labeled7(rows7), ns, CONFIGS[:2], ts, res_fn, state, res, md, False)
    judge_set("評価セット v6（直す用）", rows6, cached(load_eval6), [x for x in load_all() if x["set"] == "v6"], ns, CONFIGS[:2], ts, res_fn, state, res, md, False)

    # ---- 部品ごと（評価セット v8）
    effect = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    md.append("\n# 部品ごとの精度（規則・引数は外す）\n")
    hl = [("行き先の正解率（対象の依頼と雑談）", "acc", 3), ("　機能の語を含む依頼", "acc_K", 3), ("　ふつうの依頼", "acc_N", 3), ("　雑談", "acc_F", 3)] + \
         [(f"　再現率: {J[c_]}", f"recall_{J[c_]}", 3) for c_ in ORDER] + \
         [("対象の依頼と雑談を通した割合", "cover", 3), ("通したうち、別の操作系と判定（行）", "wrong_write", 1),
          ("機能にない依頼を止めた（棄却・確信度）", "H_stop", 3), ("　操作系として通した", "H_pass_write", 3), ("意図が不明を止めた（棄却・確信度）", "G_stop", 3), ("　操作系として通した", "G_pass_write", 3)]
    for tag, loader in (("評価セット v8", cached(load_eval8)), ("評価セット v7（直す用）", cached(load_eval7))):
        for n in ns:
            loaded = loader(n)
            if loaded is None:
                continue
            rs, last, mean = loaded
            hp = {src: head_part(n, src, rs, last, mean, effect) for src in (SRC1584, SRC1656)}
            if not all(hp.values()):
                continue
            print(f"\n===== 部品: LM ヘッド・{tag}・層 {n} 個（1,584 行 → 1,656 行）=====")
            md += [f"\n## LM ヘッド・{tag}・層 {n} 個（mlp・seed 3 つ。棄却の閾値は validation の 2%）\n", "|指標|1,584 行|1,656 行|", "|---|---|---|"]
            for lab_, key, nd in hl:
                print(f"{lab_:<28} {ms(hp[SRC1584][key], nd):>22} → {ms(hp[SRC1656][key], nd)}")
                md.append(f"|{lab_}|{ms(hp[SRC1584][key], nd)}|{ms(hp[SRC1656][key], nd)}|")
            for src in (SRC1584, SRC1656):
                md.append(f"\n取り違えの多い組（{src}。正解 → 判定。1 つのヘッドあたりの行数）: " + " / ".join(f"{x['true']} → {x['pred']} {x['rows']:.1f}" for x in hp[src]["confusions"]))
            res["head"][f"{tag}|{n}"] = hp
    bp = bert_part(ts["all"], rows8)
    res["bert"] = bp
    names = sorted(ts["all"]["models"])
    md += ["\n## BERT 案 F（評価セット v8・seed ごと。学習し直していない）\n", "|指標|" + "|".join(names) + "|", "|---|" + "---|" * len(names)]
    for lab_, key in (("機能にない依頼にフラグ以上（再現率）", "H_flag"), ("　どの学習にも無い機能", "H_flag_unseen"), ("対象の依頼と雑談に誤ってフラグ", "known_flag"), ("AUC", "auc_known")):
        md.append(f"|{lab_}|" + "|".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in bp.get(key, [])) + "|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "verdict.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（verdict.md・verdict.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "heads", "text", "judge"])
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(str(s) for s in BERT_SEEDS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P28 {a.step} =====", flush=True)
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
