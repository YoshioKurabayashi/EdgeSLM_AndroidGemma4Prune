# -*- coding: utf-8 -*-
"""P29: 「対象の機能以外をはじく」2 つのやり方の当たりを付ける。端末なし・学習なし（端末で集めた h と、評価セット v6〜v8 を使う）。

設計: DoC/設計書_P29_対象の機能以外をはじく.md の 4 節。**v6〜v8 はどれも直す用。ここの数字は当たりで、合否ではない**（合否は新しい評価セット v9）。

|やり方|中身|機能にない依頼の例を使うか|
|---|---|---|
|1. 機能の定義で絞る（`p29_scope`）|読み出しの 5 機能は、機能の定義（項目）で説明がつく語が発話にあるときだけ実行する|使わない（語の一覧は機能の定義から書いた）|
|2. 対象の依頼との近さ（h）|ヘッドが使う h で、学習の対象の行（train）からの近さを測り、遠ければはじく。閾値は対象の行（validation）だけで決める（2%・5%）|使わない（評価にだけ使う）|

    python -m app.p29_check        # results/P29/check.md・check.json
"""
from __future__ import annotations

import collections
import json

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p7_data import read_jsonl
from .p11_bert import MODES, by_mode
from .p13_eval import cell
from .p18_retrain import SRC1584, cached
from .p19_config import bert_check
from .p21_data import CH6
from .p21_verdict import KPI, STOPS, load_eval6, share_by
from .p22_fix import value_kpi
from .p22_labels import load_all
from .p26_arguments import make_resolve as args_p26
from .p27_data import CH7, labeled as labeled7
from .p27_verdict import load_eval7
from .p28_data import CH8, TR4, labeled as labeled8
from .p28_verdict import SRC1656, WORK as P28, load_eval8, measure
from .p29_scope import READ, evidence, make_resolve as args_p29

RES = PROJECT_ROOT / "results" / "P29"
CONFIGS = [("ヘッド 1,656 行・P26（P28 の本命）", SRC1656, args_p26), ("ヘッド 1,656 行・P29（機能の定義で絞る）", SRC1656, args_p29),
           ("ヘッド 1,584 行・P26", SRC1584, args_p26), ("ヘッド 1,584 行・P29", SRC1584, args_p29)]
GRADES = ("正解", "回復可能", "通常", "重大")


def groups(d) -> dict:
    g = collections.Counter()
    for x in d["by_group"]:
        key = "H unseen" if x["category"] == "H" and x["group"] == "unseen" else "H seen" if x["category"] == "H" else x["category"]
        g[(key, x["grade"])] += x["rows"]
    return g


def distance(md: list, res: dict) -> None:
    norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    rows = read_jsonl(TR4 / "utterances.jsonl")
    ins = np.array([r["route"] != "UNKNOWN" for r in rows])
    sp = np.array([r["split"] for r in rows])
    md += ["\n## 2. 対象の依頼との近さ（h）ではじく\n", "学習の対象の行（train）を基準にし、閾値は対象の行（validation）を 2%・5% 誤ってはじく値。機能にない依頼は使っていない。数字は「閾値 2% ／ 5%」のはじいた割合\n",
           "|層・測り方|評価セット|対象の依頼と雑談（誤り）|機能にない依頼 seen|機能にない依頼 unseen|意図が不明|", "|---|---|---|---|---|---|"]
    for n in (17, 26):
        z = np.load(P28 / f"L{n:02d}" / "features" / "features_F2.npz")
        X = np.hstack([norm(z["last__last"]), norm(z["last__mean"])]).astype(np.float64)
        ref, val = X[ins & (sp == "train")], X[ins & (sp == "validation")]
        mu = ref.mean(0)
        _, S, Vt = np.linalg.svd(ref - mu, full_matrices=False)
        fns = {"近い 5 行との距離（cos）": lambda A: 1 - np.sort(norm(A) @ norm(ref).T, 1)[:, -5:].mean(1),
               "マハラノビス（主成分 64）": lambda A: ((((A - mu) @ Vt[:64].T) / (S[:64] / np.sqrt(len(ref)))) ** 2).sum(1)}
        for fname, fn in fns.items():
            sv = fn(val)
            th = (np.quantile(sv, 0.98), np.quantile(sv, 0.95))
            for tag, ld in (("v6", load_eval6), ("v7", load_eval7), ("v8", load_eval8)):
                rs, last, mean = ld(n)
                s = fn(np.hstack([norm(last), norm(mean)]))
                cat = np.array([r["category"] for r in rs])
                grp = np.array([r.get("os_group") or "" for r in rs])
                rate = lambda m: "／".join(f"{np.mean(s[m] >= t):.2f}" for t in th)
                cells = [rate(np.isin(cat, list("KNF"))), rate(grp == "seen"), rate(grp == "unseen"), rate(cat == "G")]
                res["distance"][f"{n}|{fname}|{tag}"] = cells
                md.append(f"|層 {n}・{fname}|{tag}|" + "|".join(cells) + "|")
    print("近さ（h）の結果を書いた", flush=True)


def main() -> None:
    raw = json.loads((P28 / "text_scores_F.json").read_text(encoding="utf-8"))
    use_multi_library()
    from edge_router.dispatch import arguments
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    res = {"kpi": {}, "groups": {}, "changed": {}, "distance": {}, "lexicon_misses": {}}
    md = ["# P29 対象の機能以外をはじく（当たりを付ける。端末なし・学習なし）\n",
          "**評価セット v6〜v8 はどれも直す用。ここの数字は当たりで、合否ではない**（v8 の弱点を見てから考えたやり方なので。合否は新しい評価セット v9 で測る）。"
          f"ヘッドの seed 3 つの平均・BERT は {chk['one']}・規則は P21\n"]
    sets = (("v8", read_jsonl(CH8 / "utterances.jsonl"), cached(load_eval8), labeled8()),
            ("v7", read_jsonl(CH7 / "utterances.jsonl"), cached(load_eval7), labeled7()),
            ("v6", read_jsonl(CH6 / "utterances.jsonl"), cached(load_eval6), [x for x in load_all() if x["set"] == "v6"]))
    md += ["\n## 1. 機能の定義で絞る（`p29_scope`）\n", "### 1.1 語の一覧で説明がつかない、対象の読み出しの行（正しい機能のとき）\n"]
    for tag, rows, _, _ in sets:
        kn = [r for r in rows if r["action"][5:] in READ]
        miss = [r["utterance"] for r in kn if not evidence(r["action"][5:], r["utterance"])]
        res["lexicon_misses"][tag] = miss
        md.append(f"- {tag}: 読み出しの対象 {len(kn)} 行のうち {len(miss)} 行" + (f"（{' / '.join(miss)}）" if miss else "（語の一覧を見直した後なので、ここは甘い）"))
    for tag, rows, loader, lab in sets:
        n_rows = len(rows)
        for n in (17, 26):
            keep = {}
            for cname, src, make in CONFIGS:
                got = measure(n, src, ts, loader, make)
                if got is None:
                    continue
                k, d = got
                keep[cname] = (k, d, value_kpi(k, d, lab, make(arguments), state, n_rows))
            if not keep:
                continue
            print(f"\n===== {tag}・層 {n} =====", flush=True)
            heads_ = ["値まで見た EtoE", "値まで見た重大な誤り（件）", "不要な聞き返し", "通常の誤り", "違う値での実行"] + [f"止めた: {s_.replace('聞き返し（', '').replace('拒否（', '').rstrip('）')}" for s_ in STOPS]
            md += [f"\n### 1.2 流れの KPI・{tag}・層 {n}\n", "|構成|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
            for cname, (k, d, v) in keep.items():
                cnt = collections.Counter()
                for x in d["stop_rows"]:
                    cnt[x["stop"]] += x["share"]
                cells = [f"{v['e2e_value']:.3f}", f"{v['severe_value_n']:.1f}", cell(k["over"], 3), cell(k["normal"], 3), f"{v['wrong_exec']:.1f}"] + [f"{cnt[s_]:.1f}" for s_ in STOPS]
                res["kpi"][f"{tag}|{n}|{cname}"] = {"e2e_value": v["e2e_value"], "severe_value_n": v["severe_value_n"], "over": k["over"], "normal": k["normal"], "stops": [cnt[s_] for s_ in STOPS]}
                md.append(f"|{cname}|" + "|".join(cells) + "|")
                print(f"  {cname[:24]:<24} EtoE {v['e2e_value']:.3f}・重大 {v['severe_value_n']:.1f}・聞き返し {k['over']['mean']:.3f}・通常 {k['normal']['mean']:.3f}", flush=True)
            md += [f"\n種類ごとの扱い（{tag}・層 {n}・ヘッド 1,656 行。行数）:\n", "|種類|P26: " + "／".join(GRADES) + "|P29: " + "／".join(GRADES) + "|", "|---|---|---|"]
            ga, gb = groups(keep[CONFIGS[0][0]][1]), groups(keep[CONFIGS[1][0]][1])
            res["groups"][f"{tag}|{n}"] = {"P26": {f"{a}|{b}": v for (a, b), v in ga.items()}, "P29": {f"{a}|{b}": v for (a, b), v in gb.items()}}
            for key in ("H seen", "H unseen", "G", "K", "N", "F"):
                md.append(f"|{key}|" + "／".join(f"{ga[(key, g)]:.1f}" for g in GRADES) + "|" + "／".join(f"{gb[(key, g)]:.1f}" for g in GRADES) + "|")
            a, b = keep[CONFIGS[0][0]][1], keep[CONFIGS[1][0]][1]
            sa, sb = share_by(a["stop_rows"], ("聞き返し（引数が足りない）",)), share_by(b["stop_rows"], ("聞き返し（引数が足りない）",))
            cost = [[u, round(sb[u] - sa[u], 2)] for u in sb if sb[u] - sa[u] > 1e-9]
            na, nb = share_by(a["normal_rows"]), share_by(b["normal_rows"])
            gain = [[u, round(na[u] - nb[u], 2)] for u in na if na[u] - nb[u] > 1e-9]
            va, vb = share_by(a["severe_rows"]), share_by(b["severe_rows"])
            sev = [[u, round(va[u] - vb[u], 2)] for u in va if va[u] - vb[u] > 1e-9]
            res["changed"][f"{tag}|{n}"] = {"known_stopped": cost, "normal_fixed": gain, "severe_fixed": sev}
            fmt = lambda items: " / ".join(f"{u}（{s:.2f}）" for u, s in items) if items else "なし"
            md += [f"\nP26 → P29 で変わった行（{tag}・層 {n}・ヘッド 1,656 行）:\n", "|見るもの|行|", "|---|---|",
                   f"|新しく止めた対象の行（代価）（{sum(s for _, s in cost):.1f}）|{fmt(cost)}|",
                   f"|通常の誤りでなくなった行（{sum(s for _, s in gain):.1f}）|{fmt(gain)}|",
                   f"|重大な誤りでなくなった行（{sum(s for _, s in sev):.1f}）|{fmt(sev)}|"]
            print(f"  P26→P29: 新しく止めた対象 {sum(s for _, s in cost):.1f}・通常でなくなった {sum(s for _, s in gain):.1f}・重大でなくなった {sum(s for _, s in sev):.1f}", flush=True)
    distance(md, res)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（check.md・check.json）")


if __name__ == "__main__":
    main()
