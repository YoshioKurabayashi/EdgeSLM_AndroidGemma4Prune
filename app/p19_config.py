# -*- coding: utf-8 -*-
"""P19: P18 の分析で見つけた 3 つの手当てを、決まりとして書き、構成を 1 つに決める。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P19_構成を確定する.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md

|手当て|決まり|
|---|---|
|1 BERT の使うモデル|全体で学習したモデルを、**確かめ用の行**（1,101 行の test・使い終えた評価セット v1・v2）に当てる。対象の依頼と雑談にフラグを立てる割合が、学習側の交差検証での割合の 2 倍を超えた seed は使わない。端末に載せる 1 つは、残った seed のうち、確かめ用の行の機能にない依頼を一番多く見つけるもの|
|2 棄却の閾値|validation の行き先のある行の 2% → 5% を誤って回す値|
|3 規則 S5|実行しようとする操作系ツールの語が、発話に 1 つも無ければ聞き返す（`p19_rules`）|

ヘッドは 1,584 行（P18）・BERT は案 F・引数は `p17_arguments`・数え方は P15。
**評価セット v3・v4 は「直す用」。ここの数字は合否ではない**（合否は、新しい行で測る）。

    python -m app.p19_config        # results/P19/config.md・config.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import PROJECT_ROOT
from .p7_data import CH1, CH2, TR2, read_jsonl
from .p8_gate import BERT_LO, STEPS, run
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import cell, hi_rule, interval, load_eval3
from .p17_arguments import make_resolve
from .p17_rules import make_rules as rules_p17
from .p18_analyze import hook1584
from .p18_data import CH4, TR3
from .p18_retrain import SRC1584, WORK as P18, cached, has_heads, load_eval4
from .p19_rules import S5_TERMS, has_s5_term, make_rules as rules_p19

RES = PROJECT_ROOT / "results" / "P19"
FLAG_RATIO = 2.0          # 交差検証での割合の何倍を超えたら、その seed を使わないか
CONFIGS = [("A P18 の本命（BERT は seed 3 つ・閾値 2%）", "all", 0.02, rules_p17),
           ("B ＋BERT の seed の決まり", "ok", 0.02, rules_p17),
           ("C 　＋閾値 5%", "ok", 0.05, rules_p17),
           ("D 　＋規則 S5（閾値は 2%）", "ok", 0.02, rules_p19),
           ("E 　＋閾値 5%＋規則 S5", "ok", 0.05, rules_p19),
           ("D1 D で、端末に載せる BERT 1 つ", "one", 0.02, rules_p19),
           ("E1 E で、端末に載せる BERT 1 つ", "one", 0.05, rules_p19)]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("不要な聞き返し", "over", 3), ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3)]
CAT = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}


def bert_check(raw: dict) -> dict:
    """seed ごとに、交差検証でのフラグの割合・確かめ用の行でのフラグの割合と見つけた割合・使うかどうか"""
    pool = [r for r in read_jsonl(TR2 / "utterances.jsonl") if r["split"] == "test"] + read_jsonl(CH1) + read_jsonl(CH2 / "utterances.jsonl")
    known = [r["id"] for r in pool if r["action"] not in ("UNKNOWN", "MULTI")]
    os_ = [r["id"] for r in pool if r.get("category") == "H" or r.get("add_type") == "OS"]
    fit = list(raw["labels_fit"])
    y = np.array([raw["labels_fit"][i] for i in fit])
    out = {}
    for name in sorted(k for k in raw["models"] if k.startswith("bert_s")):
        m = raw["models"][name]
        cv = float(np.mean(np.array([m["cv"][i] for i in fit])[y == 0] >= BERT_LO))
        flag = float(np.mean([m["pred"][i] >= BERT_LO for i in known]))
        rec = float(np.mean([m["pred"][i] >= BERT_LO for i in os_]))
        out[name] = {"cv_flag": cv, "pool_flag": flag, "pool_recall": rec, "ok": flag <= FLAG_RATIO * cv, "rows_known": len(known), "rows_os": len(os_)}
    good = [k for k, v in out.items() if v["ok"]]
    one = max(good, key=lambda k: (out[k]["pool_recall"], -out[k]["pool_flag"])) if good else None
    return {"seeds": out, "ok": good, "one": one}


def measure(n: int, ts: dict, loader, fpr: float, rules):
    if not has_heads(n, SRC1584) or loader(n) is None:
        return None
    final = STEPS[-1]
    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=hook1584(fpr), steps=[final], sources=[SRC1584], quiet=True,
               ts=ts, watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=rules, amb_chat_ok=True)
    key = f"{n}|{SRC1584}|{final[0]}"
    return blob["kpi"][key], blob["detail"][key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    raw = json.loads((P18 / "text_scores_F.json").read_text(encoding="utf-8"))
    print("\n===== P19: 構成を確定する（評価セット v3・v4 は直す用。合否ではない）=====")
    md = ["**評価セット v3・v4 は「直す用」。ここの数字は合否ではない**。数字はヘッドの seed 3 つ × BERT の seed の平均（かっこは最小〜最大）。ヘッドは 1,584 行・BERT は案 F\n"]
    res = {"bert": {}, "s5": {}, "kpi": {}, "stops": {}, "severe": {}, "detail": {}}

    # ---- 1. BERT の使うモデル
    chk = bert_check(raw)
    res["bert"] = chk
    print(f"\n===== 1. BERT の使うモデル（確かめ用の行: 1,101 行の test・評価セット v1・v2）=====")
    md += ["\n## 1. BERT の使うモデル\n", f"確かめ用の行 = 1,101 行の test・評価セット v1・v2（対象の依頼と雑談 {next(iter(chk['seeds'].values()))['rows_known']} 行・機能にない依頼 "
           f"{next(iter(chk['seeds'].values()))['rows_os']} 行）。決まり: フラグの割合が、交差検証での割合の {FLAG_RATIO:g} 倍を超えた seed は使わない\n",
           "|seed|交差検証でのフラグ|確かめ用の行でのフラグ|倍率|機能にない依頼を見つけた割合|使うか|", "|---|---|---|---|---|---|"]
    for name, v in chk["seeds"].items():
        line = f"|{name}|{v['cv_flag']:.3f}|{v['pool_flag']:.3f}|{v['pool_flag'] / v['cv_flag']:.1f}|{v['pool_recall']:.3f}|{'使う' if v['ok'] else '**使わない**'}" + \
               ("（端末に載せる 1 つ）" if name == chk["one"] else "") + "|"
        md.append(line)
        print(line)
    ts = {"all": by_mode(raw, MODES[0]), "ok": by_mode(raw, MODES[0], only=tuple(chk["ok"])), "one": by_mode(raw, MODES[0], only=(chk["one"],))}

    # ---- 2. 規則 S5 の巻き添えの見積もり（ヘッドが正しく判定したとして、S5 が止める対象の行）
    sets_ = {"学習 1,584 行": read_jsonl(TR3 / "utterances.jsonl"), "評価セット v1": read_jsonl(CH1), "評価セット v2": read_jsonl(CH2 / "utterances.jsonl"),
             "評価セット v3": read_jsonl(CH3 / "utterances.jsonl"), "評価セット v4": read_jsonl(CH4 / "utterances.jsonl")}
    print("\n===== 2. 規則 S5 が止める行（辞書だけで数える）=====")
    md += ["\n## 2. 規則 S5 が止める行（辞書だけで数える。ヘッドの判定は入れない）\n",
           "|行|操作系の対象の行（履歴なし）|S5 が止める行|同・学習に出る語だけの辞書のとき|機能にない依頼: どの操作系の語も無い|意図が不明: 同|雑談: 同|", "|---|---|---|---|---|---|---|"]
    for name, rows in sets_.items():
        sel = [(r, t) for t in S5_TERMS for r in rows if r["action"] == f"TOOL_{t}" and not r.get("history")]
        miss = [r["utterance"] for r, t in sel if not has_s5_term(r["utterance"], t)]
        miss_t = [r["utterance"] for r, t in sel if not has_s5_term(r["utterance"], t, True)]
        none = lambda rs: f"{sum(not any(has_s5_term(r['utterance'], t) for t in S5_TERMS) for r in rs)}/{len(rs)}" if rs else "—"
        hs = [r for r in rows if r.get("category") == "H" or r.get("add_type") == "OS"]
        gs = [r for r in rows if r.get("category") == "G" or r.get("add_type") == "AM"]
        fs = [r for r in rows if r["action"] == "CHAT"]
        res["s5"][name] = {"write_rows": len(sel), "stopped": miss, "stopped_train_terms": miss_t}
        md.append(f"|{name}|{len(sel)}|{len(miss)}|{len(miss_t)}|{none(hs)}|{none(gs)}|{none(fs)}|")
        print(f"{name}: 操作系の対象 {len(sel)} 行のうち S5 が止める {len(miss)}（学習の語だけなら {len(miss_t)}）/ 語なし: 機能にない依頼 {none(hs)}・意図が不明 {none(gs)}・雑談 {none(fs)}")
    md.append("\n学習の行で S5 が止めるもの: " + " / ".join(res["s5"]["学習 1,584 行"]["stopped"]))
    md.append("\n学習に出る語だけの辞書で止まる評価セットの行: " + " / ".join(u for k, v in res["s5"].items() if k != "学習 1,584 行" for u in v["stopped_train_terms"]))

    # ---- 3. 構成ごとの KPI
    sets = [("v4", read_jsonl(CH4 / "utterances.jsonl"), cached(load_eval4)), ("v3", read_jsonl(CH3 / "utterances.jsonl"), cached(load_eval3))]
    md.append("\n## 3. 構成ごとの KPI\n")
    for tag, rows, loader in sets:
        n_rows, n_known = len(rows), sum(r["action"] != "UNKNOWN" for r in rows)
        for n in ns:
            print(f"\n===== 3. {tag}・層 {n} 個 =====")
            heads = [lab for lab, _, _ in KPI] + ["重大な誤りの内訳（機能にない依頼／意図が不明／雑談／対象）", "対象を止めた段（規則／棄却／確信度／BERT フラグ／BERT 断る／引数）"]
            md += [f"\n### {tag}・層 {n} 個（{n_rows} 行。対象の依頼と雑談 {n_known} 行）\n", "|構成|" + "|".join(heads) + "|", "|---|" + "---|" * len(heads)]
            for cname, which, fpr, rules in CONFIGS:
                got = measure(n, ts[which], loader, fpr, rules)
                if got is None:
                    continue
                k, d = got
                sev = collections.Counter()
                for x in d["by_group"]:
                    if x["grade"] == "重大":
                        sev[x["category"]] += x["rows"]
                st = collections.Counter()
                for x in d["stop_rows"]:
                    st[x["stop"]] += x["share"]
                stops = [st["聞き返し（規則）"], st["聞き返し（棄却ヘッド）"], st["聞き返し（確信度）"], st["聞き返し（BERT のフラグ）"], st["拒否（BERT が断る）"],
                         st["聞き返し（引数が足りない）"] + st["聞き返し（検査）"] + st["拒否（検査）"]]
                cells = [cell(k[key], nd) for _, key, nd in KPI] + \
                        [f"{sev['H']:.1f}／{sev['G']:.1f}／{sev['F']:.1f}／{sev['K'] + sev['N']:.1f}", "／".join(f"{x:.1f}" for x in stops)]
                res["kpi"][f"{tag}|{n}|{cname}"] = {**{key: k[key] for _, key, _ in KPI}, "severe": k["severe"], "e2e_ci": interval(k["e2e"]["mean"], n_rows)}
                res["severe"][f"{tag}|{n}|{cname}"] = dict(sev)
                res["stops"][f"{tag}|{n}|{cname}"] = stops
                if cname[0] in "DE":
                    res["detail"][f"{tag}|{n}|{cname}"] = {y: d[y] for y in ("severe_rows", "normal_rows", "stop_rows")}
                print(f"{cname:<30} " + "  ".join(f"{h[:10]}: {x}" for h, x in zip(heads, cells)))
                md.append(f"|{cname}|" + "|".join(cells) + "|")
            for cname in ("D 　＋規則 S5（閾値は 2%）", "E 　＋閾値 5%＋規則 S5"):
                d = res["detail"].get(f"{tag}|{n}|{cname}")
                if d:
                    md += [f"\n{cname.strip()} で残る重大な誤り: " + " / ".join(f"[{y['category']}]{y['utterance']} → {y['executed']}（{y['share']:.2f}）" for y in d["severe_rows"][:20]),
                           f"\n{cname.strip()} で止めた対象の行: " + " / ".join(f"{y['utterance']}（{y['stop']}・{y['share']:.2f}）" for y in d["stop_rows"][:30])]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "config.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "config.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（config.md・config.json）")


if __name__ == "__main__":
    main()
