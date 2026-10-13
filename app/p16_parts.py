# -*- coding: utf-8 -*-
"""P16: 部品ごとの分類の精度を、評価セット v3（800 行）で分けて出す。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P16_部品ごとの分類精度.md

これまでの KPI は、流れ全体（BERT → 規則 → 棄却ヘッド → 確信度 → 引数 → 対象の語）を通した結果だった。
ここでは規則・引数・対象の語の確かめを外し、**学習した 2 つの分類器だけ**を見る。

|部品|何を分けるか|出すもの|
|---|---|---|
|BERT|機能にない依頼か、それ以外か（2 値）|フラグ（0.5）と断る閾値での、見つけた割合・誤って止めた割合・適合率・AUC。種類ごと|
|LM ヘッド（行き先）|8 つの機能＋雑談の 9 クラス|対象の依頼・雑談 480 行での正解率（全体・機能ごと）と、取り違えの組|
|LM ヘッド（棄却・確信度）|実行してよいか|通した割合と、通したうちの正解率。機能にない依頼・意図が不明を止めた割合|
|組み合わせ|—|BERT とヘッドが、同じ行を見逃すか別の行を見逃すか|

BERT は P13 の案 C・案 E（seed 3 つ）。ヘッドは 933 行（P5・mlp・seed 3 つ）、棄却の閾値は validation の 2%（P10）。

    python -m app.p16_parts        # results/P16/parts.md・parts.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT
from .p6_seeds import SEEDS, auc, head_probs, load_pt
from .p7_block import HEADS
from .p7_data import read_jsonl
from .p8_gate import BERT_LO
from .p10_reject import SRC, make_hook
from .p13_data import CH3
from .p13_eval import WORK as P13, hi_rule, load_eval3

RES = PROJECT_ROOT / "results" / "P16"
J = {"CHAT": "雑談", "TOOL_control_hvac": "空調", "TOOL_control_media": "メディア", "TOOL_set_navigation": "ナビ", "TOOL_get_vehicle_status": "車両の状態",
     "TOOL_search_vehicle_manual": "説明書", "TOOL_search_poi": "周辺の検索", "TOOL_get_media_info": "曲の情報", "TOOL_capture_camera": "カメラ"}
WRITE = ("TOOL_control_hvac", "TOOL_control_media", "TOOL_set_navigation")
ORDER = ["TOOL_control_hvac", "TOOL_control_media", "TOOL_set_navigation", "TOOL_get_vehicle_status", "TOOL_search_vehicle_manual", "TOOL_search_poi",
         "TOOL_get_media_info", "TOOL_capture_camera", "CHAT"]


def ms(xs, nd=3) -> str:
    xs = [float(x) for x in xs]
    return f"{np.mean(xs):.{nd}f}" if min(xs) == max(xs) else f"{np.mean(xs):.{nd}f}（{min(xs):.{nd}f}〜{max(xs):.{nd}f}）"


def load_heads(n: int, last, mean) -> list:
    d = HEADS[SRC](n)
    reps = {r["checkpoint"]: r for r in json.loads((d / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}
    heads = []
    for seed in SEEDS:
        name = f"heads_mlp_seed{seed}.pt"
        blob, rep = load_pt(d / name), reps[name]
        heads.append((seed, blob["metadata"]["action_classes"], {k: float(v) for k, v in rep["thresholds"].items()},
                      head_probs(blob, float(rep["temperature"]), last, mean)))
    return make_hook("val", 0.02, {})(n, SRC, heads)                 # 棄却の閾値は validation の 2%（P10 の本命）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    ch3 = read_jsonl(CH3 / "utterances.jsonl")
    cat = np.array([r["category"] for r in ch3])
    grp = np.array([r.get("os_group") or r.get("g_kind") or "" for r in ch3])
    ids = [r["id"] for r in ch3]
    act = np.array([r["action"] for r in ch3])
    known = act != "UNKNOWN"
    isH, isG = cat == "H", cat == "G"
    effect = {t["name"]: t.get("effect", "read") for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    md, res = [f"評価セット v3: {len(ch3)} 行（機能にない依頼 {int(isH.sum())}・意図が不明 {int(isG.sum())}・対象の依頼と雑談 {int(known.sum())}）。"
               "数字は seed の平均（かっこは最小〜最大）。規則・引数・対象の語の確かめは外している\n"], {"bert": {}, "head": {}, "combo": {}}
    print("\n===== P16: 部品ごとの分類の精度（評価セット v3）=====")

    # ---------------------------------------------------------------- 1. BERT（2 値）
    berts = {}
    for key in ("C", "E"):
        f = P13 / f"text_scores_{key}.json"
        if not f.exists():
            continue
        ts = json.loads(f.read_text(encoding="utf-8"))
        fit = list(ts["labels_fit"])
        y = np.array([ts["labels_fit"][i] for i in fit])
        models = []
        for name in sorted(k for k in ts["models"] if k.startswith("bert_s")):
            m = ts["models"][name]
            hi = hi_rule(name, np.array([m["cv"][i] for i in fit])[y == 0])
            models.append((name, np.array([m["pred"][i] for i in ids]), hi))
        berts[key] = models
    titles = {"C": "案 C（P11。対象側の文 90 行を足した）", "E": "案 E（P13。さらに機能にない依頼 36 機能を足した）"}
    print("\n===== 1. BERT（機能にない依頼か、それ以外か）=====")
    md += ["\n## 1. BERT（機能にない依頼か、それ以外か の 2 値）\n",
           "「フラグ」= 確率 0.5 以上。「断る」= 学習側の 99.5% 点以上（seed ごとに 0.993〜0.997）\n"]
    for key, models in berts.items():
        md += [f"\n### {titles[key]}\n", "|指標|フラグ（0.5 以上）|断る（99.5% 点以上）|", "|---|---|---|"]
        rows_ = [("機能にない依頼を見つけた割合（再現率）・180 行", isH), ("　学習に同じ機能がある・60 行", isH & (grp == "seen")),
                 ("　BERT に足した機能・40 行" if key == "E" else "　案 C の学習に無い機能（案 E で足す 20 機能）・40 行", isH & (grp == "new_seen")),
                 ("　どの学習にも無い機能・80 行", isH & (grp == "unseen")),
                 ("誤って止めた割合: 対象の依頼と雑談・480 行", known), ("　機能の語を含む依頼・160 行", cat == "K"), ("　ふつうの依頼・240 行", cat == "N"),
                 ("　雑談・80 行", cat == "F"), ("意図が不明に立った割合・140 行", isG)]
        out = {}
        for lab, sel in rows_:
            lo = [float(np.mean(p[sel] >= BERT_LO)) for _, p, _ in models]
            hi = [float(np.mean(p[sel] >= h)) for _, p, h in models]
            out[lab] = {"flag": lo, "refuse": hi}
            md.append(f"|{lab}|{ms(lo)}|{ms(hi)}|")
            print(f"[{key}] {lab:<34} フラグ {ms(lo)}  断る {ms(hi)}")
        neg = ~isH
        prec_lo = [float((p[isH] >= BERT_LO).sum() / max(1, (p >= BERT_LO).sum())) for _, p, _ in models]
        prec_hi = [float((p[isH] >= h).sum() / max(1, (p >= h).sum())) for _, p, h in models]
        acc_lo = [float(np.mean((p >= BERT_LO) == isH)) for _, p, _ in models]
        aucs = [auc(p[isH], p[neg]) for _, p, _ in models]
        auc_k = [auc(p[isH], p[known]) for _, p, _ in models]
        md += [f"|適合率（止めたうち、機能にない依頼だった割合）|{ms(prec_lo)}|{ms(prec_hi)}|", f"|2 値の正解率（800 行）|{ms(acc_lo)}|—|",
               f"|AUC（機能にない依頼 ↔ それ以外 620 行）|{ms(aucs)}|—|", f"|AUC（機能にない依頼 ↔ 対象の依頼と雑談 480 行）|{ms(auc_k)}|—|"]
        print(f"[{key}] 適合率 フラグ {ms(prec_lo)}・断る {ms(prec_hi)} / 2 値の正解率 {ms(acc_lo)} / AUC {ms(aucs)}・対象だけと比べると {ms(auc_k)}")
        res["bert"][key] = {**out, "precision_flag": prec_lo, "precision_refuse": prec_hi, "accuracy_flag": acc_lo, "auc": aucs, "auc_known": auc_k,
                            "hi": [h for _, _, h in models]}
        if key == "E":                                               # 見逃した行・誤って止めた行（seed の多数決）
            vote = np.mean([p >= BERT_LO for _, p, _ in models], axis=0)
            miss = [(r["utterance"], r["os_group"]) for r, v in zip(ch3, vote) if r["category"] == "H" and v < 0.5]
            fp = [(r["utterance"], r["category"]) for r, v in zip(ch3, vote) if r["action"] != "UNKNOWN" and v >= 0.5]
            md += [f"\n案 E が見逃した機能にない依頼（seed 3 つのうち 2 つ以上で 0.5 未満）: {len(miss)} 行\n", "|組|発話|", "|---|---|"]
            for g in ("seen", "new_seen", "unseen"):
                md.append(f"|{g}|" + " / ".join(u for u, gg in miss if gg == g) + "|")
            md += [f"\n案 E が誤ってフラグを立てた対象の依頼・雑談（2 つ以上で 0.5 以上）: {len(fp)} 行\n", " / ".join(u for u, _ in fp) + "\n"]
            res["bert"]["E_missed"], res["bert"]["E_false_flag"] = miss, fp
            print(f"[E] 見逃した機能にない依頼 {len(miss)} 行 / 誤ってフラグ {len(fp)} 行")

    # ---------------------------------------------------------------- 2・3. LM ヘッド
    for n in ns:
        loaded = load_eval3(n)
        if loaded is None:
            continue
        _, last, mean = loaded
        heads = load_heads(n, last, mean)
        H = {"acc": [], "acc_K": [], "acc_N": [], "acc_F": [], "per": collections.defaultdict(list), "prec": collections.defaultdict(list),
             "conf": collections.Counter(), "cover": [], "sel_acc": [], "wrong_write": [], "wrong_other": [], "stop_reject": [], "stop_conf": [],
             "auc_H": [], "auc_G": [], "taus": []}
        hs = {"H": collections.defaultdict(list), "G": collections.defaultdict(list)}
        passes = []
        for seed, classes, th, p in heads:
            pred = np.array([classes[i] for i in p["action"].argmax(1)])
            conf = p["action"].max(1)
            tau = np.array([th["CHAT"] if c == "CHAT" else th["TOOL_write" if effect.get(c[5:]) == "write" else "TOOL_read"] for c in pred])
            rej = p["reject"] >= th["REJECT"]
            low = ~rej & (conf < tau)
            ok = ~rej & ~low                                         # ヘッドが通した（棄却でも確信度不足でもない）
            passes.append((pred, ok, rej, low))
            right = pred == act
            H["acc"].append(np.mean(right[known]))
            for c_ in "KNF":
                H[f"acc_{c_}"].append(np.mean(right[cat == c_]))
            for c_ in ORDER:
                sel = known & (act == c_)
                H["per"][c_].append(np.mean(right[sel]))
                H["prec"][c_].append(float(right[known & (pred == c_)].mean()) if (known & (pred == c_)).any() else float("nan"))
            for t_, p_ in zip(act[known & ~right], pred[known & ~right]):
                H["conf"][(t_, p_)] += 1
            H["cover"].append(np.mean(ok[known]))
            H["sel_acc"].append(np.mean(right[known & ok]))
            wr = known & ok & ~right
            H["wrong_write"].append(int(np.isin(pred[wr], WRITE).sum()))
            H["wrong_other"].append(int((~np.isin(pred[wr], WRITE)).sum()))
            H["stop_reject"].append(np.mean(rej[known]))
            H["stop_conf"].append(np.mean(low[known]))
            H["auc_H"].append(auc(p["reject"][isH], p["reject"][known]))
            H["auc_G"].append(auc(p["reject"][isG], p["reject"][known]))
            H["taus"].append(th["REJECT"])
            for name, sel in (("H", isH), ("G", isG)):
                hs[name]["reject"].append(np.mean(rej[sel]))
                hs[name]["low"].append(np.mean(low[sel]))
                hs[name]["pass_write"].append(np.mean(ok[sel] & np.isin(pred[sel], WRITE)))
                hs[name]["pass_chat"].append(np.mean(ok[sel] & (pred[sel] == "CHAT")))
                hs[name]["pass_read"].append(np.mean(ok[sel] & ~np.isin(pred[sel], WRITE) & (pred[sel] != "CHAT")))
                if name == "H":
                    for g in ("seen", "new_seen", "unseen"):
                        s2 = isH & (grp == g)
                        hs[name][f"pass_write_{g}"].append(np.mean(ok[s2] & np.isin(pred[s2], WRITE)))
        k = len(heads)
        print(f"\n===== 2. LM ヘッド・層 {n} 個 =====")
        md += [f"\n## 2. LM ヘッド・層 {n} 個（933 行で学習。mlp・seed 3 つ）\n", "### 行き先の分類（対象の依頼と雑談 480 行。一番確率の高いクラスが正解と同じか）\n",
               "|範囲|正解率|", "|---|---|", f"|全体・480 行|{ms(H['acc'])}|", f"|機能の語を含む依頼・160 行|{ms(H['acc_K'])}|",
               f"|ふつうの依頼・240 行|{ms(H['acc_N'])}|", f"|雑談・80 行|{ms(H['acc_F'])}|",
               "\n|クラス|行|再現率（そのクラスの行を当てた割合）|適合率（そのクラスと判定したうち正しかった割合）|", "|---|---|---|---|"]
        print(f"行き先の正解率: 全体 {ms(H['acc'])} / K {ms(H['acc_K'])} / N {ms(H['acc_N'])} / F {ms(H['acc_F'])}")
        for c_ in ORDER:
            md.append(f"|{J[c_]}|{int((known & (act == c_)).sum())}|{ms(H['per'][c_])}|{ms(H['prec'][c_])}|")
            print(f"  {J[c_]:<8} 再現率 {ms(H['per'][c_])}  適合率 {ms(H['prec'][c_])}")
        top = H["conf"].most_common(12)
        md += ["\n取り違えの多い組（正解 → 判定。1 つのヘッドあたりの行数）:\n", "|正解|判定|行数|", "|---|---|---|"]
        md += [f"|{J[t_]}|{J[p_]}|{v / k:.1f}|" for (t_, p_), v in top]
        print("  取り違え: " + " / ".join(f"{J[t_]}→{J[p_]} {v / k:.1f}" for (t_, p_), v in top[:8]))
        md += ["\n### 実行してよいかの判定（棄却ヘッド＋確信度の閾値）\n", f"棄却の閾値（validation の 2%）: {'・'.join(f'{t:.3f}' for t in H['taus'])}\n",
               "|対象の依頼と雑談 480 行|値|", "|---|---|", f"|ヘッドが通した割合|{ms(H['cover'])}|", f"|　棄却ヘッドが止めた|{ms(H['stop_reject'])}|",
               f"|　確信度が足りず止めた|{ms(H['stop_conf'])}|", f"|通したうちの正解率|{ms(H['sel_acc'])}|",
               f"|通したうち、別の操作系と判定（行）|{ms(H['wrong_write'], 1)}|", f"|通したうち、別の読むだけの機能・雑談と判定（行）|{ms(H['wrong_other'], 1)}|",
               "\n|真の種類|棄却ヘッドが止めた|確信度が足りず止めた|通した: 操作系と判定|通した: 読むだけの機能と判定|通した: 雑談と判定|棄却の得点の AUC（↔ 対象 480 行）|",
               "|---|---|---|---|---|---|---|"]
        for name, lab, a_ in (("H", "機能にない依頼・180 行", H["auc_H"]), ("G", "意図が不明・140 行", H["auc_G"])):
            x = hs[name]
            md.append(f"|{lab}|{ms(x['reject'])}|{ms(x['low'])}|{ms(x['pass_write'])}|{ms(x['pass_read'])}|{ms(x['pass_chat'])}|{ms(a_)}|")
            print(f"  {lab}: 棄却 {ms(x['reject'])} 確信度 {ms(x['low'])} / 通した 操作系 {ms(x['pass_write'])}・読むだけ {ms(x['pass_read'])}・雑談 {ms(x['pass_chat'])} / AUC {ms(a_)}")
        md.append("\n機能にない依頼を「操作系として通した」割合（組ごと）: " + "・".join(f"{g} {ms(hs['H'][f'pass_write_{g}'])}" for g in ("seen", "new_seen", "unseen")))
        print(f"  通した割合 {ms(H['cover'])}・通したうちの正解率 {ms(H['sel_acc'])}・別の操作系 {ms(H['wrong_write'], 1)} 行・別の読むだけ/雑談 {ms(H['wrong_other'], 1)} 行")
        res["head"][str(n)] = {"accuracy": H["acc"], "acc_K": H["acc_K"], "acc_N": H["acc_N"], "acc_F": H["acc_F"],
                               "recall": {J[c_]: H["per"][c_] for c_ in ORDER}, "precision": {J[c_]: H["prec"][c_] for c_ in ORDER},
                               "confusions": [{"true": J[t_], "pred": J[p_], "rows": v / k} for (t_, p_), v in H["conf"].most_common()],
                               "cover": H["cover"], "selective_accuracy": H["sel_acc"], "wrong_write": H["wrong_write"], "wrong_other": H["wrong_other"],
                               "H": {a_: list(map(float, b)) for a_, b in hs["H"].items()}, "G": {a_: list(map(float, b)) for a_, b in hs["G"].items()},
                               "auc_H": H["auc_H"], "auc_G": H["auc_G"], "taus": H["taus"]}

        # ---------------------------------------------------------------- 4. 組み合わせ（BERT 案 E × ヘッド）
        if "E" in berts:
            cells = collections.defaultdict(list)
            for _, bp, bhi in berts["E"]:
                bflag = bp >= BERT_LO
                for pred, ok, rej, low in passes:
                    right = pred == act
                    w = np.isin(pred, WRITE)
                    # 機能にない依頼: どちらが止めたか
                    for g, sel in (("all", isH), ("seen", isH & (grp == "seen")), ("new_seen", isH & (grp == "new_seen")), ("unseen", isH & (grp == "unseen"))):
                        cells[f"H_{g}_both"].append(np.mean(bflag[sel] & ~ok[sel]))
                        cells[f"H_{g}_bert_only"].append(np.mean(bflag[sel] & ok[sel]))
                        cells[f"H_{g}_head_only"].append(np.mean(~bflag[sel] & ~ok[sel]))
                        cells[f"H_{g}_none"].append(np.mean(~bflag[sel] & ok[sel]))
                        cells[f"H_{g}_none_write"].append(np.mean(~bflag[sel] & ok[sel] & w[sel]))
                    cells["H_head_miss"].append(np.mean(ok[isH]))
                    cells["H_bert_miss"].append(np.mean(~bflag[isH]))
                    cells["H_head_miss_given_bert_miss"].append(float(np.mean(ok[isH & ~bflag])) if (isH & ~bflag).any() else float("nan"))
                    # 対象の依頼と雑談
                    cells["K_exec_right"].append(np.mean(~bflag[known] & ok[known] & right[known]))
                    cells["K_exec_wrong_write"].append(np.mean(~bflag[known] & ok[known] & ~right[known] & w[known]))
                    cells["K_exec_wrong_other"].append(np.mean(~bflag[known] & ok[known] & ~right[known] & ~w[known]))
                    cells["K_head_stop"].append(np.mean(~bflag[known] & ~ok[known]))
                    cells["K_bert_stop_head_right"].append(np.mean(bflag[known] & ok[known] & right[known]))
                    cells["K_bert_stop_other"].append(np.mean(bflag[known] & ~(ok[known] & right[known])))
            c = {k_: ms(v) for k_, v in cells.items()}
            md += [f"\n## 3. 組み合わせ・層 {n} 個（BERT 案 E のフラグ 0.5 × ヘッドの判定。seed 3 × 3 の平均）\n",
                   "### 機能にない依頼を、どちらが止めたか\n", "|組|両方が止めた|BERT だけが止めた|ヘッドだけが止めた|**どちらも止めなかった**|うち操作系として通した|",
                   "|---|---|---|---|---|---|"]
            for g, lab in (("all", "全体・180 行"), ("seen", "学習に同じ機能がある・60 行"), ("new_seen", "BERT に足した機能・40 行"), ("unseen", "どの学習にも無い機能・80 行")):
                md.append(f"|{lab}|{c[f'H_{g}_both']}|{c[f'H_{g}_bert_only']}|{c[f'H_{g}_head_only']}|{c[f'H_{g}_none']}|{c[f'H_{g}_none_write']}|")
            md += [f"\nヘッドが見逃す割合 {c['H_head_miss']}・BERT が見逃す割合 {c['H_bert_miss']}・**BERT が見逃した行に限ったときの、ヘッドが見逃す割合 {c['H_head_miss_given_bert_miss']}**"
                   "（見逃しが独立なら、ヘッドが見逃す割合と同じになる）\n",
                   "### 対象の依頼と雑談 480 行\n", "|結果|割合|", "|---|---|", f"|両方が通し、行き先も正しい|{c['K_exec_right']}|",
                   f"|両方が通したが、別の操作系と判定|{c['K_exec_wrong_write']}|", f"|両方が通したが、別の読むだけの機能・雑談と判定|{c['K_exec_wrong_other']}|",
                   f"|BERT は通したが、ヘッドが止めた（棄却・確信度）|{c['K_head_stop']}|", f"|ヘッドは正しく通したが、BERT がフラグを立てた|{c['K_bert_stop_head_right']}|",
                   f"|BERT がフラグを立て、ヘッドも止めたか外した|{c['K_bert_stop_other']}|"]
            print(f"  [組み合わせ] 機能にない依頼: 両方 {c['H_all_both']}・BERT だけ {c['H_all_bert_only']}・ヘッドだけ {c['H_all_head_only']}・どちらも止めず {c['H_all_none']}（うち操作系 {c['H_all_none_write']}）")
            print(f"  [組み合わせ] ヘッドの見逃し {c['H_head_miss']}・BERT の見逃し {c['H_bert_miss']}・BERT が見逃した行でのヘッドの見逃し {c['H_head_miss_given_bert_miss']}")
            print(f"  [組み合わせ] 対象: 正しく実行 {c['K_exec_right']}・別の操作系 {c['K_exec_wrong_write']}・別の読むだけ/雑談 {c['K_exec_wrong_other']}・ヘッドが止めた {c['K_head_stop']}・"
                  f"BERT だけが止めた {c['K_bert_stop_head_right']}・BERT も止めた（その他）{c['K_bert_stop_other']}")
            res["combo"][str(n)] = {k_: [float(x) for x in v] for k_, v in cells.items()}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "parts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "parts.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（parts.md・parts.json）")


if __name__ == "__main__":
    main()
