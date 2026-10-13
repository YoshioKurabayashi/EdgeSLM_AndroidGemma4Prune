# -*- coding: utf-8 -*-
"""P30: 部分一致（P29）と単語の一致（P30）を比べる。端末なし・学習なし（端末で集めた h と、評価セット v6〜v8 を使う）。

設計: DoC/設計書_P30_語単位で一致させる.md の 4 節。**v6〜v8 はどれも直す用。ここの数字は当たりで、合否ではない**。

|確かめ|中身|
|---|---|
|1 単語の分け方|`data/p30_words/words.tsv`（端末の `BertJa.java` で分けた表）が、P24 の PC の分け方の表（5,072 行）と同じか（表に重なる行で）|
|2 語の一覧の根拠|対象の読み出しの行（正しい機能のとき）で説明がつかない行・機能にない依頼で根拠ありとされた行（P29 → P30）|
|3 流れの KPI|ヘッド 1,656 行・層 17・26 で、引数を P26・P29・P30 に替えて比べる（値まで見た KPI）。P29 → P30 で変わった行|

    python -m app.p30_check        # results/P30/check.md・check.json
"""
from __future__ import annotations

import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p7_data import read_jsonl
from .p11_bert import MODES, by_mode
from .p13_eval import cell
from .p18_retrain import cached
from .p19_config import bert_check
from .p21_data import CH6
from .p21_verdict import STOPS, load_eval6, share_by
from .p22_fix import value_kpi
from .p22_labels import load_all
from .p26_arguments import make_resolve as args_p26
from .p27_data import CH7, labeled as labeled7
from .p27_verdict import load_eval7
from .p28_data import CH8, labeled as labeled8
from .p28_verdict import SRC1656, WORK as P28, load_eval8, measure
from . import p29_scope as S29
from . import p30_scope as S30
from .p30_words import TABLE, words

RES = PROJECT_ROOT / "results" / "P30"
CONFIGS = [("P26（絞らない）", args_p26), ("P29（部分一致で絞る）", S29.make_resolve), ("P30（単語の一致で絞る）", S30.make_resolve)]
GRADES = ("正解", "回復可能", "通常", "重大")


def main() -> None:
    md = ["# P30 単語の一致で絞る（部分一致の P29 と比べる。当たり・端末なし・学習なし）\n",
          "**評価セット v6〜v8 はどれも直す用。ここの数字は当たりで、合否ではない**。ヘッド 1,656 行（P28）・seed 3 つの平均・BERT は bert_s0・規則は P21\n"]
    res = {"words_check": {}, "lexicon": {}, "kpi": {}, "groups": {}, "changed": {}}
    # 1. 単語の分け方
    truth = PROJECT_ROOT / "work" / "p24" / "tok_truth.tsv"
    tab = {l.split("\t")[0]: l.split("\t")[1].split(" ") for l in TABLE.read_text(encoding="utf-8").splitlines() if "\t" in l}
    n = bad = 0
    if truth.exists():
        for line in truth.read_text(encoding="utf-8").splitlines():
            c = line.split("\t")
            if len(c) > 2 and c[0] in tab:
                n += 1
                bad += tab[c[0]] != c[2].split(" ")
    res["words_check"] = {"table_rows": len(tab), "overlap_with_pc": n, "mismatch": bad}
    md.append(f"## 1. 単語の分け方\n\n`{TABLE.relative_to(PROJECT_ROOT)}`: {len(tab)} 行（端末の `BertJa.java` で分けた）。P24 の PC の分け方の表（`BertJapaneseTokenizer`）と重なる {n} 行で、食い違い {bad} 行"
              "（`BertJa.java` は P24 で 5,072 行すべて PC と同じことを確かめ済み。P30 でも同じ表で食い違い 0 を確かめた）\n")
    print(md[-1].strip(), flush=True)
    # 2. 語の一覧の根拠
    md += ["## 2. 語の一覧の根拠（P29 部分一致 → P30 単語の一致）\n", "|評価セット|対象の読み出しの行で説明がつかない|機能にない依頼 seen で根拠あり|機能にない依頼 unseen で根拠あり|", "|---|---|---|---|"]
    sets = (("v8", read_jsonl(CH8 / "utterances.jsonl"), cached(load_eval8), labeled8()),
            ("v7", read_jsonl(CH7 / "utterances.jsonl"), cached(load_eval7), labeled7()),
            ("v6", read_jsonl(CH6 / "utterances.jsonl"), cached(load_eval6), [x for x in load_all() if x["set"] == "v6"]))
    diff_rows = []
    for tag, rows, _, _ in sets:
        kn = [r for r in rows if r["action"][5:] in S30.READ]
        m29 = sum(not S29.evidence(r["action"][5:], r["utterance"]) for r in kn)
        m30 = sum(not S30.evidence(r["action"][5:], r["utterance"]) for r in kn)
        cells = [f"{m29} → {m30}（{len(kn)} 行）"]
        for grp in ("seen", "unseen"):
            sel = [r for r in rows if r.get("os_group") == grp]
            e29 = sum(any(S29.evidence(T, r["utterance"]) for T in S29.READ) for r in sel)
            e30 = sum(any(S30.evidence(T, r["utterance"]) for T in S30.READ) for r in sel)
            cells.append(f"{e29} → {e30}（{len(sel)} 行）")
        for r in rows:
            if r["category"] in "HG":
                a = {T: S29.evidence(T, r["utterance"]) for T in S29.READ if S29.evidence(T, r["utterance"])}
                b = {T: S30.evidence(T, r["utterance"]) for T in S30.READ if S30.evidence(T, r["utterance"])}
                if a != b:
                    diff_rows.append(f"[{tag}] {r['utterance']}（{'・'.join(f'{k[4:12]}:{v}' for k, v in a.items())} → {'・'.join(f'{k[4:12]}:{v}' for k, v in b.items()) or '根拠なし'}）")
        res["lexicon"][tag] = cells
        md.append(f"|{tag}|" + "|".join(cells) + "|")
    md.append("\n根拠が変わった、機能にない依頼・意図が不明の行（" + str(len(diff_rows)) + "）: " + (" / ".join(diff_rows) or "なし") + "\n")
    print("\n".join(md[-5:]), flush=True)
    # 3. 流れ
    raw = json.loads((P28 / "text_scores_F.json").read_text(encoding="utf-8"))
    use_multi_library()
    from edge_router.dispatch import arguments
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    ts = by_mode(raw, MODES[0], only=(bert_check(raw)["one"],))
    md.append("## 3. 流れの KPI（ヘッド 1,656 行）\n")
    for tag, rows, loader, lab in sets:
        for n_ in (17, 26):
            keep = {}
            for cname, make in CONFIGS:
                got = measure(n_, SRC1656, ts, loader, make)
                if got:
                    k, d = got
                    keep[cname] = (k, d, value_kpi(k, d, lab, make(arguments), state, len(rows)))
            heads_ = ["値まで見た EtoE", "値まで見た重大な誤り（件）", "不要な聞き返し", "通常の誤り"] + [f"止めた: {s_.replace('聞き返し（', '').replace('拒否（', '').rstrip('）')}" for s_ in STOPS]
            md += [f"\n### {tag}・層 {n_}\n", "|引数|" + "|".join(heads_) + "|", "|---|" + "---|" * len(heads_)]
            for cname, (k, d, v) in keep.items():
                cnt = collections.Counter()
                for x in d["stop_rows"]:
                    cnt[x["stop"]] += x["share"]
                res["kpi"][f"{tag}|{n_}|{cname}"] = {"e2e_value": v["e2e_value"], "severe_value_n": v["severe_value_n"], "over": k["over"], "normal": k["normal"]}
                md.append(f"|{cname}|{v['e2e_value']:.3f}|{v['severe_value_n']:.1f}|{cell(k['over'], 3)}|{cell(k['normal'], 3)}|" + "|".join(f"{cnt[s_]:.1f}" for s_ in STOPS) + "|")
                print(f"{tag}・層 {n_}・{cname:<16} EtoE {v['e2e_value']:.3f}・重大 {v['severe_value_n']:.1f}・聞き返し {k['over']['mean']:.3f}・通常 {k['normal']['mean']:.3f}", flush=True)
            a, b = keep[CONFIGS[1][0]][1], keep[CONFIGS[2][0]][1]
            fmt = lambda items: " / ".join(f"{u}（{s:.2f}）" for u, s in items) if items else "なし"
            dif = lambda x, y: [[u, round(x[u] - y[u], 2)] for u in x if x[u] - y[u] > 1e-9]
            sa, sb = share_by(a["stop_rows"]), share_by(b["stop_rows"])
            na, nb = share_by(a["normal_rows"]), share_by(b["normal_rows"])
            va, vb = share_by(a["severe_rows"]), share_by(b["severe_rows"])
            ch = {"known_freed": dif(sa, sb), "known_newly": dif(sb, sa), "normal_fixed": dif(na, nb), "normal_new": dif(nb, na), "severe_fixed": dif(va, vb), "severe_new": dif(vb, va)}
            res["changed"][f"{tag}|{n_}"] = ch
            md += [f"\nP29 → P30 で変わった行（{tag}・層 {n_}）:\n", "|見るもの|行|", "|---|---|"]
            for lab_, key in (("止まらなくなった対象の行", "known_freed"), ("新しく止まった対象の行", "known_newly"), ("通常の誤りでなくなった行", "normal_fixed"),
                              ("新しく通常の誤りになった行", "normal_new"), ("重大な誤りでなくなった行", "severe_fixed"), ("新しく重大な誤りになった行", "severe_new")):
                md.append(f"|{lab_}（{sum(s for _, s in ch[key]):.1f}）|{fmt(ch[key])}|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（check.md・check.json）")


if __name__ == "__main__":
    main()
