# -*- coding: utf-8 -*-
"""P15: 利用者の確認（2026-10-04）を受けて、数え方を直して測り直す。学習なし・端末なし（numpy と PyYAML だけ）。

設計: DoC/設計書_P15_利用者の確認を受けた数え方.md ｜ 評価の決まり: DoC/評価方針_EtoEとエラーバジェット.md（4 節を改定）

利用者の判断:
1. 車両の状態か説明書か … Claude が付けた正解のまま
2. 短い問いかけ（「オイルは汚れてない」）… 文脈で変わるが、この短文だけなら質問と見る（正解のまま）
3. 雑談と聞き返しは分けにくい。雑談は、必要に応じて聞き返す対話。**意図が不明の発話を雑談として受けるのは誤りではない**
4. 機能にない依頼（冷蔵庫・座面の送風）は断るのが正しい（正解のまま）

変わるのは 3 だけ: 意図が不明の発話の結果が「雑談」のとき、これまでは通常の誤り、これからは正解。
実行したとき（読むだけの機能・操作系）と断ったときは、これまでどおり誤り。

    python -m app.p15_rescore        # results/P15/rescore.md・rescore.json
"""
from __future__ import annotations

import argparse
import collections
import json

from .common import PROJECT_ROOT
from .p7_data import CH2, read_jsonl
from .p8_gate import STEPS, run
from .p9_arguments import make_resolve
from .p10_reject import SRC, make_hook
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import WORK as P13, cell, hi_rule, interval, load_eval3
from .p13_rules import make_rules as make_p13
from .p14_rules import make_rules as make_p14

RES = PROJECT_ROOT / "results" / "P15"
CONFIGS = [("P12 の構成", "C", None), ("P13 の本命（BERT に機能を足す＋規則を補う）", "E", make_p13), ("P14（＋対象の語の食い違い）", "E", make_p14)]
KPI = [("EtoE の成功", "e2e", 3), ("重大な誤り（件）", "severe_n", 1), ("重大な誤りの割合", "severe", 4), ("不要な聞き返し", "over", 3),
       ("通常の誤り", "normal", 3), ("1 回目で正しく処理", "first", 3), ("意図が不明を正しく受けた", "amb_ok", 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    tss = {k: by_mode(json.loads((P13 / f"text_scores_{k}.json").read_text(encoding="utf-8")), MODES[0]) for k in ("C", "E")}
    final = STEPS[-1]
    sets = [("評価セット v3", read_jsonl(CH3 / "utterances.jsonl"), load_eval3), ("評価セット v2", read_jsonl(CH2 / "utterances.jsonl"), None)]
    print("\n===== P15: 数え方を直して測り直す（意図が不明を雑談として受けるのは正解）=====")
    md = ["数字はヘッドの seed 3 つ × BERT の seed 3 つの平均（かっこは最小〜最大）。区間は 95% 信頼区間の目安（Wilson）。"
          "「これまで」= 意図が不明を雑談として受けたら通常の誤り。「これから」= 正解\n"]
    res = {"kpi": {}, "detail": {}}
    for sname, rows, loader in sets:
        for n in ns:
            lines = []
            for cname, bert, mk in CONFIGS:
                for gname, flag in (("これまで", False), ("これから", True)):
                    blob = run([n], make_resolve=make_resolve, out=None, heads_hook=make_hook("val", 0.02, {}), steps=[final], sources=[SRC], quiet=True,
                               ts=tss[bert], watch=set(), bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=mk, amb_chat_ok=flag)
                    key = f"{n}|{SRC}|{final[0]}"
                    if key not in blob["kpi"]:
                        continue
                    k, d = blob["kpi"][key], blob["detail"][key]
                    cells = [cell(k[x], nd) for _, x, nd in KPI] + [interval(k["e2e"]["mean"], len(rows)), interval(k["severe"]["mean"], len(rows))]
                    res["kpi"][f"{sname}|{n}|{cname}|{gname}"] = {**{x: k[x] for _, x, _ in KPI}, "e2e_ci": cells[-2], "severe_ci": cells[-1]}
                    if flag:
                        res["detail"][f"{sname}|{n}|{cname}"] = {x: d[x] for x in ("stages", "severe_rows", "normal_rows", "by_group")}
                    lines.append((cname, gname, cells, d))
            if not lines:
                continue
            heads = [lab for lab, _, _ in KPI] + ["EtoE の成功の区間", "重大な誤りの割合の区間"]
            print(f"\n===== {sname}（{len(rows)} 行）・層 {n} 個 =====")
            md += [f"\n### {sname}（{len(rows)} 行）・層 {n} 個\n", "|構成|数え方|" + "|".join(heads) + "|", "|---|---|" + "---|" * len(heads)]
            for cname, gname, cells, _ in lines:
                print(f"{cname[:12]:<12} {gname} " + "  ".join(f"{h}: {x}" for h, x in zip(heads, cells)))
                mark = "**" if gname == "これから" else ""
                md.append(f"|{cname}|{mark}{gname}{mark}|" + "|".join(cells) + "|")
            d = lines[-1][3]                                           # P14・これから
            c = collections.Counter()
            for x in d["by_group"]:
                if x["grade"] == "通常":
                    c[(x["category"], x["group"] if x["category"] in "HG" else "")] += x["rows"]
            names = {"H": "機能にない依頼", "K": "対象（機能の語を含む）", "N": "対象（ふつうの依頼）", "F": "雑談", "G": "意図が不明"}
            agg = collections.Counter()
            for (cat, _), v in c.items():
                agg[names[cat]] += v
            line = "・".join(f"{k} {v:.1f}" for k, v in agg.most_common())
            print(f"  P14・これからの数え方で残る通常の誤り（行数）: {line}")
            md += [f"\nP14・これからの数え方で残る通常の誤り（行数）: {line}\n", "|種類|発話 → 結果（出た割合）|", "|---|---|"]
            left = collections.defaultdict(list)
            for x in d["normal_rows"]:
                left[names[x["category"]]].append(f"{x['utterance']} → {x['outcome']}（{x['share']:.2f}）")
            md += [f"|{k}|" + " / ".join(v[:24]) + "|" for k, v in left.items()]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rescore.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "rescore.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（rescore.md・rescore.json）")


if __name__ == "__main__":
    main()
