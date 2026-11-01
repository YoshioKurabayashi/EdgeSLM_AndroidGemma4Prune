# -*- coding: utf-8 -*-
"""P36: アプリ v2 の判定（Java）を Python と突き合わせるための、正解の表を作る。あわせて、`p36_decide` が評価の流れと同じ結果を返すかを確かめる。

設計: DoC/設計書_P36_判定デモアプリv2.md の 5 節。`app/p24_fixtures.py` を写して、構成（規則 P31・引数 P35・ヘッド 1,656 行・S1 の seed）と発話を替えたもの。
学習なし・端末なし（numpy と PyYAML だけ。VM の python3 で動く）。

作るもの（`work/p36/fixtures/`。タブ区切り）:

|ファイル|1 行|
|---|---|
|`rules.tsv`|発話・判定（9 通り）→ 規則の後の判定・聞き返すか・当たった規則|
|`args.tsv`|発話・機能（8 つ）→ 引数がそろうか・値・最初のメモ・検査（停車中）・検査（走行中）|
|`flow_L<n>.tsv`|評価セット v10 の行。ヘッド（1,656 行・S1 の seed）の確率・棄却・閾値・BERT（seed 0）の確率 → 結果・止めた段・機能・引数・規則|
|`words.tsv`|突き合わせに使う発話の単語（`data/p30_words/words.tsv` から。Java の単語の分け方が同じかも確かめる）|

発話は、学習 1,656 行の履歴なしの行・評価セット v1〜v10・arg_eval_v1〜v4 のすべて（重なりは除く）。

確かめ（外れたら止める）: 評価セット v10・v9 を `p8_gate.run`（S1 の seed・BERT seed 0 だけ・P35 の構成・改定後の数え方）に流した KPI・止めた行・重大な誤りの行と、
`p36_decide` を 1 行ずつ呼んだ結果が、完全に同じ。

    python -m app.p36_fixtures        # work/p36/fixtures・results/P36/decide_check.md
"""
from __future__ import annotations

import collections
import json

import numpy as np

from .common import PROJECT_ROOT
from .p7_data import read_jsonl
from .p8_gate import STEPS, grade, run
from .p11_bert import MODES, by_mode
from .p13_eval import hi_rule
from .p18_retrain import cached
from .p19_config import bert_check
from .p24_fixtures import ACTIONS, canon, verdict
from .p28_data import TR4
from .p28_verdict import SRC1656
from . import p18_retrain as R
from .p31_check import pick_seed
from .p31_rules import make_rules as rules_p31
from .p32_data import CH9
from .p32_verdict import WORK as P32W, load_eval9
from .p33_check import grading, grade_new
from .p34_data import CH10
from .p34_verdict import WORK as P34W, load_eval10
from .p35_arguments import make_resolve as args_p35
from .p36_decide import decide, load_parts

WORK = PROJECT_ROOT / "work" / "p36"
FIX = WORK / "fixtures"
RES = PROJECT_ROOT / "results" / "P36"
WORDS = PROJECT_ROOT / "data" / "p30_words" / "words.tsv"


def all_utterances() -> list[str]:
    seen, out = set(), []
    sets = [[r for r in read_jsonl(TR4 / "utterances.jsonl") if not r.get("history")]]
    sets += [read_jsonl(PROJECT_ROOT / "data" / f"challenge_v{v}" / "utterances.jsonl") for v in range(1, 11)]
    sets += [read_jsonl(PROJECT_ROOT / "data" / f"arg_eval_v{v}" / "utterances.jsonl") for v in range(1, 5)]
    for rows in sets:
        for r in rows:
            u = r["utterance"]
            if u not in seen and "\t" not in u and "\n" not in u:
                seen.add(u)
                out.append(u)
    return out


def main() -> None:
    parts = load_parts()
    FIX.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    us = all_utterances()
    table = {}
    for line in WORDS.read_text(encoding="utf-8").splitlines():
        u, _, ws = line.partition("\t")
        table[u] = ws
    miss = [u for u in us if u not in table]
    if miss:
        raise SystemExit(f"単語の表に無い発話がある（{len(miss)}）: {miss[:3]}")
    print(f"\n===== P36: 正解の表を作る（発話 {len(us)}）=====", flush=True)
    with open(FIX / "words.tsv", "w", encoding="utf-8", newline="\n") as f:
        for u in us:
            f.write(f"{u}\t{table[u]}\n")
    with open(FIX / "rules.tsv", "w", encoding="utf-8", newline="\n") as f:
        for u in us:
            for a in ACTIONS:
                d = parts["rules"]({"utterance": u, "history": [], "last_tool": None}, a)
                f.write("\t".join([u, a, d.action, "1" if d.demote else "0", d.rule or "-"]) + "\n")
    tools = [a[5:] for a in ACTIONS if a.startswith("TOOL_")]
    n_args = 0
    with open(FIX / "args.tsv", "w", encoding="utf-8", newline="\n") as f:
        for u in us:
            for t in tools:
                ar = parts["resolve"](t, None, u, parts["state"])
                note = (ar.notes[0] if ar.notes else "-").replace("\t", " ")
                f.write("\t".join([u, t, "1" if ar.complete else "0", canon(ar.values), note, verdict(parts, t, ar, "stopped"), verdict(parts, t, ar, "moving")]) + "\n")
                n_args += 1
    print(f"rules.tsv: {len(us) * len(ACTIONS)} 行 / args.tsv: {n_args} 行", flush=True)

    # ---- 評価の流れとの突き合わせ（S1 の seed・BERT seed 0・P35 の構成・改定後の数え方）
    md = ["`p36_decide.decide`（1 発話ずつ）と、評価の流れ `p8_gate.run`（まとめて）の突き合わせ。ヘッドは 1,656 行・mlp・S1 の seed、BERT は seed 0、規則は P31、引数は P35、"
          "棄却の閾値は validation の 2%、数え方は改定後（機能にない依頼の雑談は正解）\n",
          "|評価セット|層|seed|行|食い違った行|EtoE の成功|重大な誤り（件）|不要な聞き返し|通常の誤り|1 回目で正しく処理|", "|---|---|---|---|---|---|---|---|---|---|"]
    info = {"layers": {}}
    bad_total = 0
    seeds = {n: pick_seed(n)[0] for n in (17, 26)}
    sets = (("v10", cached(lambda n_: load_eval10(n_)), P34W), ("v9", cached(lambda n_: load_eval9(n_)), P32W))
    for tag, loader, wdir in sets:
        raw = json.loads((wdir / "text_scores_F.json").read_text(encoding="utf-8"))
        chk = bert_check(raw)
        ts = by_mode(raw, MODES[0], only=(chk["one"],))
        final = STEPS[-1]
        for n in (17, 26):
            grab = {}
            base = R.hook_for(SRC1656)

            def hook(n_, src, heads, grab=grab, base=base, s=seeds[n]):
                heads = [h for h in base(n_, src, heads) if h[0] == s]
                grab["head"] = heads[0]
                return heads

            with grading(True):
                blob = run([n], make_resolve=args_p35, out=None, heads_hook=hook, steps=[final], sources=[SRC1656], quiet=True, ts=ts, watch=set(),
                           bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=rules_p31, amb_chat_ok=True)
            key = f"{n}|{SRC1656}|{final[0]}"
            k, det = blob["kpi"][key], blob["detail"][key]
            seed, classes, th, p = grab["head"]
            rows, _, _ = loader(n)
            bhi = blob["bert_hi"][chk["one"]]
            pred = ts["models"][chk["one"]]["pred"]
            outs = []
            for i, r in enumerate(rows):
                state = {**parts["state"], **r.get("state", {})}
                outs.append(decide(r["utterance"], [float(x) for x in p["action"][i]], classes, float(p["reject"][i]), th, pred[r["id"]], bhi, state, parts))
            grades = [grade_new(r, o["outcome"], o["tool"] if o["outcome"] != "雑談" else None, parts["write"], True) for r, o in zip(rows, outs)]
            known = [r["action"] != "UNKNOWN" for r in rows]
            mine = {"e2e": float(np.mean([g in ("正解", "回復可能") for g in grades])), "first": float(np.mean([g == "正解" for g in grades])),
                    "severe_n": float(sum(g == "重大" for g in grades)), "normal": float(np.mean([g == "通常" for g in grades])),
                    "over": float(np.mean([o["outcome"] == "聞き返し" for o, kn in zip(outs, known) if kn]))}
            diff = [key_ for key_ in mine if abs(mine[key_] - k[key_]["mean"]) > 1e-12]
            stop_ref = collections.Counter({(x["utterance"], x["stop"]): round(x["share"]) for x in det["stop_rows"]})
            stop_mine = collections.Counter((r["utterance"], f"{o['outcome']}（{o['stage']}）") for r, o, kn in zip(rows, outs, known) if kn and o["outcome"] in ("聞き返し", "拒否"))
            sev_ref = collections.Counter({(x["utterance"], x["executed"]): round(x["share"]) for x in det["severe_rows"]})
            sev_mine = collections.Counter((r["utterance"], f"{o['tool']}") for r, o, g in zip(rows, outs, grades) if g == "重大")
            n_bad = len(diff) + sum((stop_ref - stop_mine).values()) + sum((stop_mine - stop_ref).values()) + sum((sev_ref - sev_mine).values()) + sum((sev_mine - sev_ref).values())
            bad_total += n_bad
            print(f"評価セット {tag}・層 {n}・seed {seed}: 食い違い {n_bad} / EtoE {mine['e2e']:.4f}・重大 {mine['severe_n']:.0f} 件・不要な聞き返し {mine['over']:.4f}・通常 {mine['normal']:.4f}"
                  f"（閾値: 棄却 {th['REJECT']:.4f}・操作系 {th['TOOL_write']}・読むだけ {th['TOOL_read']}・雑談 {th['CHAT']} / BERT が断る {bhi:.4f}）", flush=True)
            md.append(f"|{tag}|{n}|{seed}|{len(rows)}|{n_bad}|{mine['e2e']:.3f}|{mine['severe_n']:.0f}|{mine['over']:.3f}|{mine['normal']:.3f}|{mine['first']:.3f}|")
            info["layers"][f"{tag}|{n}"] = {"seed": seed, "kpi": mine, "thresholds": th, "bert_hi": bhi, "classes": classes, "mismatch": n_bad}
            if tag == "v10":
                with open(FIX / f"flow_L{n}.tsv", "w", encoding="utf-8", newline="\n") as f:
                    f.write("#classes\t" + "\t".join(classes) + "\n")
                    f.write("#thresholds\t" + "\t".join(f"{key_}={th[key_]!r}" for key_ in ("CHAT", "TOOL_read", "TOOL_write", "REJECT")) + f"\tBERT_HI={bhi!r}\n")
                    for i, (r, o) in enumerate(zip(rows, outs)):
                        f.write("\t".join([r["utterance"], " ".join(repr(float(x)) for x in p["action"][i]), repr(float(p["reject"][i])), repr(float(pred[r["id"]])),
                                           o["outcome"], o["stage"] or "-", o["tool"] or "-", canon(o["args"]), o["rule"] or "-"]) + "\n")
    md.append(f"\n食い違いの合計: {bad_total}（0 なら、1 発話ずつの判定は評価の流れと同じ）")
    md.append("\nここの KPI は、ヘッドの seed 1 つ（アプリに載せるもの・S1）の数字")
    (RES / "decide_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RES / "decide_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if bad_total:
        raise SystemExit(f"食い違いがある（{bad_total}）")
    print(f"保存しました: {FIX}・{RES / 'decide_check.md'}")


if __name__ == "__main__":
    main()
