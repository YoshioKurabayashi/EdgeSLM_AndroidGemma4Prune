# -*- coding: utf-8 -*-
"""P24: アプリの判定（Java）を Python と突き合わせるための、正解の表を作る。あわせて、`p24_decide` が評価の流れと同じ結果を返すかを確かめる。

設計: DoC/設計書_P24_判定デモアプリ.md の 4 節。学習なし・端末なし（numpy と PyYAML だけ）。

作るもの（`work/p24/fixtures/`。タブ区切り）:

|ファイル|1 行|行数の目安|
|---|---|---|
|`rules.tsv`|発話・判定（9 通り）→ 規則の後の判定・聞き返すか・当たった規則|発話 × 9|
|`args.tsv`|発話・機能（8 つ）→ 引数がそろうか・値・最初のメモ・検査（停車中）・検査（走行中）|発話 × 8|
|`flow_L<n>.tsv`|評価セット v6 の行。ヘッド（seed 7）の確率・棄却・閾値・BERT（seed 0）の確率 → 結果・止めた段・機能・引数|600|

発話は、学習の履歴なしの行・評価セット v1〜v6・arg_eval_v1・v2 のすべて（重なりは除く）。

確かめ（外れたら止める）: 評価セット v6・v5 を `p8_gate.run`（ヘッド seed 7・BERT seed 0 だけ）に流した KPI・止めた行・重大な誤りの行と、
`p24_decide.decide` を 1 行ずつ呼んだ結果が、完全に同じ。

    python -m app.p24_fixtures        # work/p24/fixtures・results/P24/decide_check.md
"""
from __future__ import annotations

import collections
import json

import numpy as np

from .common import PROJECT_ROOT
from .p7_data import CH1, CH2, read_jsonl
from .p8_gate import STEPS, grade, run
from .p11_bert import MODES, by_mode
from .p13_data import CH3
from .p13_eval import hi_rule
from .p18_analyze import hook1584
from .p18_data import CH4, TR3
from .p18_retrain import SRC1584, cached
from .p19_config import bert_check
from .p20_data import CH5
from .p20_verdict import load_eval5
from .p21_data import CH6
from .p21_rules import make_rules as rules_p21
from .p21_verdict import WORK as P21, load_eval6
from .p22_argeval import OUT as ARG1
from .p23_argeval import OUT as ARG2
from .p23_arguments import make_resolve as args_p23
from .p24_decide import decide, load_parts

WORK = PROJECT_ROOT / "work" / "p24"
FIX = WORK / "fixtures"
RES = PROJECT_ROOT / "results" / "P24"
HEAD_SEED = 7
ACTIONS = ["CHAT", "TOOL_control_hvac", "TOOL_control_media", "TOOL_set_navigation", "TOOL_get_vehicle_status", "TOOL_search_vehicle_manual",
           "TOOL_search_poi", "TOOL_get_media_info", "TOOL_capture_camera"]


def canon(values: dict | None) -> str:
    """引数を、Java と同じ形の文字列にする（キーの順・区切り・数の書き方をそろえる）"""
    if values is None:
        return "-"
    parts = []
    for k in sorted(values):
        v = values[k]
        if isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, int):
            s = str(v)
        elif isinstance(v, float):
            s = repr(v)
        else:
            s = '"' + str(v) + '"'
        parts.append(f'"{k}": {s}')
    return "{" + ", ".join(parts) + "}"


def all_utterances() -> list[str]:
    seen, out = set(), []
    sets = [[r for r in read_jsonl(TR3 / "utterances.jsonl") if not r.get("history")], read_jsonl(CH1), read_jsonl(CH2 / "utterances.jsonl"), read_jsonl(CH3 / "utterances.jsonl"),
            read_jsonl(CH4 / "utterances.jsonl"), read_jsonl(CH5 / "utterances.jsonl"), read_jsonl(CH6 / "utterances.jsonl"), read_jsonl(ARG1 / "utterances.jsonl"), read_jsonl(ARG2 / "utterances.jsonl")]
    for rows in sets:
        for r in rows:
            u = r["utterance"]
            if u not in seen and "\t" not in u and "\n" not in u:
                seen.add(u)
                out.append(u)
    return out


def verdict(parts: dict, tool: str, ar, motion: str) -> str:
    if not ar.complete:
        return "-"
    v = parts["policy"].inspect(tool, ar.values, {**parts["state"], "motion": motion})
    return "ok" if v.ok else ("ask" if v.missing else "refuse")


def main() -> None:
    parts = load_parts()
    FIX.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    us = all_utterances()
    print(f"\n===== P24: 正解の表を作る（発話 {len(us)}）=====")
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
    print(f"rules.tsv: {len(us) * len(ACTIONS)} 行 / args.tsv: {n_args} 行")

    # ---- 評価の流れとの突き合わせ（ヘッド seed 7・BERT seed 0）
    raw = json.loads((P21 / "text_scores_F.json").read_text(encoding="utf-8"))
    chk = bert_check(raw)
    ts = by_mode(raw, MODES[0], only=(chk["one"],))
    final = STEPS[-1]
    md = [f"`p24_decide.decide`（1 発話ずつ）と、評価の流れ `p8_gate.run`（まとめて）の突き合わせ。ヘッドは 1,584 行・mlp・seed {HEAD_SEED}、BERT は {chk['one']}、規則は P21、引数は P23、棄却の閾値は validation の 2%\n",
          "|評価セット|層|行|食い違った行|EtoE の成功|重大な誤り（件）|不要な聞き返し|通常の誤り|1 回目で正しく処理|", "|---|---|---|---|---|---|---|---|---|"]
    info = {"bert": chk["one"], "head_seed": HEAD_SEED, "layers": {}}
    bad_total = 0
    for tag, loader, rows_all in (("v6", cached(load_eval6), read_jsonl(CH6 / "utterances.jsonl")), ("v5", cached(load_eval5), read_jsonl(CH5 / "utterances.jsonl"))):
        for n in (17, 26):
            grab = {}

            def hook(n_, src, heads, grab=grab):
                heads = [h for h in hook1584(0.02)(n_, src, heads) if h[0] == HEAD_SEED]
                grab["head"] = heads[0]
                return heads

            blob = run([n], make_resolve=args_p23, out=None, heads_hook=hook, steps=[final], sources=[SRC1584], quiet=True, ts=ts, watch=set(),
                       bert_hi=hi_rule, flag_all=True, eval_loader=loader, make_rules=rules_p21, amb_chat_ok=True)
            key = f"{n}|{SRC1584}|{final[0]}"
            k, det = blob["kpi"][key], blob["detail"][key]
            seed, classes, th, p = grab["head"]
            rows, _, _ = loader(n)
            bhi = blob["bert_hi"][chk["one"]]
            pred = ts["models"][chk["one"]]["pred"]
            outs = []
            for i, r in enumerate(rows):
                state = {**parts["state"], **r.get("state", {})}
                outs.append(decide(r["utterance"], [float(x) for x in p["action"][i]], classes, float(p["reject"][i]), th, pred[r["id"]], bhi, state, parts))
            grades = [grade(r, o["outcome"], o["tool"] if o["outcome"] != "雑談" else None, parts["write"], True) for r, o in zip(rows, outs)]
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
            print(f"評価セット {tag}・層 {n}: 食い違い {n_bad} / EtoE {mine['e2e']:.4f}・重大 {mine['severe_n']:.0f} 件・不要な聞き返し {mine['over']:.4f}・通常 {mine['normal']:.4f}・1 回目 {mine['first']:.4f}"
                  f"（閾値: 棄却 {th['REJECT']:.4f}・操作系 {th['TOOL_write']}・読むだけ {th['TOOL_read']}・雑談 {th['CHAT']} / BERT が断る {bhi:.4f}）")
            md.append(f"|{tag}|{n}|{len(rows)}|{n_bad}|{mine['e2e']:.3f}|{mine['severe_n']:.0f}|{mine['over']:.3f}|{mine['normal']:.3f}|{mine['first']:.3f}|")
            info["layers"][f"{tag}|{n}"] = {"kpi": mine, "thresholds": th, "bert_hi": bhi, "classes": classes, "mismatch": n_bad}
            if tag == "v6":
                with open(FIX / f"flow_L{n}.tsv", "w", encoding="utf-8", newline="\n") as f:
                    f.write("#classes\t" + "\t".join(classes) + "\n")
                    f.write("#thresholds\t" + "\t".join(f"{key_}={th[key_]!r}" for key_ in ("CHAT", "TOOL_read", "TOOL_write", "REJECT")) + f"\tBERT_HI={bhi!r}\n")
                    for i, (r, o) in enumerate(zip(rows, outs)):
                        f.write("\t".join([r["utterance"], " ".join(repr(float(x)) for x in p["action"][i]), repr(float(p["reject"][i])), repr(float(pred[r["id"]])),
                                           o["outcome"], o["stage"] or "-", o["tool"] or "-", canon(o["args"]), o["rule"] or "-"]) + "\n")
    md.append(f"\n食い違いの合計: {bad_total}（0 なら、1 発話ずつの判定は評価の流れと同じ）")
    md.append("\nここの KPI は、ヘッドの seed 1 つ（アプリに載せるもの）の数字。記録にある数字（seed 3 つの平均）とは少し違う")
    (RES / "decide_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RES / "decide_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if bad_total:
        raise SystemExit(f"食い違いがある（{bad_total}）")
    print(f"保存しました: {FIX}・{RES / 'decide_check.md'}")


if __name__ == "__main__":
    main()
