# -*- coding: utf-8 -*-
r"""P37 手順 3: 評価セット v11 を、**端末のアプリ v2 の中だけで**判定させ、PC の判定（`p36_decide`・P35 の構成）と突き合わせる。値まで見た KPI も、端末の判定から出す。

設計: DoC/設計書_P37_新しい評価セットv11で合否.md の 5 節。`app/p36_device.py` の送り込み・端末での実行を使う（P36 のソースは変えない）。

|段|何をするか|
|---|---|
|`prep`|PC の判定（v11 の端末の h（`work/p37`）→ ヘッド 1,656 行・S1 の seed → `p36_decide`）と Gemma の ID 列を `work/p37/device/router/eval_v11.jsonl` に書く。画面の設定は P36 のものを写し、確かめ用の行だけ v11 に替える|
|`deploy`|`eval_v11.jsonl` と設定を、アプリ v2 の置き場へ送る（ほかは P36 で送ったもの）|
|`check`|アプリ v2 に v11 の 600 行を層 17・26 の順に判定させ、PC と突き合わせる → `results/P37/app_check.md`|

    python -m app.p37_app prep
    python -m app.p37_app deploy
    python -m app.p37_app check
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import PROJECT_ROOT
from . import p36_device as D
from .p6_seeds import head_probs, load_pt
from .p7_block import HEADS
from .p7_data import read_jsonl
from .p22_labels import match
from .p24_fixtures import canon
from .p28_verdict import SRC1656, hook1656
from .p31_check import pick_seed
from .p33_check import grade_new
from .p36_decide import decide, load_parts
from .p37_data import CH11, labeled
from .p37_verdict import WORK as P37W, load_eval11

WORK = PROJECT_ROOT / "work" / "p37"
OUT = WORK / "device" / "router"
IO = WORK / "device_io"
RES = PROJECT_ROOT / "results" / "P37"
CFG36 = PROJECT_ROOT / "work" / "p36" / "device" / "router" / "router.json"
LAYERS = (17, 26)


def prep() -> None:
    parts = load_parts()
    cfg = json.loads(CFG36.read_text(encoding="utf-8"))
    bhi = float(cfg["bert"]["hi"])
    raw = json.loads((P37W / "text_scores_F.json").read_text(encoding="utf-8"))
    bert = raw["models"]["bert_s0"]["pred"]
    toks = json.loads((P37W / "token_ids_eval.json").read_text(encoding="utf-8"))
    rows_all = read_jsonl(CH11 / "utterances.jsonl")
    pc = {r["id"]: {} for r in rows_all}
    for n in LAYERS:
        seed = pick_seed(n)[0]
        hd = HEADS[SRC1656](n)
        name = f"heads_mlp_seed{seed}.pt"
        rep = {r["checkpoint"]: r for r in json.loads((hd / "evaluate_report.json").read_text(encoding="utf-8"))["results"]}[name]
        blob = load_pt(hd / name)
        classes = list(blob["metadata"]["action_classes"])
        th = hook1656(n, SRC1656, [(seed, classes, {k: float(v) for k, v in rep["thresholds"].items()}, None)])[0][2]
        rows, last, mean = load_eval11(n)
        p = head_probs(blob, float(rep["temperature"]), last, mean)
        for i, r in enumerate(rows):
            state = {**parts["state"], **r.get("state", {})}
            o = decide(r["utterance"], [float(x) for x in p["action"][i]], classes, float(p["reject"][i]), th, bert[r["id"]], bhi, state, parts)
            pc[r["id"]][str(n)] = {"outcome": o["outcome"], "stage": o["stage"], "tool": o["tool"], "args": None if o["args"] is None else canon(o["args"]),
                                   "rule": o["rule"], "bert": float(bert[r["id"]]), "reject": float(p["reject"][i]), "conf": float(max(p["action"][i]))}
        print(f"層 {n}・seed {seed}: PC の判定を {len(rows)} 行", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "eval_v11.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows_all:
            f.write(json.dumps({"id": r["id"], "utterance": r["utterance"], "motion": r["state"]["motion"], "gemma_ids": toks[r["id"]], "pc": pc[r["id"]]}, ensure_ascii=False) + "\n")
    cfg["eval"] = "router/eval_v11.jsonl"
    (OUT / "router.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存しました: {OUT}（eval_v11.jsonl・router.json）")


def deploy() -> None:
    D.ensure_app()
    for name in ("eval_v11.jsonl", "router.json"):
        D.push(OUT / name, f"router/{name}")
    print("送り終わりました", flush=True)


def kpis(rows: dict, outs: list[tuple], lab: dict) -> dict:
    """outs = [(id, outcome, tool, args の文字列)]。値まで見た KPI（違う値での実行は重大・EtoE から引く）"""
    write = {"control_hvac", "control_media", "set_navigation"}
    g, wrong = [], 0
    for i, o, t, a in outs:
        x = grade_new(rows[i], o, t if o != "雑談" else None, write, True)
        if o == "実行" and i in lab and t == lab[i]["tool"] and x == "正解" and not match(t, lab[i]["expected"], True, json.loads(a)):
            x, wrong = "重大", wrong + 1
        g.append(x)
    known = [rows[i]["action"] != "UNKNOWN" for i, _, _, _ in outs]
    return {"e2e": float(np.mean([x in ("正解", "回復可能") for x in g])), "severe_n": int(sum(x == "重大" for x in g)), "wrong_value": wrong,
            "over": float(np.mean([o == "聞き返し" for (_, o, _, _), k in zip(outs, known) if k])), "normal": float(np.mean([x == "通常" for x in g]))}


def check(layers: str, timeout_s: int) -> None:
    D.IO = IO
    D.run_eval(layers, timeout_s)
    rows = {r["id"]: r for r in read_jsonl(CH11 / "utterances.jsonl")}
    lab = {x["id"]: x for x in labeled()}
    pcs = {json.loads(x)["id"]: json.loads(x) for x in (OUT / "eval_v11.jsonl").read_text(encoding="utf-8").split("\n") if x}
    truth = {}
    tt = WORK / "tok_truth.tsv"
    if tt.exists():
        for line in tt.read_text(encoding="utf-8").split("\n"):
            c = line.split("\t")
            if len(c) >= 2 and c[1]:
                truth[c[0]] = [int(x) for x in c[1].split(" ")]
    md = ["# P37 アプリ v2 で評価セット v11 を判定（端末の中だけ）\n", "端末のアプリ v2 の判定と、PC の判定（`p36_decide`・P35 の構成。h は P37 でお試しアプリから集めたもの）の突き合わせ。値まで見た KPI は、それぞれの判定の引数を正解の引数と比べて出した（改定後の数え方）\n",
          "|層|行|端末で失敗|判定が PC と同じ|Gemma の入力が同じ|BERT の入力が同じ|BERT の確率の差|確信度の差|端末: EtoE・重大（うち違う値）・聞き返し・通常|PC: 同じ|1 回の判定（中央値）|", "|---|---|---|---|---|---|---|---|---|---|---|"]
    info, diffs = {}, []
    for n in [x.strip() for x in layers.split(",")]:
        got = [json.loads(x) for x in (IO / f"router_eval_L{n}.jsonl").read_text(encoding="utf-8").split("\n") if x]
        ok = [g for g in got if not g.get("error")]
        same = [g for g in ok if g.get("pc_same")]
        gem = sum(1 for g in ok if g.get("gemma_ids_match"))
        btok = sum(1 for g in ok if truth.get(g["utterance"]) == g.get("bert_ids")) if truth else None
        d_b = max((abs(g["bert"] - pcs[g["id"]]["pc"][n]["bert"]) for g in ok), default=float("nan"))
        d_c = max((abs(g["conf"] - pcs[g["id"]]["pc"][n]["conf"]) for g in ok if "conf" in g), default=float("nan"))
        kd = kpis(rows, [(g["id"], g["outcome"], g.get("tool"), g.get("args") or "{}") for g in ok], lab)
        kp = kpis(rows, [(i, p["pc"][n]["outcome"], p["pc"][n]["tool"], p["pc"][n]["args"] or "{}") for i, p in pcs.items()], lab)
        wall = float(np.median([g["wall_ms"] for g in ok if g.get("wall_ms") is not None])) if ok else float("nan")
        f = lambda k: f"{k['e2e']:.3f}・{k['severe_n']}（{k['wrong_value']}）・{k['over']:.3f}・{k['normal']:.3f}"
        md.append(f"|{n}|{len(got)}|{len(got) - len(ok)}|{len(same)}|{gem}|{'—' if btok is None else btok}|{d_b:.1e}|{d_c:.1e}|{f(kd)}|{f(kp)}|{wall:.0f} ms|")
        print(f"層 {n}: {len(got)} 行・判定が PC と同じ {len(same)}・Gemma の入力が同じ {gem}・BERT の入力が同じ {btok}・端末 {f(kd)}・PC {f(kp)}・中央値 {wall:.0f} ms", flush=True)
        for g in ok:
            if not g.get("pc_same"):
                p = pcs[g["id"]]["pc"][n]
                diffs.append(f"- 層 {n}: {g['utterance']} … PC {p['outcome']}（{p['stage'] or '—'}）{p['tool'] or ''} {p['args'] or ''} ／ 端末 {g['outcome']}（{g.get('stage') or '—'}）{g.get('tool') or ''} {g.get('args') or ''}")
        info[n] = {"rows": len(got), "same": len(same), "gemma_same": gem, "bert_ids_same": btok, "bert_absdiff_max": d_b, "conf_absdiff_max": d_c, "kpi_device": kd, "kpi_pc": kp, "wall_ms_median": wall}
    md += ["\n## PC と判定が違った行\n"] + (diffs or ["なし"])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "app_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RES / "app_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存しました: {RES / 'app_check.md'}", flush=True)


def truth() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "work" / "p20" / "bert_F" / "bert_s0"))
    with open(WORK / "tok_truth.tsv", "w", encoding="utf-8", newline="\n") as f:
        for r in read_jsonl(CH11 / "utterances.jsonl"):
            f.write(r["utterance"] + "\t" + " ".join(str(i) for i in tok(r["utterance"], truncation=True, max_length=64)["input_ids"]) + "\n")
    print(f"保存しました: {WORK / 'tok_truth.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "truth", "deploy", "check"])
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()
    print(f"\n===== P37 app {a.step} =====", flush=True)
    if a.step == "check":
        check(a.layers, a.timeout)
    else:
        {"prep": prep, "truth": truth, "deploy": deploy}[a.step]()


if __name__ == "__main__":
    main()
