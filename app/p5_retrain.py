# -*- coding: utf-8 -*-
"""P5 手順 2: 足した発話（train_add_v1）の `h` を端末で集め、714 行と合わせてヘッドを学習し直し、評価セット（challenge_v1）で前と後を比べる。

設計: DoC/設計書_P5_学習データを足してヘッドを学習し直す.md の 4 節。

バックボーン（Gemma 4 E2B の層 17・26・35 のグラフ）は変えない。**変えるのはヘッドの学習データだけ**。
評価セットの `h` は P4 で端末から集めたもの（`work/p4/device_L<n>.json`）をそのまま使う（端末で流し直さない）。

|段|何をするか|venv|
|---|---|---|
|`encode`|足した 219 行を F2 の文にし、画面と同じチャットテンプレートで包んでトークン列にする。714 行の符号化と同じ作りかを確かめる|EdgeSLM_Android|
|`device`|お試しアプリで、足した行を層 17・26・35 のグラフに流し、`h`（最後・全位置の平均）を集める|EdgeSLM_Android|
|`heads`|P2 の 714 行の `h` と足した行の `h` を繋ぎ、ヘッドを学習・評価・交差検証（EdgeSLM_MultiHead_Multi のコードをそのまま呼ぶ）。画面と同じ形に書き出す|EdgeSLM_MultiHead_Multi|
|`judge`|評価セットを、前のヘッド（714 行・`work/p3/L<n>/head`）と後のヘッド（933 行・`work/p5/L<n>/head`）で判定し、種類ごとに並べる|EdgeSLM_Android|

    python -m app.p5_retrain encode
    python -m app.p5_retrain device --layers 17,26,35
    python -m app.p5_retrain heads --layers 17,26,35
    python -m app.p5_retrain judge --layers 17,26,35
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

import numpy as np

from .common import HF_DIR, PROJECT_ROOT, SEQUENCE_CHECKPOINT, use_multi_library
from .g3_headlib import export_head, final_of, load_exported, numpy_head
from .p5_train_data import CHALLENGE_DISPUTED, CHALLENGE_H_SEEN, TYPE_NOTE

DATA = PROJECT_ROOT / "data" / "train_add_v1"
P2 = PROJECT_ROOT / "work" / "p2"
P3 = PROJECT_ROOT / "work" / "p3"
P4W = PROJECT_ROOT / "work" / "p4"
WORK = PROJECT_ROOT / "work" / "p5"
RES = PROJECT_ROOT / "results" / "P5"
T = 64
ORDER = ["A", "B", "C", "D", "E", "I", "F", "G", "H", "V"]
SAFE = ("REJECT", "CLARIFY")


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def layers_of(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", ",").split(",") if x]


# ---------------------------------------------------------------- encode

def encode() -> None:
    from transformers import AutoTokenizer
    from .p2_device import G1
    from .p4_eval import render_f2
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    sj = P3 / "L17" / "head" / "samples.json"
    if not sj.exists():
        raise SystemExit(f"{sj} が無い（チャットテンプレートの前後を読む）。先に python -m app.p3_assets --layers 17")
    o = json.loads(sj.read_text(encoding="utf-8"))
    pre, suf = o["chat_prefix"], o["chat_suffix"]
    enc = lambda r: tok(pre + render_f2(r) + suf, add_special_tokens=False)["input_ids"]
    # 確かめ: 714 行のうち履歴の無い行を同じ作りで符号化すると、ヘッドの学習に使ったトークン列（G1）と一致する
    g1 = json.loads((G1 / "token_ids.json").read_text(encoding="utf-8"))
    base = [r for r in read_jsonl(DATA / "utterances.jsonl") if "added" not in r and not r["history"]]
    pick = base[:: max(1, len(base) // 40)]
    bad = [r["id"] for r in pick if enc(r) != g1[r["id"]]]
    print(f"符号化の確かめ（714 行のうち履歴なし {len(pick)} 行 ↔ G1 のトークン列）: 不一致 {len(bad)}")
    if bad:
        raise SystemExit(f"714 行と同じ符号化にならない: {bad[:5]}")
    rows = read_jsonl(DATA / "added.jsonl")
    ids = {r["id"]: enc(r) for r in rows}
    over = [(i, len(v)) for i, v in ids.items() if len(v) > T]
    if over:
        raise SystemExit(f"{T} トークンを超える行がある（学習データなので外さず止める）: {over}")
    ln = [len(v) for v in ids.values()]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids_added.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"足した {len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最小 {min(ln)}・最大 {max(ln)}（グラフは {T} まで）")


# ---------------------------------------------------------------- device

def device(ns: list[int]) -> None:
    from . import p2_device as D
    D.ensure_app()
    ids = json.loads((WORK / "token_ids_added.json").read_text(encoding="utf-8"))
    keys = list(ids)
    for n in D.graphs(ns):
        f = WORK / f"device_L{n}.json"
        if f.exists() and len(json.loads(f.read_text(encoding="utf-8"))["rows"]) == len(keys):
            print(f"層 {n}: 済み（{f.name}。やり直すならこのファイルを別の場所へ移す）", flush=True)
            continue
        got = []
        for k, s in enumerate(range(0, len(keys), D.BATCH)):
            part = keys[s:s + D.BATCH]
            res = D.run_job({**D.base_job(f"p5_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
            rs = res.get("results", [])
            bad = [(i, r.get("error")) for i, r in zip(part, rs) if not r.get("ok")]
            if len(rs) != len(part) or bad:
                raise SystemExit(f"層 {n}・{k}: 通らない行 {bad[:3]}（{len(rs)}/{len(part)}）")
            got += [{"id": i, "last": r["h_last"], "mean": r["h_mean"], "run_ms": r["run_ms"]} for i, r in zip(part, rs)]
            print(f"  層 {n} 個 {k + 1}: {len(part)} 行・NPU 中央値 {np.median([r['run_ms'] for r in rs]):.1f}ms", flush=True)
        f.write_text(json.dumps({"rows": got}), encoding="utf-8")
        print(f"層 {n}: {len(got)} 行 → {f}", flush=True)


# ---------------------------------------------------------------- heads（特徴を繋ぐ → 学習・評価・交差検証 → 書き出し）

def features(n: int) -> None:
    src = P2 / f"L{n:02d}" / "device" / "features"
    z = np.load(src / "features_F2.npz")
    meta = json.loads((src / "features_F2.meta.json").read_text(encoding="utf-8"))
    rows = read_jsonl(DATA / "utterances.jsonl")
    ids = [r["id"] for r in rows]
    k = len(meta["ids"])
    if ids[:k] != meta["ids"]:
        raise SystemExit(f"層 {n}: 合わせた学習データの先頭 {k} 行が、P2 の特徴の並びと違う")
    got = {r["id"]: r for r in json.loads((WORK / f"device_L{n}.json").read_text(encoding="utf-8"))["rows"]}
    miss = [i for i in ids[k:] if i not in got]
    if miss:
        raise SystemExit(f"層 {n}: 端末の h が無い行 {miss[:5]}（先に device）")
    last = np.vstack([z["last__last"], np.array([got[i]["last"] for i in ids[k:]], np.float32)]).astype(np.float32)
    mean = np.vstack([z["last__mean"], np.array([got[i]["mean"] for i in ids[k:]], np.float32)]).astype(np.float32)
    d = WORK / f"L{n:02d}" / "features"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "features_F2.npz", last__last=last, last__mean=mean)
    m = {"format": "F2", "ids": ids, "layers": ["last"], "poolings": ["last", "mean"],
         "meta": {"source": f"device (Xperia 5 III, 層 {n} 個のまとめ読み T=64, QNN HTP)。先頭 {k} 行は P2、残りは P5",
                  "n_layers_kept": n, "mean": "全位置の平均"}}
    (d / "features_F2.meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    toks = json.loads((src / "token_ids.json").read_text(encoding="utf-8"))
    toks.update(json.loads((WORK / "token_ids_added.json").read_text(encoding="utf-8")))
    (d / "token_ids.json").write_text(json.dumps(toks), encoding="utf-8")
    a, b = np.linalg.norm(last[:k], axis=1), np.linalg.norm(last[k:], axis=1)
    print(f"層 {n}: 特徴 {last.shape}（714 行＋{len(ids) - k} 行）/ h の大きさ（最後の位置）714 行 {a.mean():.1f}・足した行 {b.mean():.1f}", flush=True)


def export(n: int) -> None:
    import torch
    from edge_router.dispatch import policy
    from edge_router.heads.model import MultiHead
    base = WORK / f"L{n:02d}"
    head, hmeta = MultiHead.load(base / "heads" / SEQUENCE_CHECKPOINT, map_location="cpu")
    report = next(x for x in json.loads((base / "heads" / "evaluate_report.json").read_text(encoding="utf-8"))["results"]
                  if x["checkpoint"] == SEQUENCE_CHECKPOINT)
    head.temperature.fill_(float(report["temperature"]))
    meta, flat = export_head(head, hmeta, report, policy.load_tools(),
                             {"checkpoint": str(base / "heads" / SEQUENCE_CHECKPOINT), "variant": f"prune_L{n}_add",
                              "mean": f"層 {n} 個のグラフの端末の h・714 行＋train_add_v1（P5）", "mean_skip_ids": []})
    out = base / "head"
    out.mkdir(parents=True, exist_ok=True)
    flat.tofile(out / "head.bin")
    flat2 = np.fromfile(out / "head.bin", dtype="<f4")
    z = np.load(base / "features" / "features_F2.npz")
    last, mean = z["last__last"].astype(np.float64), z["last__mean"].astype(np.float64)
    mine = numpy_head(meta, flat2, last, mean)
    with torch.no_grad():
        tp = head.probabilities(torch.from_numpy(last.astype(np.float32)),
                                torch.from_numpy(np.hstack([last, mean]).astype(np.float32)))
    dp = float(np.abs(mine["action"] - tp["action"].numpy()).max())
    dr = float(np.abs(mine["reject"] - tp["reject"][:, 1].numpy()).max())
    meta["self_check"] = {"rows": len(last), "action_prob_absdiff_max": dp, "reject_absdiff_max": dr}
    print(f"層 {n}: ヘッドの書き出し（温度 {float(report['temperature']):.2f}・閾値 {report['thresholds']}）/ "
          f"自己検査（画面と同じ式 ↔ torch、{len(last)} 行）確率の差 {dp:.2e}・棄却 {dr:.2e}", flush=True)
    if dp > 1e-4 or dr > 1e-4:
        raise SystemExit("ヘッドの書き出しが torch と一致しない。止める")
    (out / "head.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def heads(ns: list[int], force: bool) -> None:
    use_multi_library()
    from edge_router.heads import crossval, evaluate, train
    from .p1_compare import summarize_crossval, summarize_evaluate

    def run(module, argv: list[str]) -> None:
        print(f"\n===== {module.__name__} =====", flush=True)
        sys.argv = [module.__name__] + argv
        module.main()

    for n in ns:
        base = WORK / f"L{n:02d}"
        if not (WORK / f"device_L{n}.json").exists():
            print(f"層 {n}: 足した行の端末の h が無い（先に device）。飛ばす", flush=True)
            continue
        print(f"\n########## 層 {n} 個（714 行＋足した行）##########", flush=True)
        features(n)
        if (base / "crossval" / "crossval_report.json").exists() and not force:
            print(f"層 {n}: ヘッドは作成済み（作り直すなら --force）", flush=True)
        else:
            common = ["--data", str(DATA / "utterances.jsonl"), "--features", str(base / "features")]
            run(train, common + ["--out", str(base / "heads"), "--force"])
            run(evaluate, common + ["--run", str(base / "heads")])
            run(crossval, common + ["--out", str(base / "crossval")])
        export(n)

    rows = []
    print(f"\n{'学習データ':<26} {'交差検証 mlp':>16} {'全基準':>6} {'誤実行':>6} {'棄却再現':>8} {'test s7':>8}")
    for n in ns:
        for src, base in (("714 行（P2）", P2 / f"L{n:02d}" / "device"), ("933 行（P5）", WORK / f"L{n:02d}")):
            cv = summarize_crossval(base / "crossval" / "crossval_report.json")
            ev = summarize_evaluate(base / "heads" / "evaluate_report.json")
            if not cv:
                continue
            m, t = cv["mlp"], ev.get("mlp_s7_test_accuracy")
            rows.append({"n_layers": n, "data": src, "crossval_mlp": m, "test_mlp_s7": t, "test_failed": ev.get("mlp_s7_checks_failed")})
            print(f"層 {n:>2}・{src:<18} {m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f} {m['configs_all_ok']}/{m['configs']:<4} "
                  f"{m['wrong_writes']:>6} {m['rejection_recall_mean']:>8.3f} {t if t is None else f'{t:.4f}':>8}", flush=True)
            if m["failed_checks"]:
                print(f"    満たさなかった基準: {m['failed_checks']}", flush=True)
    print("（714 行と 933 行は測っている行が違うので、数字をそのまま比べない。933 行には長い文・口語・機能にない依頼が入っている）")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "heads_compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- judge

def decide(rows: list[dict], meta: dict, p: dict) -> list[dict]:
    """行ごとの判定（p4_eval.judge_one と同じ分け方）。p は {"action": [行, クラス], "reject": [行]}。P6 からも使う"""
    cls = meta["action_classes"]
    write = {"TOOL_" + k for k, v in meta["tool_effect"].items() if v == "write"}
    out = []
    for k, r in enumerate(rows):
        fin = final_of(meta, p["action"][k], float(p["reject"][k]))
        exp = r["action"]
        if exp == "MULTI":
            kind = "参考"
        elif exp == "UNKNOWN":
            kind = "確認へ" if fin in SAFE else "雑談へ" if fin == "CHAT" else "誤実行" if fin in write else "読むだけの機能へ"
        else:
            kind = "正解" if fin == exp else "確認へ" if fin in SAFE else "誤実行" if fin in write else "別の行き先"
        out.append({"id": r["id"], "category": r["category"], "utterance": r["utterance"], "expected": exp, "final": fin,
                    "top1": cls[int(p["action"][k].argmax())], "conf": round(float(p["action"][k].max()), 4),
                    "reject": round(float(p["reject"][k]), 4), "kind": kind,
                    "ok": kind == "正解" or (exp == "UNKNOWN" and kind == "確認へ")})
    return out


def challenge_h(n: int, rows: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """評価セットの行（端末の h がある行だけ）と、その h（最後・全位置の平均）。P4 で集めたもの"""
    got = {r["id"]: r for r in json.loads((P4W / f"device_L{n}.json").read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in rows if r["id"] in got]
    return (rows, np.array([got[r["id"]]["last"] for r in rows], np.float64),
            np.array([got[r["id"]]["mean"] for r in rows], np.float64))


def judge_with(n: int, rows: list[dict], head_dir, reject_tau: float | None = None) -> dict:
    """p4_eval.judge_one と同じ判定（ヘッドの置き場だけ引数にした）。`reject_tau` を渡すと、棄却の閾値だけその値に置き換える"""
    meta, flat = load_exported(head_dir)
    if reject_tau is not None:
        meta = {**meta, "thresholds": {**meta["thresholds"], "REJECT": float(reject_tau)}}
    rows, last, mean = challenge_h(n, rows)
    return {"thresholds": meta["thresholds"], "temperature": meta["temperature"], "rows": decide(rows, meta, numpy_head(meta, flat, last, mean))}


def summarize(out: list[dict]) -> dict:
    sc = [r for r in out if r["category"] != "M"]
    known = [r for r in sc if r["expected"] != "UNKNOWN"]
    unk = [r for r in sc if r["expected"] == "UNKNOWN"]
    ww = [r for r in sc if r["kind"] == "誤実行"]

    def part(rs: list[dict]) -> dict:
        return {"n": len(rs), "ok": sum(r["ok"] for r in rs), "clarify": sum(r["kind"] == "確認へ" for r in rs),
                "wrong_write": sum(r["kind"] == "誤実行" for r in rs)}

    h = [r for r in sc if r["category"] == "H"]
    return {"n": len(sc), "ok": sum(r["ok"] for r in sc),
            "accuracy_known": float(np.mean([r["kind"] == "正解" for r in known])),
            "over_clarify_known": float(np.mean([r["kind"] == "確認へ" for r in known])),
            "reject_recall": float(np.mean([r["kind"] == "確認へ" for r in unk])),
            "wrong_write": len(ww), "wrong_write_known": sum(r["expected"] != "UNKNOWN" for r in ww),
            "wrong_write_disputed": sum(r["utterance"] in CHALLENGE_DISPUTED for r in ww),
            "by_cat": {c: part([r for r in sc if r["category"] == c]) for c in ORDER},
            "H_seen": part([r for r in h if r["utterance"] in CHALLENGE_H_SEEN]),
            "H_unseen": part([r for r in h if r["utterance"] not in CHALLENGE_H_SEEN])}


def judge(ns: list[int]) -> None:
    from .p4_eval import load_rows
    rows = load_rows()
    notes = {r["category"]: r["category_note"] for r in rows}
    res, md = {}, []
    for n in ns:
        old, new = P3 / f"L{n:02d}" / "head", WORK / f"L{n:02d}" / "head"
        if not (P4W / f"device_L{n}.json").exists() or not (old / "head.json").exists() or not (new / "head.json").exists():
            print(f"層 {n}: 評価セットの h（P4）か、前・後のヘッドが無い。飛ばす")
            continue
        b, a = judge_with(n, rows, old), judge_with(n, rows, new)
        sb, sa = summarize(b["rows"]), summarize(a["rows"])
        res[n] = {"before": {**b, "summary": sb}, "after": {**a, "summary": sa}}
        print(f"\n########## 層 {n} 個: 前（714 行のヘッド）→ 後（933 行のヘッド）##########")
        print(f"閾値 前 {b['thresholds']}（温度 {b['temperature']:.2f}）\n     後 {a['thresholds']}（温度 {a['temperature']:.2f}）")
        print(f"{'種類':<24} {'行':>3} {'正解 前→後':>10} {'確認へ 前→後':>11} {'誤実行 前→後':>11}")
        md += [f"\n### 層 {n} 個\n", f"閾値 前 `{b['thresholds']}` → 後 `{a['thresholds']}`\n",
               "|種類|行|正解 前|正解 後|確認・聞き返しへ 前|後|誤実行 前|誤実行 後|", "|---|---|---|---|---|---|---|---|"]
        lines = [(c + " " + notes[c], sb["by_cat"][c], sa["by_cat"][c]) for c in ORDER]
        lines.insert(9, ("　H のうち 学習に同じ機能がある", sb["H_seen"], sa["H_seen"]))
        lines.insert(10, ("　H のうち 学習に同じ機能が無い", sb["H_unseen"], sa["H_unseen"]))
        for name, x, y in lines:
            print(f"{name:<24} {x['n']:>3} {x['ok']:>4} → {y['ok']:<4} {x['clarify']:>4} → {y['clarify']:<4} {x['wrong_write']:>4} → {y['wrong_write']:<4}")
            md.append(f"|{name}|{x['n']}|{x['ok']}|{y['ok']}|{x['clarify']}|{y['clarify']}|{x['wrong_write']}|{y['wrong_write']}|")
        for tag, s in (("前", sb), ("後", sa)):
            line = (f"{tag}: 正解 {s['ok']}/{s['n']}・行き先のある行の正解率 {s['accuracy_known']:.3f}・確認へ回した割合 {s['over_clarify_known']:.3f} / "
                    f"意図が不明・機能にない依頼を確認へ回した割合 {s['reject_recall']:.3f} / 誤実行 {s['wrong_write']} 件"
                    f"（うち行き先のある行 {s['wrong_write_known']}・ラベルが食い違う行 {s['wrong_write_disputed']}）")
            print(line)
            md.append("\n" + line)
        # 行ごとの変わり方
        bb = {r["id"]: r for r in b["rows"]}
        ch = collections.Counter()
        fixed, broke = [], []
        for r in a["rows"]:
            o = bb[r["id"]]
            if r["category"] == "M":
                continue
            if o["kind"] == "誤実行" and r["kind"] != "誤実行":
                fixed.append((o, r))
            if o["kind"] != "誤実行" and r["kind"] == "誤実行":
                broke.append((o, r))
            ch[(bool(o["ok"]), bool(r["ok"]))] += 1
        print(f"行ごと: 両方正解 {ch[(True, True)]}・後だけ正解 {ch[(False, True)]}・前だけ正解 {ch[(True, False)]}・両方不正解 {ch[(False, False)]} / "
              f"誤実行が止まった {len(fixed)} 行・新しく誤実行 {len(broke)} 行")
        md.append(f"\n行ごと: 両方正解 {ch[(True, True)]}・後だけ正解 {ch[(False, True)]}・前だけ正解 {ch[(True, False)]}・両方不正解 {ch[(False, False)]} / "
                  f"誤実行が止まった {len(fixed)} 行・新しく誤実行 {len(broke)} 行")
        for title, items in (("新しく誤実行になった行", broke), ("後も誤実行のままの行", [(bb[r["id"]], r) for r in a["rows"]
                                                                       if r["kind"] == "誤実行" and bb[r["id"]]["kind"] == "誤実行"])):
            if items:
                print(f"{title}:")
                md += [f"\n{title}:\n", "|種類|発話|正解|前|後|後の確信度|後の棄却の得点|", "|---|---|---|---|---|---|---|"]
                for o, r in items:
                    print(f"  [{r['category']}]「{r['utterance']}」正解 {r['expected']} / 前 {o['final']} → 後 {r['final']}（確信度 {r['conf']:.2f}・棄却 {r['reject']:.3f}）")
                    md.append(f"|{r['category']}|{r['utterance']}|{r['expected']}|{o['final']}|{r['final']}|{r['conf']:.2f}|{r['reject']:.3f}|")
        res[n]["changes"] = {"both_ok": ch[(True, True)], "after_only": ch[(False, True)], "before_only": ch[(True, False)],
                             "fixed_wrong_write": len(fixed), "new_wrong_write": len(broke)}

    if not res:
        raise SystemExit("判定できる層が無い")
    # 棄却の閾値を動かしたとき（後のヘッド）。閾値は validation で決めた値が層ごとに違うので、同じ値にそろえて比べる
    sweep = ["\n## 棄却の閾値をそろえたとき（後のヘッド。評価セットで閾値を選んだ値ではなく、感度を見るための表）\n",
             "|層|棄却の閾値|行き先のある行の正解率|同・確認へ回した割合|意図が不明・機能にない依頼を確認へ回した割合|誤実行|うち機能にない依頼（H）|", "|---|---|---|---|---|---|---|"]
    print("\n===== 棄却の閾値をそろえたとき（後のヘッド）=====")
    for n in res:
        own = float(res[n]["after"]["thresholds"]["REJECT"])
        res[n]["sweep_after"] = []
        for tau in (own, 0.9, 0.63, 0.5, 0.3):
            s = summarize(judge_with(n, rows, WORK / f"L{n:02d}" / "head", tau)["rows"])
            res[n]["sweep_after"].append({"reject_tau": tau, "accuracy_known": s["accuracy_known"], "over_clarify_known": s["over_clarify_known"],
                                          "reject_recall": s["reject_recall"], "wrong_write": s["wrong_write"],
                                          "wrong_write_H": s["by_cat"]["H"]["wrong_write"]})
            name = f"{tau:.3f}（validation で決めた値）" if tau == own else f"{tau:.2f}"
            print(f"層 {n:>2} τ={name:<24} 正解率 {s['accuracy_known']:.3f}・確認へ {s['over_clarify_known']:.3f}・棄却 {s['reject_recall']:.3f}・"
                  f"誤実行 {s['wrong_write']}（H {s['by_cat']['H']['wrong_write']}）")
            sweep.append(f"|層 {n}|{name}|{s['accuracy_known']:.3f}|{s['over_clarify_known']:.3f}|{s['reject_recall']:.3f}|{s['wrong_write']}|{s['by_cat']['H']['wrong_write']}|")
    # 層をまたいだまとめ
    keys = [("正解（350 行）", lambda s: f"{s['ok']}"), ("行き先のある行の正解率", lambda s: f"{s['accuracy_known']:.3f}"),
            ("同・確認へ回した割合", lambda s: f"{s['over_clarify_known']:.3f}"),
            ("意図が不明・機能にない依頼を確認へ回した割合", lambda s: f"{s['reject_recall']:.3f}"),
            ("誤実行", lambda s: f"{s['wrong_write']}"), ("　うち行き先のある行", lambda s: f"{s['wrong_write_known']}"),
            ("　うち意図が不明（G）", lambda s: f"{s['by_cat']['G']['wrong_write']}"),
            ("　うち機能にない依頼（H）", lambda s: f"{s['by_cat']['H']['wrong_write']}"),
            ("　　学習に同じ機能がある 15 行", lambda s: f"{s['H_seen']['wrong_write']}"),
            ("　　学習に同じ機能が無い 15 行", lambda s: f"{s['H_unseen']['wrong_write']}"),
            ("機能の語を含む雑談（F・40 行）の正解", lambda s: f"{s['by_cat']['F']['ok']}"),
            ("長い文（B・30 行）の正解", lambda s: f"{s['by_cat']['B']['ok']}"), ("口語（C・30 行）の正解", lambda s: f"{s['by_cat']['C']['ok']}"),
            ("敬語・方言（E・30 行）の正解", lambda s: f"{s['by_cat']['E']['ok']}")]
    top = ["\n## 層をまたいだまとめ（前 = 714 行のヘッド → 後 = 933 行のヘッド）\n",
           "|指標|" + "|".join(f"層 {n}" for n in res) + "|", "|---|" + "---|" * len(res)]
    print("\n===== まとめ（前 → 後）=====")
    for name, f in keys:
        cells = [f"{f(res[n]['before']['summary'])} → {f(res[n]['after']['summary'])}" for n in res]
        print(f"{name:<28} " + "   ".join(f"層 {n}: {c}" for n, c in zip(res, cells)))
        top.append(f"|{name}|" + "|".join(cells) + "|")
    RES.mkdir(parents=True, exist_ok=True)
    for n, r in res.items():
        (RES / f"challenge_L{n}_before_after.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "challenge_before_after.md").write_text("\n".join(top + sweep + md) + "\n", encoding="utf-8")
    print(f"保存しました: {RES}（challenge_before_after.md・challenge_L<n>_before_after.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "heads", "judge"])
    ap.add_argument("--layers", default="17,26,35")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ns = layers_of(a.layers)
    print(f"\n===== P5 {a.step}（足した行: " + "・".join(TYPE_NOTE.values()) + "）=====", flush=True)
    if a.step == "encode":
        encode()
    elif a.step == "device":
        device(ns)
    elif a.step == "heads":
        heads(ns, a.force)
    else:
        judge(ns)


if __name__ == "__main__":
    main()
