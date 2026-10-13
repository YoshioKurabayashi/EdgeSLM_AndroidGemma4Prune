# -*- coding: utf-8 -*-
"""P4 手順 2: データの外の発話（challenge_v1）を、端末の前半 17 層と 35 層のグラフで分類し、種類ごとに比べる。

設計: DoC/設計書_P4_データの外の発話での評価.md の 4 節。

ヘッドは 714 行で学習済みのもの（P2: 端末の `h` で学習、画面に載せる `heads_mlp_seed7.pt`）をそのまま使う。**学習し直さない**。
判定はアプリの画面と同じ（分類ヘッドだけ。規則・Dispatcher なし）: 棄却の得点 ≥ 閾値 → 確認へ回す（REJECT）／
1 位の確信度 < 閾値 → 聞き返す（CLARIFY）／それ以外 → 1 位の行き先。

|段|何をするか|
|---|---|
|`encode`|発話を F2 の文にして Gemma 4 のチャットテンプレートで包み、トークン列にする（64 トークンを超える行は外して記録）|
|`device`|お試しアプリで、層 17・35 のグラフに全行を流し、`h`（最後・全位置の平均）を集める|
|`judge`|書き出し済みのヘッド（`work/p3/L<n>/head/`。Kotlin と同じ式）で判定し、種類ごとに集計。誤実行した行・2 つのグラフで判定が違う行も出す|

指標（種類ごと）:
- 行き先のある行（A〜E・I・F・V）: 正解（判定＝正解の行き先）／確認・聞き返しへ（安全だが手間）／別の行き先／**誤実行**（正解でない操作系を実行）
- 意図が不明・機能にない依頼（G・H）: 確認・聞き返しへ（正解）／雑談へ／読むだけの機能へ／**誤実行**（操作系を実行）
- 複数の依頼（M）: 参考（どの行き先になったかだけ記録）

    python -m app.p4_eval encode
    python -m app.p4_eval device --layers 17,35
    python -m app.p4_eval judge --layers 17,35
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .common import HF_DIR, PROJECT_ROOT
from .g3_headlib import load_exported, numpy_head, final_of

DATA = PROJECT_ROOT / "data" / "challenge_v1" / "utterances.jsonl"
WORK = PROJECT_ROOT / "work" / "p4"
RES = PROJECT_ROOT / "results" / "P4"
T = 64
ORDER = ["A", "B", "C", "D", "E", "I", "F", "G", "H", "V", "M"]
SAFE = ("REJECT", "CLARIFY")


def load_rows() -> list[dict]:
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def render_f2(r: dict) -> str:
    """EdgeSLM_MultiHead_Multi `edge_router.data.build.render(row, "F2")` と同じ（履歴は無い行だけ）"""
    s = r["state"]
    return f"[状態] 走行:{s['motion']} 権限:{s['permission']}\n[履歴] なし\n[発話] {r['utterance']}"


def encode() -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    rows = load_rows()
    # チャットテンプレートの前後は、画面の資材（p3_assets が HF のテンプレートから作った samples.json）と同じものを使う
    # ＝アプリの画面で入力したときと同じ符号化になる
    sj = PROJECT_ROOT / "work" / "p3" / "L17" / "head" / "samples.json"
    pre, suf = "<bos><|turn>user\n", "<turn|>\n<|turn>model\n"
    if sj.exists():
        o = json.loads(sj.read_text(encoding="utf-8"))
        pre, suf = o.get("chat_prefix", pre), o.get("chat_suffix", suf)
    ids = {}
    for r in rows:
        ids[r["id"]] = tok(pre + render_f2(r) + suf, add_special_tokens=False)["input_ids"]
    over = [(r["id"], len(ids[r["id"]]), r["utterance"]) for r in rows if len(ids[r["id"]]) > T]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "token_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    ln = [len(v) for v in ids.values()]
    print(f"{len(rows)} 行 / トークン 平均 {np.mean(ln):.1f}・最小 {min(ln)}・最大 {max(ln)}（グラフは {T} まで）")
    for i, n, u in over:
        print(f"  ★{T} を超えるので外す: {i}（{n} トークン）「{u}」")
    (WORK / "encode.json").write_text(json.dumps({"rows": len(rows), "token_mean": float(np.mean(ln)), "token_max": max(ln),
                                                  "over": over}, ensure_ascii=False, indent=1), encoding="utf-8")
    return ids


def device(ns: list[int]) -> None:
    from . import p2_device as D
    D.ensure_app()
    ids = json.loads((WORK / "token_ids.json").read_text(encoding="utf-8"))
    keys = [k for k, v in ids.items() if len(v) <= T]
    for n in D.graphs(ns):
        f = WORK / f"device_L{n}.json"
        if f.exists():
            print(f"層 {n}: 済み（{f.name}。やり直すならこのファイルを別の場所へ移す）", flush=True)
            continue
        got = []
        for k, s in enumerate(range(0, len(keys), D.BATCH)):
            part = keys[s:s + D.BATCH]
            res = D.run_job({**D.base_job(f"p4_L{n}_{k:02d}", n), "prompts": [ids[i] for i in part]}, 900)
            rs = res.get("results", [])
            bad = [(i, r.get("error")) for i, r in zip(part, rs) if not r.get("ok")]
            if len(rs) != len(part) or bad:
                raise SystemExit(f"層 {n}・{k}: 通らない行 {bad[:3]}（{len(rs)}/{len(part)}）")
            got += [{"id": i, "last": r["h_last"], "mean": r["h_mean"], "run_ms": r["run_ms"]} for i, r in zip(part, rs)]
            print(f"  層 {n} 個 {k + 1}: {len(part)} 行・NPU 中央値 {np.median([r['run_ms'] for r in rs]):.1f}ms", flush=True)
        f.write_text(json.dumps({"rows": got}), encoding="utf-8")
        print(f"層 {n}: {len(got)} 行 → {f}", flush=True)


def judge_one(n: int, rows: list[dict]) -> dict:
    meta, flat = load_exported(PROJECT_ROOT / "work" / "p3" / f"L{n:02d}" / "head")
    got = {r["id"]: r for r in json.loads((WORK / f"device_L{n}.json").read_text(encoding="utf-8"))["rows"]}
    rows = [r for r in rows if r["id"] in got]
    last = np.array([got[r["id"]]["last"] for r in rows], np.float64)
    mean = np.array([got[r["id"]]["mean"] for r in rows], np.float64)
    p = numpy_head(meta, flat, last, mean)
    cls = meta["action_classes"]
    write = {"TOOL_" + k for k, v in meta["tool_effect"].items() if v == "write"}
    out = []
    for k, r in enumerate(rows):
        fin = final_of(meta, p["action"][k], float(p["reject"][k]))
        top = cls[int(p["action"][k].argmax())]
        exp = r["action"]
        if exp == "MULTI":
            kind = "参考"
        elif exp == "UNKNOWN":
            kind = "確認へ" if fin in SAFE else "雑談へ" if fin == "CHAT" else "誤実行" if fin in write else "読むだけの機能へ"
        else:
            kind = "正解" if fin == exp else "確認へ" if fin in SAFE else "誤実行" if fin in write else "別の行き先"
        out.append({"id": r["id"], "category": r["category"], "utterance": r["utterance"], "expected": exp, "final": fin, "top1": top,
                    "conf": round(float(p["action"][k].max()), 4), "reject": round(float(p["reject"][k]), 4), "kind": kind,
                    "ok": kind == "正解" or (exp == "UNKNOWN" and kind == "確認へ"),
                    "top1_correct": (top == exp) if exp not in ("UNKNOWN", "MULTI") else None})
    return {"n_layers": n, "thresholds": meta["thresholds"], "rows": out}


def table(res: dict) -> dict:
    by = collections.defaultdict(collections.Counter)
    top = collections.defaultdict(list)
    for r in res["rows"]:
        by[r["category"]][r["kind"]] += 1
        by[r["category"]]["n"] += 1
        by[r["category"]]["ok"] += int(r["ok"])
        if r["top1_correct"] is not None:
            top[r["category"]].append(r["top1_correct"])
    return {c: {**dict(by[c]), "top1_acc": float(np.mean(top[c])) if top[c] else None} for c in ORDER if c in by}


def judge(ns: list[int]) -> None:
    rows = load_rows()
    notes = {r["category"]: r["category_note"] for r in rows}
    allres = {}
    for n in ns:
        if not (WORK / f"device_L{n}.json").exists():
            print(f"層 {n}: 端末の h が無い。飛ばす")
            continue
        if not (PROJECT_ROOT / "work" / "p3" / f"L{n:02d}" / "head" / "head.json").exists():
            raise SystemExit(f"層 {n}: 書き出したヘッドが無い。先に python -m app.p3_assets --layers {n}")
        allres[n] = judge_one(n, rows)
    RES.mkdir(parents=True, exist_ok=True)
    md = []
    for n, res in allres.items():
        t = table(res)
        print(f"\n########## 層 {n} 個（閾値 {res['thresholds']}）##########")
        print(f"{'種類':<24} {'行':>3} {'正解':>4} {'確認へ':>5} {'別の行き先':>6} {'雑談へ':>5} {'読むだけへ':>6} {'誤実行':>4} {'1 位の正解率':>8}")
        md += [f"\n### 層 {n} 個\n", "|種類|行|正解|確認・聞き返しへ|別の行き先|雑談へ|読むだけの機能へ|誤実行|1 位の正解率（閾値なし）|", "|---|---|---|---|---|---|---|---|---|"]
        for c, v in t.items():
            unk = c in ("G", "H")
            ok = v.get("ok", 0)
            acc = "—" if v["top1_acc"] is None else f"{v['top1_acc']:.3f}"
            print(f"{c + ' ' + notes[c]:<24} {v['n']:>3} {ok if c != 'M' else '—':>4} {v.get('確認へ', 0):>5} {v.get('別の行き先', 0):>6} "
                  f"{v.get('雑談へ', 0):>5} {v.get('読むだけの機能へ', 0):>6} {v.get('誤実行', 0):>4} {acc:>8}")
            md.append(f"|{c} {notes[c]}|{v['n']}|{'—' if c == 'M' else ok}|{'（正解）' if unk else v.get('確認へ', 0)}|{v.get('別の行き先', 0)}|"
                      f"{v.get('雑談へ', 0)}|{v.get('読むだけの機能へ', 0)}|{v.get('誤実行', 0)}|{acc}|")
        scored = [r for r in res["rows"] if r["category"] != "M"]
        tools = [r for r in scored if r["expected"] not in ("UNKNOWN",)]
        unk = [r for r in scored if r["expected"] == "UNKNOWN"]
        summ = {"n": len(scored), "accuracy_known": float(np.mean([r["kind"] == "正解" for r in tools])),
                "over_clarify_known": float(np.mean([r["kind"] == "確認へ" for r in tools])),
                "reject_recall": float(np.mean([r["kind"] == "確認へ" for r in unk])),
                "wrong_write": sum(r["kind"] == "誤実行" for r in scored)}
        res["summary"], res["table"] = summ, t
        print(f"まとめ（M を除く {summ['n']} 行）: 行き先のある行の正解率 {summ['accuracy_known']:.3f}・確認へ回した割合 {summ['over_clarify_known']:.3f} / "
              f"意図が不明・機能にない依頼を確認へ回した割合 {summ['reject_recall']:.3f} / 誤実行 {summ['wrong_write']} 件")
        md.append(f"\nまとめ（M を除く {summ['n']} 行）: 行き先のある行の正解率 {summ['accuracy_known']:.3f}・確認へ回した割合 {summ['over_clarify_known']:.3f} / "
                  f"意図が不明・機能にない依頼を確認へ回した割合 {summ['reject_recall']:.3f} / 誤実行 {summ['wrong_write']} 件")
        ww = [r for r in res["rows"] if r["kind"] == "誤実行"]
        if ww:
            print("誤実行した行:")
            md += ["\n誤実行した行:\n", "|種類|発話|正解|判定|確信度|棄却の得点|", "|---|---|---|---|---|---|"]
            for r in ww:
                print(f"  [{r['category']}]「{r['utterance']}」正解 {r['expected']} → {r['final']}（確信度 {r['conf']:.2f}・棄却 {r['reject']:.3f}）")
                md.append(f"|{r['category']}|{r['utterance']}|{r['expected']}|{r['final']}|{r['conf']:.2f}|{r['reject']:.3f}|")
        (RES / f"challenge_L{n}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if len(allres) >= 2:
        a, b = list(allres)[:2]
        ra = {r["id"]: r for r in allres[a]["rows"]}
        diff = [(ra[r["id"]], r) for r in allres[b]["rows"] if r["id"] in ra and ra[r["id"]]["final"] != r["final"]]
        print(f"\n層 {a} と層 {b} で判定が違う行: {len(diff)} / {len(ra)}")
        md += [f"\n### 層 {a} と層 {b} で判定が違う行（{len(diff)} / {len(ra)}）\n", f"|種類|発話|正解|層 {a}|層 {b}|", "|---|---|---|---|---|"]
        for x, y in diff:
            md.append(f"|{x['category']}|{x['utterance']}|{x['expected']}|{x['final']}（{x['kind']}）|{y['final']}（{y['kind']}）|")
    (RES / "challenge_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"保存しました: {RES}（challenge_L<n>.json・challenge_table.md）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["encode", "device", "judge"])
    ap.add_argument("--layers", default="17,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    print(f"\n===== P4 {a.step} =====", flush=True)
    if a.step == "encode":
        encode()
    elif a.step == "device":
        device(ns)
    else:
        judge(ns)


if __name__ == "__main__":
    main()
