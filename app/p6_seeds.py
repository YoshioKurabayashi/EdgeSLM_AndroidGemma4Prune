# -*- coding: utf-8 -*-
"""P6: 学習済みのヘッドを seed・種類（mlp・linear）ごとに評価セット（challenge_v1）へ当て直し、P5 の差がたまたまかを見る。numpy だけで動く。

設計: DoC/設計書_P6_ヘッドのseedを変えた判定.md。

P5 の判定は、画面に載せるヘッド 1 つ（mlp・seed 7）だけで出した。層 17 の「機能にない依頼を止める力が弱い」という差が、
seed を変えても残るかを見る。**新しく学習しない・端末も使わない**。使うのは作成済みのもの（読むだけ）:

|使うもの|場所|
|---|---|
|前のヘッド（714 行）6 つ × 3 層|`work/p2/L<n>/device/heads/heads_{mlp,linear}_seed{7,17,29}.pt`|
|後のヘッド（933 行）6 つ × 3 層|`work/p5/L<n>/heads/heads_{mlp,linear}_seed{7,17,29}.pt`|
|温度・閾値（validation で決めた値）|同じフォルダの `evaluate_report.json`|
|評価セットの `h`|`work/p4/device_L<n>.json`（P4 で端末から集めたもの）|

`.pt` は torch の保存形式（zip の中に pickle と重みの生データ）。torch を入れずに読むため、重みだけを numpy に取り出す小さな読み手を持つ（`load_pt`）。
計算式は `g3_headlib.numpy_head`（画面と同じ式）に linear を足したもの。**確かめ**: mlp・seed 7 の結果が P5 の判定（書き出したヘッド）と一致すること。

出す指標（評価セット・M を除く 350 行）:
- validation で決めた閾値のとき: 行き先のある行の正解率・確認へ回した割合・誤実行（全体・機能にない依頼 H・うち学習に同じ機能が無い 15 行・意図が不明 G）
- 棄却の閾値を 0.63 にそろえたとき: 誤実行（全体・H）。seed ごとに閾値が大きく違う（0.30〜0.99）ので、そろえた値も並べる
- 閾値に依らない指標: 棄却の得点の AUC（行き先のある行 ↔ 機能にない依頼 H／意図が不明 G）。1.0 に近いほど、棄却の得点だけで見分けられる

    python -m app.p6_seeds        # results/P6/seeds.json・seeds_table.md
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import pickle
import zipfile

import numpy as np

from .common import MULTI_ROOT, PROJECT_ROOT
from .g3_headlib import erf_as
from .p5_retrain import P2, P4W, WORK as P5, challenge_h, decide, summarize

RES = PROJECT_ROOT / "results" / "P6"
KINDS, SEEDS = ("mlp", "linear"), (7, 17, 29)
SOURCES = {"前（714 行）": lambda n: P2 / f"L{n:02d}" / "device" / "heads", "後（933 行）": lambda n: P5 / f"L{n:02d}" / "heads"}
ALIGNED_TAU = 0.63
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32,
           "HalfStorage": np.float16, "BoolStorage": np.bool_}


# ---------------------------------------------------------------- .pt を torch なしで読む

class _Storage:
    def __init__(self, name: str):
        self.name = name


def load_pt(path) -> dict:
    """`MultiHead.save` の .pt（{"state", "config", "metadata"}）を読み、重みを numpy にする。torch のクラスは重みの入れ物だけを受け付ける"""
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    root = pkl[:-len("data.pkl")]

    def rebuild(storage, offset, size, stride, *_):
        n = int(np.prod(size)) if len(size) else 1
        return storage[offset:offset + n].reshape(tuple(size)).copy()

    class Reader(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name in _DTYPES:
                return _Storage(name)
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"{path}: 想定していないクラス {mod}.{name}（重み・設定以外は読まない）")

        def persistent_load(self, pid):
            _, st, key, _loc, _numel = pid
            return np.frombuffer(z.read(f"{root}data/{key}"), dtype=_DTYPES[st.name])

    return Reader(io.BytesIO(z.read(pkl))).load()


def head_probs(blob: dict, temperature: float, last: np.ndarray, mean: np.ndarray) -> dict:
    """`g3_headlib.numpy_head` と同じ式（mlp）＋ linear。action は温度付き softmax、reject は concat(最後, 平均) の 2 クラスの 1 番"""
    st = {k: np.asarray(v, np.float64) for k, v in blob["state"].items()}

    def head(z, p):
        if blob["config"]["kind"] == "linear":
            return z @ st[f"{p}.weight"].T + st[f"{p}.bias"]
        h = z @ st[f"{p}.0.weight"].T + st[f"{p}.0.bias"]
        h = 0.5 * h * (1.0 + erf_as(h / math.sqrt(2.0)))
        return h @ st[f"{p}.2.weight"].T + st[f"{p}.2.bias"]

    def softmax(v, t):
        v = v / t
        e = np.exp(v - v.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    z = (last - st["standardizer.mean"]) / st["standardizer.std"]
    zr = (np.hstack([last, mean]) - st["reject_standardizer.mean"]) / st["reject_standardizer.std"]
    return {"action": softmax(head(z, "heads.action"), temperature), "reject": softmax(head(zr, "heads.reject"), 1.0)[:, 1]}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """pos の得点が neg より高い確率（同点は半分）"""
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


# ---------------------------------------------------------------- 判定

def one(blob: dict, rep: dict, tools: dict, rows: list[dict], last: np.ndarray, mean: np.ndarray) -> dict:
    meta = {"action_classes": blob["metadata"]["action_classes"], "thresholds": {k: float(v) for k, v in rep["thresholds"].items()},
            "tool_effect": tools}
    p = head_probs(blob, float(rep["temperature"]), last, mean)
    s = summarize(decide(rows, meta, p))
    al = summarize(decide(rows, {**meta, "thresholds": {**meta["thresholds"], "REJECT": ALIGNED_TAU}}, p))
    cat = np.array([r["category"] for r in rows])
    known = np.array([r["action"] not in ("UNKNOWN", "MULTI") for r in rows])
    return {"temperature": float(rep["temperature"]), "thresholds": meta["thresholds"],
            "ok": s["ok"], "accuracy_known": s["accuracy_known"], "over_clarify_known": s["over_clarify_known"],
            "reject_recall": s["reject_recall"], "wrong_write": s["wrong_write"], "wrong_write_known": s["wrong_write_known"],
            "wrong_write_H": s["by_cat"]["H"]["wrong_write"], "wrong_write_H_unseen": s["H_unseen"]["wrong_write"],
            "wrong_write_G": s["by_cat"]["G"]["wrong_write"],
            "aligned_wrong_write": al["wrong_write"], "aligned_wrong_write_H": al["by_cat"]["H"]["wrong_write"],
            "aligned_over_clarify_known": al["over_clarify_known"], "aligned_accuracy_known": al["accuracy_known"],
            "auc_H": auc(p["reject"][cat == "H"], p["reject"][known]), "auc_G": auc(p["reject"][cat == "G"], p["reject"][known])}


METRICS = [("行き先のある行の正解率", "accuracy_known", "{:.3f}"), ("同・確認へ回した割合", "over_clarify_known", "{:.3f}"),
           ("誤実行", "wrong_write", "{:.0f}"), ("　うち機能にない依頼（H）", "wrong_write_H", "{:.0f}"),
           ("　　うち学習に同じ機能が無い 15 行", "wrong_write_H_unseen", "{:.0f}"), ("　うち意図が不明（G）", "wrong_write_G", "{:.0f}"),
           (f"棄却の閾値を {ALIGNED_TAU} にそろえたときの誤実行", "aligned_wrong_write", "{:.0f}"),
           ("　うち機能にない依頼（H）", "aligned_wrong_write_H", "{:.0f}"),
           ("　そのときの確認へ回した割合", "aligned_over_clarify_known", "{:.3f}"),
           ("棄却の得点の AUC（機能にない依頼 ↔ 行き先のある行）", "auc_H", "{:.3f}"),
           ("棄却の得点の AUC（意図が不明 ↔ 行き先のある行）", "auc_G", "{:.3f}")]


def main() -> None:
    from .p4_eval import load_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26,35")
    a = ap.parse_args()
    ns = [int(x) for x in a.layers.replace(" ", ",").split(",") if x]
    tools = {t["name"]: t["effect"] for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    allrows = load_rows()
    res = {}
    print("\n===== P6: ヘッドの seed・種類を変えて、評価セットを判定し直す =====", flush=True)
    for n in ns:
        if not (P4W / f"device_L{n}.json").exists():
            print(f"層 {n}: 評価セットの h（P4）が無い。飛ばす")
            continue
        rows, last, mean = challenge_h(n, allrows)
        for src, where in SOURCES.items():
            d = where(n)
            ev = d / "evaluate_report.json"
            if not ev.exists():
                print(f"層 {n}・{src}: {ev} が無い。飛ばす")
                continue
            reps = {r["checkpoint"]: r for r in json.loads(ev.read_text(encoding="utf-8"))["results"]}
            for kind in KINDS:
                for seed in SEEDS:
                    name = f"heads_{kind}_seed{seed}.pt"
                    if name not in reps or not (d / name).exists():
                        continue
                    r = one(load_pt(d / name), reps[name], tools, rows, last, mean)
                    res[(n, src, kind, seed)] = r
                    print(f"層 {n:>2}・{src}・{kind:<6} seed {seed:>2}: 正解率 {r['accuracy_known']:.3f}・確認へ {r['over_clarify_known']:.3f}・"
                          f"誤実行 {r['wrong_write']:>2}（H {r['wrong_write_H']:>2}・G {r['wrong_write_G']:>2}）/ 棄却の閾値 {r['thresholds']['REJECT']:.3f} → "
                          f"{ALIGNED_TAU} にそろえると 誤実行 {r['aligned_wrong_write']:>2}（H {r['aligned_wrong_write_H']:>2}）/ "
                          f"AUC H {r['auc_H']:.3f}・G {r['auc_G']:.3f}", flush=True)
    if not res:
        raise SystemExit("判定できるヘッドが無い")

    # 確かめ: mlp・seed 7 は、P5 の判定（書き出したヘッド）と同じ数字になる
    for n in ns:
        f = PROJECT_ROOT / "results" / "P5" / f"challenge_L{n}_before_after.json"
        if not f.exists():
            continue
        p5 = json.loads(f.read_text(encoding="utf-8"))
        for src, tag in (("前（714 行）", "before"), ("後（933 行）", "after")):
            r = res.get((n, src, "mlp", 7))
            if r is None:
                continue
            want = p5[tag]["summary"]
            if (r["ok"], r["wrong_write"]) != (want["ok"], want["wrong_write"]):
                raise SystemExit(f"層 {n}・{src}・mlp seed 7 が P5 の判定と合わない: {r['ok']}/{r['wrong_write']} ↔ {want['ok']}/{want['wrong_write']}")
    print("確かめ: mlp・seed 7 の正解・誤実行は、P5 の判定（書き出したヘッド）と一致")

    # seed をまたいだまとめ（平均と、最小〜最大）
    md, agg = [], {}
    for kind in KINDS:
        cols = [(n, src) for n in ns for src in SOURCES if any((n, src, kind, s) in res for s in SEEDS)]
        if not cols:
            continue
        md += [f"\n## {kind}（seed {'・'.join(map(str, SEEDS))} の平均。かっこは最小〜最大）\n",
               "|指標|" + "|".join(f"層 {n} {src}" for n, src in cols) + "|", "|---|" + "---|" * len(cols)]
        print(f"\n===== {kind}: seed {SEEDS} の平均（最小〜最大）=====")
        for label, key, fmt in METRICS:
            cells = []
            for n, src in cols:
                v = [res[(n, src, kind, s)][key] for s in SEEDS if (n, src, kind, s) in res]
                agg[f"{n}|{src}|{kind}|{key}"] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "values": v}
                m = f"{np.mean(v):.1f}" if fmt == "{:.0f}" else fmt.format(np.mean(v))
                cells.append(f"{m}（{fmt.format(min(v))}〜{fmt.format(max(v))}）")
            md.append(f"|{label}|" + "|".join(cells) + "|")
            print(f"{label:<34} " + "  ".join(f"層{n}{src[0]}: {c}" for (n, src), c in zip(cols, cells)))
    md += ["\n## ヘッドごと\n", "|層|学習データ|種類|seed|温度|棄却の閾値|正解率|確認へ|誤実行|H|H 同じ機能なし|G|閾値をそろえた誤実行|そのときの H|AUC H|AUC G|",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (n, src, kind, seed), r in res.items():
        md.append(f"|層 {n}|{src}|{kind}|{seed}|{r['temperature']:.2f}|{r['thresholds']['REJECT']:.3f}|{r['accuracy_known']:.3f}|"
                  f"{r['over_clarify_known']:.3f}|{r['wrong_write']}|{r['wrong_write_H']}|{r['wrong_write_H_unseen']}|{r['wrong_write_G']}|"
                  f"{r['aligned_wrong_write']}|{r['aligned_wrong_write_H']}|{r['auc_H']:.3f}|{r['auc_G']:.3f}|")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "seeds.json").write_text(json.dumps({"aligned_tau": ALIGNED_TAU, "aggregate": agg,
                                               "heads": [{"n_layers": n, "data": src, "kind": k, "seed": s, **r} for (n, src, k, s), r in res.items()]},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "seeds_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"保存しました: {RES}（seeds.json・seeds_table.md）")


if __name__ == "__main__":
    main()
