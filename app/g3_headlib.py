# -*- coding: utf-8 -*-
"""G3: 分類ヘッドを端末（Kotlin の HeadModel）と同じ式で計算する道具。numpy だけで動く（torch は要らない）。

設計: DoC/設計書_G3_端末での一括比較とデモ.md の 4.1 節。

前段 EdgeSLM_AndroidHead の `app/p3_assets.py`（`numpy_head`・`final_of`・`erf_as`・`LABELS_JA`・`export_head`）と
同じ式を、本プロジェクトの道具として書き直したもの（前段のファイルは読むだけ。式は `HeadModel.kt` と同じ）。

- `numpy_head`: 標準化 → Linear → GELU（erf は Kotlin と同じ近似）→ Linear。action は温度付き softmax、
  reject は concat(最後, 平均) を別の標準化に通して 2 クラスの softmax の 1 番
- `final_of`: reject ≥ τ_REJECT → REJECT、確信度 < τ（CHAT / 書き込み / 読み取り）→ CLARIFY、それ以外は 1 位
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LABELS_JA = {
    "CHAT": "雑談", "REJECT": "確認へ回す", "CLARIFY": "聞き返す", "UNKNOWN": "意図が不明",
    "TOOL_control_hvac": "エアコン操作", "TOOL_control_media": "音楽・メディア操作",
    "TOOL_get_vehicle_status": "車両の状態を調べる", "TOOL_search_vehicle_manual": "取扱説明書を調べる",
    "TOOL_set_navigation": "ナビの目的地を設定", "TOOL_capture_camera": "カメラで撮影・表示",
    "TOOL_search_poi": "周辺の施設を探す", "TOOL_get_media_info": "再生中の曲を調べる",
    "ABSOLUTE": "値を指定", "INCREASE": "上げる", "DECREASE": "下げる", "MAX_MIN": "最大・最小",
    "TOGGLE": "オン・オフ", "UNSPECIFIED": "指定なし",
}

# 平均から除くトークン（G2 の診断: 正規化の前の値が小さく、端末の量子化の誤差が正規化で拡大される位置）
GEMMA_BOS, GEMMA_TURN_OPEN = 2, 105
MEAN_SKIP_IDS = [GEMMA_BOS, GEMMA_TURN_OPEN]


def export_head(head, hmeta: dict, report: dict, tools: dict, source: dict) -> tuple[dict, np.ndarray]:
    """MultiHead の重みを 1 本の float32 列（head.bin）と形・位置の表（head.json）にする（p3_assets.export_head と同じ）。"""
    state = {k: v.detach().cpu().float().numpy() for k, v in head.state_dict().items() if k != "temperature"}
    tensors, chunks, off = {}, [], 0
    for name in sorted(state):
        a = np.ascontiguousarray(state[name], dtype="<f4")
        tensors[name] = {"offset": off, "shape": list(a.shape)}
        chunks.append(a.ravel())
        off += a.size
    flat = np.concatenate(chunks).astype("<f4")
    meta = {
        "hidden": int(head.hidden_size), "reject_hidden": int(head.reject_hidden_size), "width": int(head.width),
        "kind": head.kind, "temperature": float(report["temperature"]),
        "action_classes": list(hmeta["action_classes"]), "argmode_classes": list(hmeta["argmode_classes"]),
        "thresholds": {k: float(v) for k, v in report["thresholds"].items()},
        "tool_effect": {name: t.get("effect", "read") for name, t in tools.items()},
        "argmode_tools": ["control_hvac", "control_media"],
        "labels_ja": LABELS_JA, "tensors": tensors, "total_floats": int(flat.size), "source": source,
    }
    return meta, flat


def load_exported(head_dir: Path) -> tuple[dict, np.ndarray]:
    meta = json.loads((head_dir / "head.json").read_text(encoding="utf-8"))
    flat = np.fromfile(head_dir / "head.bin", dtype="<f4")
    return meta, flat


def erf_as(x: np.ndarray) -> np.ndarray:
    """Kotlin の HeadModel と同じ近似（Abramowitz–Stegun 7.1.26）"""
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.where(x >= 0, y, -y)


def numpy_head(meta: dict, flat: np.ndarray, last: np.ndarray, mean: np.ndarray) -> dict:
    """HeadModel.kt と同じ計算（float64 で積和）。last・mean は [行, hidden]"""
    last, mean = np.asarray(last, np.float64), np.asarray(mean, np.float64)

    def T(n):
        t = meta["tensors"][n]
        return flat[t["offset"]:t["offset"] + int(np.prod(t["shape"]))].reshape(t["shape"]).astype(np.float64)

    def mlp(z, p):
        h = z @ T(f"{p}.0.weight").T + T(f"{p}.0.bias")
        h = 0.5 * h * (1.0 + erf_as(h / math.sqrt(2.0)))
        return h @ T(f"{p}.2.weight").T + T(f"{p}.2.bias")

    def softmax(v, temp):
        v = v / temp
        e = np.exp(v - v.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    z = (last - T("standardizer.mean")) / T("standardizer.std")
    xr = np.hstack([last, mean])
    zr = (xr - T("reject_standardizer.mean")) / T("reject_standardizer.std")
    return {"action": softmax(mlp(z, "heads.action"), meta["temperature"]),
            "reject": softmax(mlp(zr, "heads.reject"), 1.0)[:, 1]}


def final_of(meta: dict, p: np.ndarray, rej: float) -> str:
    """HeadModel.kt と同じ最終判定"""
    i = int(p.argmax())
    a = meta["action_classes"][i]
    th = meta["thresholds"]
    tau = th["CHAT"] if a == "CHAT" else th["TOOL_write" if meta["tool_effect"].get(a[5:], "read") == "write" else "TOOL_read"]
    return "REJECT" if rej >= th["REJECT"] else ("CLARIFY" if p[i] < tau else a)


def decisions(meta: dict, flat: np.ndarray, last: np.ndarray, mean: np.ndarray) -> dict:
    """行ごとの 1 位・確信度・棄却・最終判定"""
    p = numpy_head(meta, flat, last, mean)
    cls = meta["action_classes"]
    top = p["action"].argmax(1)
    return {"top1": [cls[int(k)] for k in top], "conf": p["action"].max(1).tolist(),
            "reject": p["reject"].tolist(),
            "final": [final_of(meta, p["action"][k], float(p["reject"][k])) for k in range(len(top))]}


def expected_final(row: dict) -> str:
    return "REJECT" if row["action"] == "UNKNOWN" else row["action"]
