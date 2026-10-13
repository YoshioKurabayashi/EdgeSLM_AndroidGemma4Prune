# -*- coding: utf-8 -*-
"""お試し: Gemma 4 E2B の層を減らしても分類が落ちないか（EdgeSLM_AndroidGemma4Prune）の共通の場所と定数。

設計: DoC/設計書_P1_層ごとの分類性能.md の 3 節。

★ほかのプロジェクトは**読むだけ**（書き込まない）:
- EdgeSLM_AndroidGemma4 … QAT の重み（HF 形式）
- EdgeSLM_MultiHead_Multi … ルーターの 714 行のデータ・ヘッドの学習／評価／交差検証のコード（edge_router）
- EdgeSLM_AndroidGemma4Head … 比べる基準（G1: 35 層ぜんぶの h で学習したヘッドの結果・特徴）
書き込むのは本プロジェクトの work/・results/ だけ。
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT = PROJECT_ROOT.parent
MULTI_ROOT = PARENT / "EdgeSLM_MultiHead_Multi"                 # 読むだけ
GEMMA_ROOT = PARENT / "EdgeSLM_AndroidGemma4"                   # 読むだけ
HEAD_PRJ = PARENT / "EdgeSLM_AndroidGemma4Head"                 # 読むだけ（比べる基準）

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
HF_DIR = GEMMA_ROOT / "work" / "hf" / "gemma-4-E2B-it-qat-mobile-transformers"
ROUTER_DATA = MULTI_ROOT / "data" / "synthetic_v2" / "utterances.jsonl"
G1_SRQ = HEAD_PRJ / "work" / "g1" / "srq"                       # 35 層ぜんぶの h（G1。SRQ あり）

N_LAYERS = 35
# 試す層の数（層 0〜n-1 を残す）。15 は「K/V を使い回す層（15〜34）をすべて外す」切り目
DEFAULT_LAYERS = [4, 6, 8, 10, 12, 14, 15, 17, 20, 24, 28, 35]
SEQUENCE_CHECKPOINT = "heads_mlp_seed7.pt"

WORK = PROJECT_ROOT / "work" / "p1"
RESULTS = PROJECT_ROOT / "results" / "P1"


def use_multi_library() -> None:
    src = str(MULTI_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def env_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def layer_dir(n: int) -> Path:
    return WORK / f"L{n:02d}"


def write_provenance(step: str, extra: dict | None = None) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "provenance.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    log.append({"step": step, "at": datetime.datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_ID, "hf_dir": str(HF_DIR), **(extra or {})})
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
