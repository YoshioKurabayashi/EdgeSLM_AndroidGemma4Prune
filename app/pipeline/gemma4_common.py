# -*- coding: utf-8 -*-
"""EdgeSLM_AndroidGemma4 の共通の置き場所と定数（2026-09-20）。

対象は Gemma 4 E2B（Google の QAT モバイル版 `google/gemma-4-E2B-it-qat-mobile-transformers`）のテキスト部分。
画像・音声のエンコーダは扱わない。

★EdgeSLM_AndroidGemma4Head での改修（2026-09-27。設計: DoC/設計書_G2_hの出力を持つcontext_binary.md の 4.1 節）:
  書き込み先（`WORK`・`RESULTS`）は本プロジェクト、**読むだけのもの**（QAT の重み `HF_DIR`・状態の取り込み `CAPTURE`・
  端末の表 `TABLES`・量子化済みのかたまり `GEMMA_OPS`）は EdgeSLM_AndroidGemma4 の work を指す。原本は変更しない。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
RESULTS = ROOT / "results"

# 読むだけ（EdgeSLM_AndroidGemma4 の生成物。本プロジェクトからは書き込まない）
GEMMA_ROOT = ROOT.parent / "EdgeSLM_AndroidGemma4"
GEMMA_WORK = GEMMA_ROOT / "work"
CAPTURE = GEMMA_WORK / "capture"          # HF（丸めあり）の状態。較正と配線の確認に使う
TABLES = GEMMA_WORK / "tables"            # 端末と同じ表（埋め込み 2bit・PLE 4bit・位置の表）
GEMMA_OPS = GEMMA_WORK / "ops"            # 量子化済みのかたまり A〜E（g4c*_L256）

HF_REPO = "google/gemma-4-E2B-it-qat-mobile-transformers"
HF_DIR = GEMMA_WORK / "hf" / "gemma-4-E2B-it-qat-mobile-transformers"
# HF の作業用の置き場（キャッシュ・Xet のチャンク）。**C: に書かせない**（C: は残りが少ない）
HF_HOME = GEMMA_WORK / "hf_home"

# 取得するファイル（画像・音声の前処理の設定も小さいので一緒に取る。モデル本体は1ファイル）
HF_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "processor_config.json", "preprocessor_config.json", "README.md",
    "model.safetensors",
]
