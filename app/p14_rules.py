# -*- coding: utf-8 -*-
"""P14: 対象の語の食い違いを確かめる規則（S4）。`p13_rules` の上に重ねる。EdgeSLM_MultiHead_Multi の規則は読むだけ。

設計: DoC/設計書_P14_対象の語の食い違いを確かめる.md

P13 の評価セット v3 で、層 17 の重大な誤りの半分は、対象の依頼を別の操作系ツールで実行したものだった
（「風量を2に下げて」→ メディア、「ボリュームを3つ上げて」→ 空調）。ヘッドは取り違えたが、発話には対象の語が書いてある。

S4: 操作系ツール A（空調・メディア・ナビ）を実行しようとしているとき、発話に
    **A の対象の語が 1 つも無く、別の操作系ツール B の対象の語がある** なら、実行せず聞き返す。
    履歴のある発話には触れない（「もう少し下げて」のような続きの依頼は、履歴で対象が決まる）。

辞書は本プロジェクトのもの（EdgeSLM_MultiHead_Multi の `TARGET_TERMS` は使わない）。
元の辞書は「度」「設定」「場所」のような語を含み、「もう一度」「設定温度」で食い違いと誤って判定するため。

    make_rules(rules) -> apply(row, predicted_action)      # `p8_gate.run(make_rules=...)` に渡す
"""
from __future__ import annotations

import re

from .p13_rules import Decision, make_rules as make_p13

TERMS = {
    "control_hvac": ("エアコン", "冷房", "暖房", "空調", "温度", "風", "ヒーター", "クーラー", "デフロスタ", "曇り",
                     "暖か", "暖め", "涼し", "冷や", "暑", "寒"),
    "control_media": ("音楽", "音量", "音", "曲", "ボリューム", "ラジオ", "再生", "ミュート", "消音", "歌"),
    "set_navigation": ("ナビ", "案内", "目的地", "行き先", "ルート", "経由", "向かっ", "寄"),
}
DEGREES = re.compile(r"[0-9０-９]+度")          # 「24度」は空調の語。「もう一度」は数えない


def has_term(utterance: str, tool: str) -> bool:
    if tool == "control_hvac" and DEGREES.search(utterance):
        return True
    return any(t in utterance for t in TERMS[tool])


def mismatch(utterance: str, tool: str) -> str | None:
    """A（tool）の語が無く、別の操作系 B の語があるとき、B の名前を返す"""
    if tool not in TERMS or has_term(utterance, tool):
        return None
    for other in TERMS:
        if other != tool and has_term(utterance, other):
            return other
    return None


def make_rules(rules):
    p13 = make_p13(rules)

    def apply(row: dict, predicted_action: str):
        d = p13(row, predicted_action)
        if d.demote or d.rule is not None or not predicted_action.startswith("TOOL_") or row.get("history"):
            return d
        if mismatch(row.get("utterance", ""), predicted_action[len("TOOL_"):]):
            return Decision(predicted_action, True, "S4_TARGET_MISMATCH")
        return d
    return apply
