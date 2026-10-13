# -*- coding: utf-8 -*-
"""P19: 規則 S5（対象の語が 1 つも無い操作は、実行せず聞き返す）。`p17_rules` の上に重ねる。EdgeSLM_MultiHead_Multi の規則は読むだけ。

設計: DoC/設計書_P19_構成を確定する.md の 5 節。

P18（評価セット v4）で残った重大な誤りの半分以上は、対象の語の無い曖昧な言い方を、操作として実行したものだった
（「ほどほどにして」→ メディア、「いい具合にして」→ 空調、「首がこってるんだ」→ 空調、「ガラスを暗くして」→ 空調）。
S4（P14）は「別の操作系ツールの語がある」ときだけ止めるので、どのツールの語も無い発話は通ってしまう。

S5: 操作系ツール A（空調・メディア・ナビ）を実行しようとしているとき、発話に **A の語が 1 つも無ければ**、実行せず聞き返す。
    履歴のある発話には触れない（「もう少し下げて」のような続きの依頼は、履歴で対象が決まる）。
    読むだけの機能と雑談には使わない（誤っても重大な誤りにならない）。

辞書（`S5_TERMS`）は、P14 の辞書に、次の 2 つを足したもの。
- `S5_TRAIN`: 学習データ（1,584 行）の操作系の行に出てくる語
- `S5_EVAL`: 使い終えた評価セット v1・v3 の操作系の行にだけ出てくる語（AC・BGM・ひんやり・有料 など 8 語。**評価セットを見て足した**）
足さなかった語: 「止めて」「かけて」「鳴らして」「ロック」「オート」（機能にない依頼にも普通に出る: ワイパーを止めて・電話をかけて・ドアをロックして）。
このため、対象を言わない「止めて」「もっと大きく」「オートにして」は、履歴が無ければ聞き返す。
`has_s5_term(..., train_only=True)` は `S5_TRAIN` だけで判定する（新しい言い方での巻き添えを見積もるため。流れでは使わない）。

    make_rules(rules) -> apply(row, predicted_action)      # `p8_gate.run(make_rules=...)` に渡す
"""
from __future__ import annotations

import re

from .p14_rules import DEGREES, TERMS
from .p13_rules import Decision
from .p17_rules import make_rules as make_p17

S5_TRAIN = {
    "control_hvac": ("冷え", "冷た", "換気", "汗", "じめじめ", "蒸", "息苦し", "足元", "曇", "ガラスが白", "乾かし", "吹き出し", "送風"),
    "control_media": ("流し", "聴", "聞こえ", "聞き取", "うるさ", "スキップ", "飛ばし", "一時停止", "ポーズ", "プレイ", "トラック", "シャッフル", "リピート", "ランダム",
                      "Bluetooth", "FM", "ジャズ", "クラシック"),
    "set_navigation": ("まで", "行き", "行っ", "帰", "連れて", "向か", "高速", "下道", "道", "駅", "空港", "場所", "途中", "デスティネーション"),
}
S5_EVAL = {
    "control_hvac": ("ひんやり", "AC"),
    "control_media": ("BGM", "ミュージック", "ソング", "ポップス", "局"),
    "set_navigation": ("有料",),
}
S5_TERMS = {tool: tuple(TERMS[tool]) + S5_TRAIN[tool] + S5_EVAL[tool] for tool in TERMS}
S5_TERMS_TRAIN = {tool: tuple(TERMS[tool]) + S5_TRAIN[tool] for tool in TERMS}
AIR = re.compile(r"空気(?!圧)")          # 「空気がこもってる」は空調。「空気圧」は数えない


def has_s5_term(utterance: str, tool: str, train_only: bool = False) -> bool:
    if tool == "control_hvac" and (DEGREES.search(utterance) or AIR.search(utterance)):
        return True
    return any(t in utterance for t in (S5_TERMS_TRAIN if train_only else S5_TERMS)[tool])


def make_rules(rules):
    p17 = make_p17(rules)

    def apply(row: dict, predicted_action: str):
        d = p17(row, predicted_action)
        if d.demote or d.rule is not None or not predicted_action.startswith("TOOL_") or row.get("history"):
            return d
        tool = predicted_action[len("TOOL_"):]
        if tool in S5_TERMS and not has_s5_term(row.get("utterance", ""), tool):
            return Decision(predicted_action, True, "S5_NO_TARGET_TERM")
        return d
    return apply
