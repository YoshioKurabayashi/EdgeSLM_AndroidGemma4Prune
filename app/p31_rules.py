# -*- coding: utf-8 -*-
"""P31: 規則の仕上げ。`p21_rules` の上に重ねる（P21 の規則は変えない）。

設計: DoC/設計書_P31_仕上げ.md の 4 節。

|直し|中身|v7・v8 の例（対象の行・正しい機能のとき）|
|---|---|---|
|R1 「〜に戻って」はナビの語|S5（判定した機能の語が文に無い）でナビを止めたとき、「場所 ＋ に・へ ＋ 戻って・戻る」があれば止めない|鍵を取りに自宅へ戻って／カメラを置き忘れたからホテルに戻って／カメラを取りに事務所へ戻って|
"""
from __future__ import annotations

import re

from .p21_rules import make_rules as make_p21

BACK_TO = re.compile(r"[一-龥ァ-ヴーA-Za-z]+(?:に|へ)戻(?:って|る|り|ろ)")


def make_rules(rules):
    p21 = make_p21(rules)

    def apply(row: dict, predicted_action: str):
        d = p21(row, predicted_action)
        if d.demote and d.rule == "S5_NO_TARGET_TERM" and predicted_action == "TOOL_set_navigation" and BACK_TO.search(row.get("utterance", "")):
            return rules.Decision(predicted_action, False, None)
        return d
    return apply
