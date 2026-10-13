# -*- coding: utf-8 -*-
"""P17: 元の規則（EdgeSLM_MultiHead_Multi）の「指示語だけ」が、指示語でない語を拾う所に例外を置く。`p14_rules` の上に重ねる。

設計: DoC/設計書_P17_不要な聞き返しを減らす.md の 5 節。

元の規則は「これ」「それ」「直して」などを**部分一致**で探す。評価セット v3 で、次の対象の依頼を聞き返していた（どの層でも毎回）。

|発話|拾った語|実際|
|---|---|---|
|これまでに何キロ走ったか教えて|これ|「これまで」は指示語ではない|
|ドアの傷を直してくれる板金屋を調べて|直して|「直してくれる（店）」は指示ではない|
|これ歌ってるのは誰／これは何という曲|これ|いま流れている曲を指す。曲の情報なら履歴なしで決まる|

例外は 2 つ。元の規則が「指示語だけ（ANAPHORA_WITHOUT_HISTORY）」で止めたときだけ試す。
1. 指示語でない言い回し（これまで・これから・それぞれ・それほど・それから・直してくれる・直せる）を除いてから、もう一度規則にかける
2. 行き先が曲の情報で、曲・歌・誰などの語があるとき、「これ」は「この曲」と読む（元の辞書でも「この曲」は対象の語）
元のファイルは変えない。例外で通した後も、ほかの規則（S1〜S4）は通る。
"""
from __future__ import annotations

from .p14_rules import make_rules as make_p14

SONG_WORDS = ("曲", "歌", "誰", "何て", "タイトル", "アーティスト", "バンド")      # 「これ」が、いま流れている曲を指すと読める語
NOT_DEICTIC = ("これまで", "これから", "それぞれ", "それほど", "それから", "それなら", "直してくれる", "直してもらえる", "直せる")


def make_rules(rules):
    p14 = make_p14(rules)

    def apply(row: dict, predicted_action: str):
        d = p14(row, predicted_action)
        if d.rule != "ANAPHORA_WITHOUT_HISTORY":
            return d
        u = row.get("utterance", "")
        v = u
        for w in NOT_DEICTIC:
            v = v.replace(w, "")
        if predicted_action == "TOOL_get_media_info" and any(w in u for w in SONG_WORDS):
            v = v.replace("これ", "この曲")
        if v == u:
            return d
        return p14({**row, "utterance": v}, predicted_action)
    return apply
