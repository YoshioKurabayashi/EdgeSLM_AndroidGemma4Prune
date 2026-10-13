# -*- coding: utf-8 -*-
"""P21: 規則の部分一致の誤りを直し、型から外れた言い直しを拾う。`p19_rules` の上に重ねる。EdgeSLM_MultiHead_Multi の規則は読むだけ。

設計: DoC/設計書_P21_規則と引数の誤りを直す.md の 3 節。

P20（評価セット v5）で、不要な聞き返しの約半分は決定的な処理の誤りだった。規則の分は次の 2 つ。
どちらも、語を**部分一致**で探しているのが原因（元のファイル・`p13_rules` は変えず、ここで直す）。

|直すもの|誤っていた例|直し方|
|---|---|---|
|A 「指示語だけ」が、指示語でない語を拾う|右**前の**タイヤ・**前の**ガラス（位置の「前」）／入れ**直して**（動詞の後ろの「直す」）|位置の「前の」と、動詞に付いた「直し」を、かなに書き換えてから規則にかけ直す|
|A' 同上。対象の語があるのに、元の辞書に無い|**空調**を入れ直して・車内をもう少し**暖めて**|度合いや動作の語（もう少し・直して・前の など）で止まったとき、操作系の対象そのものを指す語（`TARGET_NOUNS`。P14 の辞書から、動作の語「寄」「向かっ」を除いたもの）があれば通す。「あれ」「それ」「さっき」があるときは通さない|
|B 取り消し（S2）が、取り消しでない語を拾う|消し**忘れて**ないか見て／高速**なしで**行って|動詞に付いた「忘れて」・「〜てない」の形と、名詞に付いた「なしで」を書き換えてから規則にかけ直す|

あわせて、P18・P20 で残った重大な誤りのうち、規則の型から外れていたものを拾う。

|足すもの|例|中身|
|---|---|---|
|C 言い直し（S1）に「待って」を足す|流して、あ、ちょっと待ってね|読点のあとの「待って」「待った」。その前に中身があるとき|
|D 言いよどみ（S3）に「〜なんだけど」を足す|いまの曲なんだけど、えっと／カメラなんだけど|「名詞＋なんだけど」で終わり、その前に「を・に・で・へ」が無い（何をするかを言っていない）|
|E S5 の「空気」の扱い|（空調と判定）前のタイヤに空気を足して|空調の語が「空気」だけで「タイヤ」があるときは、空調の語とみなさない|

    make_rules(rules) -> apply(row, predicted_action)      # `p8_gate.run(make_rules=...)` に渡す
"""
from __future__ import annotations

import re

from .p13_rules import FILLER, LEAD, Decision
from .p14_rules import DEGREES, TERMS
from .p19_rules import AIR, S5_TERMS, make_rules as make_p19

# ---- A: 指示語でない「前の」「直し」
FRONT = re.compile(r"((?:右|左|斜め|目の|手|すぐ|真ん)?)前の(?=ガラス|カメラ|席|座席|シート|映像|タイヤ|バンパー|ほう|方|車|様子|窓|ライト|ワイパー|ドア|角|景色|信号|道)")
FRONT_SIDE = re.compile(r"(右|左|斜め|目の|手|すぐ)前の")
REDO_VERB = re.compile(r"(入れ|かけ|つけ|設定し|案内し|引き|探し|調べ|映し|流し|セットし|計算し|付け|掛け)直し")
# A': 対象そのものを指す語。「寄」「向かっ」「まで」「行っ」のような動作の語は入れない（「もう少し左に寄って」は運転操作で、ナビではない）
TARGET_NOUNS = {"control_hvac": TERMS["control_hvac"], "control_media": TERMS["control_media"],
                "set_navigation": ("ナビ", "案内", "目的地", "行き先", "ルート", "経由", "道")}
PRONOUNS = ("あれ", "それ", "これ", "あっち", "そっち", "こっち", "さっき", "例の", "いつもの", "元に", "この前")      # これがあれば、元の規則の判断のまま
# ---- B: 取り消しでない「忘れて」「なしで」
FORGET_VERB = re.compile(r"(?<=[ぁ-ん一-龥ァ-ヶ])(?<![はをもがの])忘れて")            # 消し忘れて・閉め忘れて・置き忘れて
FORGET_STATE = re.compile(r"忘れて(?=ない|なかっ|た|る|き|しま|いな)")                 # 忘れてない・忘れてた・忘れてきた
WITHOUT = re.compile(r"(?<=[一-龥ァ-ヶー])(?:無し|なし)(?=で|に)")                     # 高速なしで・有料道路なしで（名詞に直に付く）
# ---- C・D
WAIT = re.compile(rf"[、。 　](?:あ、|あ |いや、?|やっぱ、?)?(?:ちょっと|少し)?待っ(?:て|た)")
NOUN_BUT = re.compile(rf"^(?P<head>[^、。]{{1,14}}?)(?:のこと|の件)?なんだけど(?:[、 　]?{FILLER})?[ー…。、]*$")
CASE = ("を", "に", "で", "へ")


def soften_anaphora(u: str) -> str:
    """位置の「前の」と、動詞に付いた「直し」を、かなに書き換える（元の規則の印に当たらなくする。ほかの語は変えない）"""
    v = FRONT.sub(lambda m: m.group(1) + "まえの", u)
    v = FRONT_SIDE.sub(lambda m: m.group(1) + "まえの", v)
    return REDO_VERB.sub(lambda m: m.group(1) + "なおし", v)


def soften_undo(u: str) -> str:
    """動詞に付いた「忘れて」・「忘れてない」の形・名詞に付いた「なしで」を書き換える"""
    v = FORGET_VERB.sub("わすれて", u)
    v = FORGET_STATE.sub("わすれて", v)
    return WITHOUT.sub("ぬき", v)


def has_noun(u: str, tool: str) -> bool:
    if tool == "control_hvac" and DEGREES.search(u):
        return True
    return any(t in u for t in TARGET_NOUNS[tool])


def tire_air_only(u: str) -> bool:
    """E: 空調と判定されたが、空調の語が「空気」だけで、「タイヤ」がある（タイヤの空気は空調ではない）"""
    return "タイヤ" in u and bool(AIR.search(u)) and not DEGREES.search(u) and not any(t in u for t in S5_TERMS["control_hvac"])


def wait_with_content(u: str) -> bool:
    """C: 「〜して、あ、ちょっと待って」。印の前に、言いよどみ以外の中身があるか"""
    m = WAIT.search(u)
    return bool(m) and len(LEAD.sub("", u[:m.start()]).strip("、。 　")) >= 1


def noun_but(u: str) -> bool:
    """D: 「名詞＋なんだけど」で終わり、何をするかを言っていない"""
    m = NOUN_BUT.match(u)
    return bool(m) and not any(c in m.group("head") for c in CASE)


def make_rules(rules):
    p19 = make_p19(rules)

    def apply(row: dict, predicted_action: str):
        d = p19(row, predicted_action)
        u = row.get("utterance", "")
        if row.get("history") or not predicted_action.startswith("TOOL_"):
            return d
        tool = predicted_action[len("TOOL_"):]
        if d.rule == "ANAPHORA_WITHOUT_HISTORY":
            v = soften_anaphora(u)
            if v != u:
                return apply({**row, "utterance": v}, predicted_action)                 # A: 書き換えた文で、最初からかけ直す
            if tool in TARGET_NOUNS and has_noun(u, tool) and not any(p in u for p in PRONOUNS) and not rules.is_undo(u):
                d = Decision(predicted_action, False, None)                             # A': 対象の語がある。指示語だけではない
        if d.rule == "S2_UNDO":
            v = soften_undo(u)
            if v != u:
                return apply({**row, "utterance": v}, predicted_action)                 # B
        if d.demote or d.rule is not None:
            return d
        if tool == "control_hvac" and tire_air_only(u):
            return Decision(predicted_action, True, "S5_NO_TARGET_TERM")               # E
        if wait_with_content(u):
            return Decision(predicted_action, True, "S1_REDO_WAIT")                     # C
        if noun_but(u):
            return Decision(predicted_action, True, "S3_NOUN_BUT")                      # D
        return d
    return apply
