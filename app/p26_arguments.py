# -*- coding: utf-8 -*-
"""P26: 引数の歯止めの 2 段目。**発話の中に、取り出した値とは別の値の根拠があれば**聞き返す。`p25_arguments` は変えない（上に重ねる）。

設計: DoC/設計書_P26_歯止めの2段目.md の 3 節。

P25 の歯止め（G1〜G4）は「値に根拠があるか」を見た。見ていない行で、違う値での実行は 6.7% → 4.2% に減ったが、
残った行は「根拠が 2 つあって、片方を取り違えた」型だった（止める語があるのに入れる・「半」「いっぱい」を読まずに 1 段だけ動かす・理由の節の語を値にする・「寄って」を目的地にする）。
ここでは、**取り出した値と食い違う根拠が発話にあれば聞き返す**。

|歯止め|中身|止める例|
|---|---|---|
|G5 止める語と値の食い違い（空調）|空調の語のすぐ後に止める語があるのに、値がその対象のまま（冷房を切りたい → cool・エアコンを消して → auto・風を止めて → 風量が 0 でない）。または、値が「止める」なのに、打ち消した止める語（切らずに・止めないで）がある|冷房を切りたいな → cool／エアコンは切らずに風だけ止めて → off|
|G6 止める語と再生の食い違い（メディア）|値が「再生」なのに、止める語（止めて・ストップ・一時停止）がある（打ち消し・「止めた曲を続きから」は除く）|電話をかけるので曲を止めておいて → 再生|
|G7 程度の語と値の食い違い|値の語（音量・風量・温度）のすぐ後に最大・最小・いっぱい・ゼロ・一番 などがあるのに、値がいまの値から 1 段だけ動いた。「半分」なのに値がいまの値の半分でない。「N度半」なのに値が N.5 でない|音量をいっぱいまで上げて → 14／今の半分に下げて → 10／二十四度半 → 24|
|G8 ひらがなの数|ひとつ・ふたつ・みっつ … があるのに、値といまの値の差がその数でない|風量をふたつ上げて → 4|
|G9 別の場所への移動の語（ナビ）|目的地の設定（経由地の追加は除く）で、目的地のほかに、場所の語 ＋ に・へ・まで ＋ 移動の語（寄る・向かう・行く・着く・戻る）がある。「A に寄ってから B」と、目的の語（修理に行く・買い物に行く など）は除く|駅に着く前に花屋に寄って → 目的地 花屋／トイレに行きたいからサービスエリアに寄って → 目的地 トイレ|
|G10 入れる指定と値の食い違い（空調）|暖房・冷房・デフロスターを「入れる・つける」と言っているのに、値に mode が無い（温度や風量だけを動かした）|足元が冷えるので暖房をつけて → 温度 26|

止めたときは、引数が足りない（`missing = ["value_check"]`）として返す（P25 と同じ）。

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

import re

from .p23_arguments import KANJI_NUM, _num
from .p25_arguments import NEGATED, STATE_KEY, WRITE, make_resolve as make_p25

# G5
HVAC_STOP2 = re.compile(r"(エアコン|冷房|暖房|空調|ヒーター|クーラー|デフロスター?|曇り止め|曇り取り|送風|風)(?:を|は|も|だけ)?(?:もう)?(?:切|消|止め|とめ|オフ)")
# 止める語の対象ごとに、食い違いとみなす値（冷房を切る → cool のまま なら食い違い）
STOP_TARGET = {"冷房": ("cool",), "クーラー": ("cool",), "暖房": ("heat",), "ヒーター": ("heat",), "デフロスタ": ("defrost",), "デフロスター": ("defrost",),
               "曇り止め": ("defrost",), "曇り取り": ("defrost",), "エアコン": ("auto", "cool", "heat", "defrost", None), "空調": ("auto", "cool", "heat", "defrost", None)}
# G6
MEDIA_STOP = re.compile(r"止め|とめて|とめと|ストップ|一時停止|停止|ポーズ")
MEDIA_NEGATED = re.compile(r"(?:止め|とめ|停止し)(?:ないで|ずに)|止めるんじゃなく|停止じゃなく|ストップじゃなく|止めた|止めてた|止まった|止まって")      # 「止めた曲を続きから」は再生
# G7
EXTREME = re.compile(r"(?:音量|ボリューム|音|風量|風の量|風の強さ|風|温度)(?:を|は|の|も)?[^、。]{0,4}?(?:いっぱい|目いっぱい|最大|マックス|最高|全開|最小|最低|ゼロ|一番|いちばん|フル)")      # 値の語のすぐ後の程度の語だけ（「窓を全開にしたら」は見ない）
HALF_OF = re.compile(r"半分")
HALF = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*度\s*半")
# G8
KANA_NUM = {"ひとつ": 1, "ふたつ": 2, "みっつ": 3, "よっつ": 4, "いつつ": 5}
# G9
PLACE_MOVE = re.compile(r"([一-龥ァ-ヶー々A-Za-z0-9]{1,15})(?:に|へ|まで)(?:寄|向か|行|着|戻|帰)")
VIA_THEN = re.compile(r"寄ってから|寄った後|寄ったあと")
PURPOSE = ("修理", "買い物", "食事", "散歩", "仕事", "迎え", "送り", "トイレ", "休憩", "用事", "給油", "充電", "お参り", "見学", "診察")
# G10
MODE_ON = re.compile(r"(?:暖房|冷房|クーラー|ヒーター|デフロスター?|曇り止め|曇り取り)(?:を|は)?(?:つけ|付け|入れ|オン)")


def _state(state: dict, key: str):
    a, b = STATE_KEY[key]
    v = state.get(a, {}).get(b)
    return None if v is None else float(v)


def check(tool: str, utterance: str, values: dict, state: dict) -> str | None:
    u = KANJI_NUM.sub(_num, utterance)
    if tool == "control_hvac":
        mode = values.get("mode")
        v = NEGATED.sub("", u)
        for m in HVAC_STOP2.finditer(v):
            noun = m.group(1)
            if noun in ("風", "送風"):
                bad = mode != "off" and values.get("fan_level") not in (0, None)
            else:
                bad = mode in STOP_TARGET.get(noun, ())
            if bad:
                return f"G5: 「{noun}」を止める語があるのに、値が止めるでない"
        if mode == "off" and NEGATED.search(u):
            return "G5: 値は止めるだが、打ち消した止める語がある"
        if "mode" not in values and MODE_ON.search(u):
            return "G10: 暖房・冷房などを入れると言っているのに、値に mode が無い"
    if tool == "control_media" and values.get("action") == "play":
        if MEDIA_STOP.search(MEDIA_NEGATED.sub("", u)):
            return "G6: 止める語があるのに、値が再生"
    if tool in ("control_hvac", "control_media"):
        for key in ("temperature", "fan_level", "volume"):
            if key not in values or values[key] is None or tool != ("control_media" if key == "volume" else "control_hvac"):
                continue
            now = _state(state, key)
            got = float(values[key])
            step = {"temperature": 1.0, "fan_level": 1.0, "volume": 2.0}[key]
            if now is not None and EXTREME.search(u) and abs(abs(got - now) - step) < 1e-9:
                return "G7: 最大・最小などの語があるのに、値が 1 段だけ動いた"
            if now is not None and HALF_OF.search(u) and abs(got - now / 2) > step / 2 + 1e-9:
                return "G7: 「半分」なのに、値がいまの値の半分でない"
            m = HALF.search(u)
            if key == "temperature" and m and abs(got - (float(m.group(1)) + 0.5)) > 1e-9:
                return "G7: 「N度半」なのに、値が N.5 でない"
            for w, n in KANA_NUM.items():
                if w in u and now is not None and abs(abs(got - now) - n) > 1e-9 and abs(got - n) > 1e-9:
                    return f"G8: 「{w}」なのに、値といまの値の差が {n} でない"
    if tool == "set_navigation" and values.get("action") == "set" and values.get("destination") and not VIA_THEN.search(u):
        d = str(values["destination"])
        others = [m.group(1) for m in PLACE_MOVE.finditer(u) if d not in m.group(1) and m.group(1) not in d and not any(w in m.group(1) for w in PURPOSE)]
        if others:
            return f"G9: 目的地 {d!r} のほかに、別の場所への移動の語がある（{others[0]}）"
    return None


def make_resolve(A):
    p25 = make_p25(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        out = p25(tool, argmode, utterance, state, **kw)
        if tool not in WRITE or not out.complete or kw.get("recent_poi"):
            return out
        why = check(tool, utterance, out.values, state)
        if why is None:
            return out
        return A.Arguments(values=dict(out.values), missing=["value_check"], notes=[f"P26: {why} → 聞き返す", *out.notes])
    return resolve
