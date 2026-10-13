# -*- coding: utf-8 -*-
"""P25: 引数の**歯止め**。取り出した値に、発話の中の根拠が無ければ、実行せずに聞き返す。`p23_arguments` は変えない（上に重ねる）。

設計: DoC/設計書_P25_引数の歯止め.md の 3 節。

P23 までは「値を正しく取り出す」直しだった。見ていない行で、違う値での実行は 27% → 11% → 4.2% と減ったが、直すと別の外し方が出る段階に来た。
ここでは考え方を変える。**値を取り出せたかではなく、その値に根拠があるかを確かめ、根拠が無ければ聞き返す**。
違う値での実行は重大な誤り（利用者が決めた・2026-10-07）。聞き返しは回復できる誤り。**重大な誤りを減らす代わりに聞き返しが増える交換は受け入れる**（利用者・2026-10-09）。

操作系の 3 つ（空調・メディア・ナビ）で、`p23_arguments` が「そろった」と返したときだけ見る。読むだけの機能は見ない。

|歯止め|中身|止める例|
|---|---|---|
|G1 既定値には根拠の語|元の処理が「指定が無いので既定値で実行」したとき、それを支える語（空調: つける・入れる・動かす／メディア: かける・流す・再生・ラジオ など）が無ければ聞き返す。別の操作の語（切る・強く・次・音量 など）があれば聞き返す|冷房きって → auto／音おおきく → 再生|
|G2 数字は使い切る|空調・メディアで、発話の数字が、取り出した値で説明できなければ聞き返す。説明できる = 値がその数字（「〜に・〜まで」の後なら、これだけ）か、いまの値との差がその数字。「倍」はいまの値 × 数字|温度は24度、風量は2にして → 風量だけ／設定を23度に → 24 度／音量を3倍 → 3|
|G3 操作を 2 つ言う（空調）|取り出した値が「止める」なのに、入れる・強弱の語（入れる・つける・強く・弱く・上げる・下げる）もある。または、空調の語のすぐ後に止める語（冷房を切って など）があるのに、値は止める以外で、入れる・強弱の語もある。どちらも聞き返す（1 回の実行で 1 つの mode しか渡せない）|デフロスターを止めて暖房を入れて → off|
|G4 目的地に余計な語（ナビ）|目的地に、副詞（できるだけ・早く・すぐ・先に・とりあえず など）・述語の切れ端（して・って・たい・ので・から）が入る、または 20 字を超えるなら聞き返す|できるだけ早く会社|

止めたときは、引数が足りない（`missing = ["value_check"]`）として返す。流れの中では「引数が足りない」の段で聞き返しになる。

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

import re

from .p23_arguments import KANJI_NUM, _num, make_resolve as make_p23

WRITE = ("control_hvac", "control_media", "set_navigation")
NUMERIC = ("temperature", "fan_level", "volume")
STATE_KEY = {"temperature": ("hvac", "temperature"), "fan_level": ("hvac", "fan_level"), "volume": ("media", "volume")}

# G1 既定値を支える語
SUPPORT_HVAC = re.compile(r"つけ|付け|点け|入れ|動か|オン|スイッチ|起動|かけ")
SUPPORT_MEDIA = re.compile(r"かけ|掛け|流し|流す|流そ|再生|スタート|プレイ|ラジオ|FM|AM|リピート|聴|聞きたい|BGM|音楽に|曲に|曲を変え|曲変え|にして|(?:音楽|曲).{0,4}(?:ほしい|欲しい)")
OTHER_HVAC = re.compile(r"切|消|止|オフ|強|弱|上げ|下げ|高く|低く|最大|最小|暖か|涼し|冷や")
OTHER_MEDIA = re.compile(r"次|前|戻|飛ば|スキップ|音量|ボリューム|大き|小さ|上げ|下げ|止|消|ミュート|最初")
# G2 数字
NUM = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?")
NOT_AMOUNT = re.compile(r"^\s*(?:曲|回|分|秒|時|号|キロ|km|人|個|枚|件|番|日|年|月|週|メートル|m(?![a-z])|%|パーセント)")
TRACK = re.compile(r"^\s*つ\s*(?:前|後|先|戻)")       # 「1つ前」は曲の数
KEEP = re.compile(r"^\s*(?:度|段階|段目|段|つ)?\s*(?:でいい|のまま|はそのまま|で大丈夫|でよい|で十分)")      # 「風量は2でいいから」は、変えない値の話
ABSOLUTE = re.compile(r"^\s*(?:度|段階|段目|段|目盛り?|つ)?\s*(?:くらい|ぐらい|程度|位|ほど)?\s*(?:に|まで|へ)")
TIMES = re.compile(r"^\s*倍")
# G3 操作を 2 つ
NEGATED = re.compile(r"(?:止め|切ら|消さ|とめ)(?:ないで|ずに)")
# 空調の語のすぐ後の止める語（「ライトを消して」は空調の止める語ではない）
HVAC_STOP = re.compile(r"(?:エアコン|冷房|暖房|空調|ヒーター|クーラー|デフロスター?|曇り止め|曇り取り|送風|風)(?:を|は|も)?(?:もう)?(?:切|消|止め|とめ|オフ)")
GO_WORD = re.compile(r"入れ|つけ|付け|オン|強|弱|上げ|下げ")
# G4 目的地
DEST_NOISE = re.compile(r"できるだけ|なるべく|早く|急いで|すぐ|とりあえず|ひとまず|いったん|先に|まずは|もう一度|もう一回|そのまま|ゆっくり|今から|いまから|これから"
                        r"|して|って|たい|ない|ので|から|けど|ください|ほしい")
DEST_MAX = 20


def _zen(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９．", "0123456789."))


def _numbers(u: str):
    """発話の中の数字と、その後ろの文字列"""
    for m in NUM.finditer(u):
        if u[max(0, m.start() - 2):m.start()] == "国道":
            continue
        yield float(_zen(m.group())), u[m.end():]


def _state(state: dict, key: str):
    a, b = STATE_KEY[key]
    v = state.get(a, {}).get(b)
    return None if v is None else float(v)


def check(tool: str, utterance: str, values: dict, notes: list, state: dict) -> str | None:
    """止める理由（止めなければ None）"""
    u = KANJI_NUM.sub(_num, utterance)          # 発話そのもの（漢数字だけそろえる。後ろの理由や打ち消しの前も含めて見る）
    first = notes[0] if notes else ""
    if tool == "control_hvac" and "既定値" in first:
        if not SUPPORT_HVAC.search(u) or OTHER_HVAC.search(u):
            return "G1: 既定値（auto）を支える語が無い・別の操作の語がある"
    if tool == "control_media" and first.startswith("指定が無いので再生"):
        if not SUPPORT_MEDIA.search(u) or OTHER_MEDIA.search(u):
            return "G1: 既定値（再生）を支える語が無い・別の操作の語がある"
    if tool in ("control_hvac", "control_media"):
        nums = [(k, float(values[k])) for k in NUMERIC if k in values and values[k] is not None]
        for n, rest in _numbers(u):
            if NOT_AMOUNT.match(rest) or KEEP.match(rest) or TRACK.match(rest):
                continue
            if TIMES.match(rest):
                ok = any(_state(state, k) is not None and abs(v - _state(state, k) * n) < 1e-9 for k, v in nums)
            elif ABSOLUTE.match(rest):
                ok = any(abs(v - n) < 1e-9 for _, v in nums)
            else:
                ok = any(abs(v - n) < 1e-9 or (_state(state, k) is not None and abs(abs(v - _state(state, k)) - n) < 1e-9) for k, v in nums)
            if not ok:
                return f"G2: 数字 {n:g} を、取り出した値で説明できない"
    if tool == "control_hvac":
        v = NEGATED.sub("", u)
        if (values.get("mode") == "off" and GO_WORD.search(v)) or (values.get("mode") != "off" and HVAC_STOP.search(v) and GO_WORD.search(v)):
            return "G3: 止める操作と、入れる・強弱の操作の両方を言っている"
    if tool == "set_navigation" and values.get("destination"):
        d = str(values["destination"])
        if DEST_NOISE.search(d) or len(d) > DEST_MAX:
            return f"G4: 目的地 {d!r} に余計な語がある"
    return None


def make_resolve(A):
    p23 = make_p23(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        out = p23(tool, argmode, utterance, state, **kw)
        if tool not in WRITE or not out.complete or kw.get("recent_poi"):
            return out
        why = check(tool, utterance, out.values, out.notes, state)
        if why is None:
            return out
        return A.Arguments(values=dict(out.values), missing=["value_check"], notes=[f"P25: {why} → 聞き返す", *out.notes])
    return resolve
