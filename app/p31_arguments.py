# -*- coding: utf-8 -*-
"""P31: 引数の仕上げ。`p26_arguments` の上に重ねる（P26 までのソースは変えない）。

設計: DoC/設計書_P31_仕上げ.md の 3・4 節。

1. **機能の範囲外は断る**（利用者の決定・2026-10-13「できないことは断ってもよい」）: 読み出しの 5 機能で、機能の定義で説明がつく語が無ければ（`p31_scope`・単語の一致）、
   値に `__scope__` を付けて返す。検査（`policy.inspect`）は定義に無い引数を「その指定には対応していません」として**断る**（拒否）。聞き返しには入らない
2. **対象の言い方の取りこぼしを直す**（対象の側は閉じた集合なので、見て直してよい。評価セット v7・v8 の**対象の**行で、正しい機能なのに止まった・違う値になった行を見て直した。合否は新しい評価セット v9 で）

|直し|中身|v7・v8 の例|
|---|---|---|
|F1 音量ゼロ|「音量（ボリューム・音）を ゼロ・0・零」→ 音量 0|音量をゼロにしてください → 止まっていた|
|F2 ルートの言い方|「早く着く・空いて・混んでない・近い ＋ 道・ルート」「道・ルートに切り替え・変え」「〜を使わず・通らず・避けて」→ 引き直し（目的地が取れないか、動詞のかけらのとき）|早く着ける道に切り替えて → 止まっていた／高速を使わずに行ってください → 目的地「使わず」|
|F3 目的地のかけら|目的地が「〜て＋名詞」「〜で＋名詞」なら、後ろの名詞だけにする|電話で頼まれて駅まで → 目的地「頼まれて駅」|
|F4 曲の数を音量にしない|音量の値が、発話の中で「曲・回」などが続く数字からしか来ていなければ聞き返す|1曲リピートにして → 音量 1|
|F5 止めて、また流す|値が一時停止なのに、止める語の後に「再び・再開・続き・流して・かけて」があれば聞き返す|一時停止中の曲を再び流して → 一時停止|
|F6 歯止めの言い過ぎ（G1・G2）|空調の「自動」はオートを支える語にする。メディアの「Nつ飛ばし・Nつ先・Nつ進め」と、音量でない操作の「一度・一回」は数字の説明に使わない|エアコンを自動で調整するようにして／曲を2つ飛ばしてください／再生中の曲を一度ポーズにして → 止まっていた|

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

import re

from .p26_arguments import make_resolve as make_p26
from .p31_scope import READ, evidence

ZERO_VOL = re.compile(r"(?:音量|ボリューム|音)(?:を|は|も)?(?:ゼロ|0|０|零)")
REROUTE = re.compile(r"(?:早く着|はやく着|空いて|すいて|混んでな|近い|速い|短い)[^、。]{0,6}(?:道|ルート|経路)|(?:道|ルート|経路)(?:に|へ)(?:切り替え|切りかえ|変え)|(?:使わ|通ら|乗ら)ず|避けて")
VERB_FRAG = re.compile(r"(?:ず|ずに|て|って|れて|せて)$")
TE_NOUN = re.compile(r"^.*[ぁ-ん][てで]([一-龥ァ-ヴーA-Za-z][^てで]*)$")
COUNT_ONLY = re.compile(r"(\d+)\s*(?:曲|回|つ前|つ先|つ後|つ戻|分|秒)")
REPLAY = re.compile(r"(?:止め|停止|ポーズ|ストップ)[^、。]{0,8}(?:再び|再開|続き|流して|流す|かけて|かける|もう一度流)")
G2_TRACK = re.compile(r"\d+\s*つ\s*(?:飛ばし|先|進め|戻)")
ONCE = re.compile(r"一度|一回|1度|1回")


def make_resolve(A):
    p26 = make_p26(A)

    def fix(tool: str, u: str, out):
        """値の直し・歯止めの言い過ぎの取り消し。直したら新しい Arguments、直さないなら None"""
        if tool == "control_media":
            if ZERO_VOL.search(u) and (not out.complete or out.values.get("volume") not in (0, None)):
                return A.Arguments(values={"action": "volume_set", "volume": 0}, notes=["P31: F1 音量ゼロ"])
            if out.missing == ["value_check"] and out.notes and ("G2" in out.notes[0]) and (G2_TRACK.search(u) or (out.values.get("action") != "volume_set" and ONCE.search(u))):
                return A.Arguments(values=dict(out.values), notes=["P31: F6 歯止めの言い過ぎ（曲の数・一度）", *out.notes])
            if out.complete and out.values.get("action") == "volume_set":
                nums = re.findall(r"\d+", u)
                counted = {m.group(1) for m in COUNT_ONLY.finditer(u)}
                if nums and all(n in counted for n in nums) and str(out.values.get("volume")) in counted:
                    return A.Arguments(values=dict(out.values), missing=["value_check"], notes=["P31: F4 曲の数を音量にしない → 聞き返す"])
            if out.complete and out.values.get("action") == "pause" and REPLAY.search(u):
                return A.Arguments(values=dict(out.values), missing=["value_check"], notes=["P31: F5 止めて、また流す → 聞き返す"])
        if tool == "control_hvac":
            if out.missing == ["value_check"] and out.notes and "G1" in out.notes[0] and "auto" in out.notes[0] and "自動" in u:
                return A.Arguments(values=dict(out.values), notes=["P31: F6 歯止めの言い過ぎ（自動）", *out.notes])
        if tool == "set_navigation":
            dest = out.values.get("destination") or ""
            if REROUTE.search(u) and (not out.complete or VERB_FRAG.search(dest) or not dest):
                return A.Arguments(values={"action": "reroute"}, notes=["P31: F2 ルートの言い方"])
            if out.complete and out.values.get("action") in ("set", "add_waypoint") and dest:
                m = TE_NOUN.match(dest)
                if m and len(m.group(1)) >= 1:
                    return A.Arguments(values={**out.values, "destination": m.group(1)}, notes=[f"P31: F3 目的地のかけら {dest} → {m.group(1)}"])
        return None

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        out = p26(tool, argmode, utterance, state, **kw)
        if kw.get("recent_poi"):
            return out
        fixed = fix(tool, utterance, out)
        if fixed is not None:
            out = fixed
        if tool in READ and out.complete and not evidence(tool, utterance):
            return A.Arguments(values={**out.values, "__scope__": True}, notes=[f"P31: {tool} の定義で説明がつく語が無い → 断る（その機能はありません）", *out.notes])
        return out
    return resolve
