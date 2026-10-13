# -*- coding: utf-8 -*-
"""P9: 引数の取り出しを補う層（お試し）。EdgeSLM_MultiHead_Multi の `dispatch/arguments.py`（md5 f1f167e9）は**変えない**。

設計: DoC/設計書_P9_引数の取り出しを補う.md。

元の `arguments.resolve` をまず呼び、**引数が足りないと返ったときだけ**、ここの追加の規則を試す。
元の処理で通っていた発話の結果は変わらない（足すだけ）。追加の規則は、学習側の発話（714 行・学習に足した行）で
取り出せなかった言い方から作った。評価セット（challenge_v1・v2）の文に合わせて作らない。

|ツール|足した規則|例|
|---|---|---|
|ナビ|中止（ナビの語があるとき）|ナビを消して／ルート案内は終了していいよ|
|ナビ|ルートの引き直し（高速・下道・一般道・有料道路・別のルート）|高速を使わないで／他の道で案内して|
|ナビ|経由地（途中・経由・寄り道 ＋ 種類の語）|途中で給油したい|
|ナビ|目的地の言い方を足す|東京駅を目的地にして／目的地を市役所に設定して／空港までお願い／家までの道を案内して／家に帰る|
|空調|席の指定（後ろ・助手席・運転席）→ zone|後ろの席にも風を送って|
|空調|強め・弱め → 今の値から 1 段|風量を少し弱めて|
|メディア|音が大きい・うるさい → 下げる／半分 → 今の半分|ちょっと音が大きい／音量を半分にして|

足しても取り出せないもの（聞き返すのが正しい）: 目的地の無い変更（`目的地を変更して`）・指示語の目的地（`最寄りの駅まで`。周辺検索の結果が要る）・
schema に無い指定（`足元に風を送って`）。
"""
from __future__ import annotations

import re

NAV_WORDS = ("ナビ", "案内", "ルート", "目的地", "行き先", "行先")
CANCEL_WORDS = ("消して", "終了", "いらない", "止めて", "ストップ", "やめ", "中止", "キャンセル", "取り消")
REROUTE_WORDS = ("別のルート", "他のルート", "他の道", "別の道", "引き直", "高速", "下道", "一般道", "有料道路", "有料を")
WAYPOINT_WORDS = ("経由", "寄り道", "途中")
HOME = r"家|自宅|うち|実家|会社|職場"
DEST = r"目的地|行き先|行先|デスティネーション|ゴール"
# 前置き（理由・状況）の終わり。目的地はその後ろにある
_CLAUSE = re.compile(r"、|から|ので|けど|して|たら|なら")
_ADVERBS = ("いったん", "一度", "とりあえず", "まず", "ひとまず", "今から", "これから")
PATTERNS = (
    re.compile(rf"(?:{DEST})[をは](.+?)に(?:して|設定|変更|セット|変え)"),          # 目的地を市役所に設定して
    re.compile(rf"(.+?)を(?:{DEST})に"),                                            # 東京駅を目的地にして
    re.compile(r"(.+?)(?:まで|へ)(?:お願い|の道|のルート|の案内|の道案内|連れ|ナビ|案内)"),   # 空港までお願い／家までの道を案内して
    re.compile(rf"({HOME})(?:まで|へ|に)"),                                          # 家まで案内して（1 文字の目的地）
    re.compile(r"(.+?)に(?:帰|戻)"),                                                # 家に帰る
)
ZONES = (("rear", ("後ろ", "後席", "リア")), ("passenger", ("助手席",)), ("driver", ("運転席",)))
UP = ("強め", "上げ", "強く")
DOWN = ("弱め", "下げ", "弱く", "絞")
LOUD = ("大きい", "大きすぎ", "うるさ", "でかい")
QUIET = ("小さい", "小さすぎ", "聞こえな")


def _last_chunk(text: str) -> str:
    """前置きを落として、最後のかたまりを目的地の候補にする"""
    t = _CLAUSE.split(text)[-1].strip()
    for a in _ADVERBS:
        if t.startswith(a):
            t = t[len(a):]
    return t.strip("、 をはに")


def _navigation(utterance: str, base, A):
    has_nav = any(w in utterance for w in NAV_WORDS)
    if has_nav and any(w in utterance for w in CANCEL_WORDS):
        return A.Arguments(values={"action": "cancel"}, notes=["P9: ナビの語と中止の語 → cancel"])
    if any(w in utterance for w in REROUTE_WORDS):
        return A.Arguments(values={"action": "reroute"}, notes=["P9: ルートの条件・引き直し → reroute"])
    if any(w in utterance for w in WAYPOINT_WORDS):
        for category, words in A.CATEGORY_HINTS:
            if any(w in utterance for w in words):
                return A.Arguments(values={"action": "add_waypoint", "destination": category}, notes=[f"P9: 経由地 {category}"])
        return base                                                  # 経由地が決まらない → 聞き返す
    for pat in PATTERNS:
        m = pat.search(utterance)
        if not m:
            continue
        dest = A.clean_destination(_last_chunk(m.group(1)))
        if not dest or len(dest) > 15 or A.is_anaphoric_destination(dest) or any(w in dest for w in NAV_WORDS):
            continue
        return A.Arguments(values={"action": "set", "destination": dest}, notes=[f"P9: 目的地として {dest!r} を抽出"])
    return base


def _hvac(utterance: str, state: dict, base, A):
    current = state.get("hvac", {})
    slot = base.missing[0] if base.missing else None
    if slot in ("fan_level", "temperature") and current.get(slot) is not None:
        sign = +1 if any(w in utterance for w in UP) else -1 if any(w in utterance for w in DOWN) else 0
        if sign:
            value = A.clamp(slot, current[slot] + sign * A.STEPS[slot])
            return A.Arguments(values={slot: int(value) if slot == "fan_level" else value}, notes=[f"P9: 今の値から {sign:+d} 段"])
    for zone, words in ZONES:
        if any(w in utterance for w in words):
            return A.Arguments(values={"zone": zone}, notes=[f"P9: 席の指定 → zone={zone}"])
    return base


def _media(utterance: str, state: dict, base, A):
    volume = state.get("media", {}).get("volume")
    if volume is None:
        return base
    if "半分" in utterance:
        return A.Arguments(values={"action": "volume_set", "volume": int(volume // 2)}, notes=["P9: 半分 → 今の半分"])
    sign = -1 if any(w in utterance for w in LOUD) else +1 if any(w in utterance for w in QUIET) else 0
    if sign:
        return A.Arguments(values={"action": "volume_set", "volume": int(A.clamp("volume", volume + sign * A.STEPS["volume"]))},
                           notes=[f"P9: 音の大きさの訴え → {sign:+d} 段"])
    return base


def make_resolve(A):
    """元の `arguments` モジュールを受け取り、補う層を重ねた resolve を返す"""
    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        base = A.resolve(tool, argmode, utterance, state, **kw)
        if base.complete:
            return base                                              # 元の処理で決まったものは変えない
        if tool == "set_navigation":
            return _navigation(utterance, base, A)
        if tool == "control_hvac":
            return _hvac(utterance, state, base, A)
        if tool == "control_media":
            return _media(utterance, state, base, A)
        return base
    return resolve
