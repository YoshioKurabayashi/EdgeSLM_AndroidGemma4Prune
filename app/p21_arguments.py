# -*- coding: utf-8 -*-
"""P21: 引数の取り出しを補う層の 3 つ目。`p17_arguments` の上に重ねる。EdgeSLM_MultiHead_Multi の `arguments.py` は変えない。

設計: DoC/設計書_P21_規則と引数の誤りを直す.md の 4 節。

P20（評価セット v5）で「引数が足りない」と聞き返した対象の依頼（層 26 で 5.3 行）と、確かめの途中で見つけた誤った引数から作った。

|ツール|足した規則|例|
|---|---|---|
|ナビ|中止の語を足す（終わり・おしまい・切って・オフ。ナビの語があるとき）|ナビを終わりにして|
|ナビ|道の条件で行き方を言う → ルートの引き直し|空いてる道で行って／お金のかからない道で案内して|
|ナビ|**取り出した目的地が、かなに続く「道」「ルート」で終わる → 目的地にせず、引き直しにする**（元の処理が完成と返しても直す。「北海道」「国道」は当てない）|混んでない道に変えて（元は 目的地 =「混んでない道」）|
|ナビ|**「もう一度」「再開」で引き直しと返っても、指示語の場所（あの場所・あそこ・さっきの店）があれば聞き返す**（元の処理が完成と返しても直す）|あの場所へもう一度|
|空調|おまかせ・自動 → mode=auto|エアコンの温度調整をおまかせにして|
|メディア|絞る・控える → 1 段下げる|ボリュームをちょっとだけ絞って|

足さないもの（聞き返すのが正しい、のまま）: 「風を出して」「送風だけにして」（ツールの引数に、風だけを出す指定が無い）。
"""
from __future__ import annotations

import re

from .p9_arguments import NAV_WORDS
from .p17_arguments import make_resolve as make_p17

CANCEL_MORE = ("終わり", "おしまい", "切って", "オフ")
# 「混んでない道」「空いてるルート」のように、かなに続く道の語。「北海道」「国道」「ゴルフコース」は当てない
ROUTE_END = re.compile(r"[ぁ-ん](?:道|道路|ルート|経路|コース)$")
ROUTE_COND = re.compile(r"[ぁ-ん](?:道|道路|ルート|経路|コース)(?:で|に|を|が)?(?:行|案内|変え|向か|お願い|頼|し)")
AGAIN = ("もう一度", "再開")
PLACE_DEICTIC = re.compile(r"あそこ|そこ(?:へ|に|まで)|(?:あの|その|さっきの|例の|この前の|この間の)(?:場所|店|ところ|辺|へん|駅|あたり)")
AUTO = ("おまかせ", "お任せ", "自動", "オート")
DOWN_MEDIA = ("絞", "控え")


def make_resolve(A):
    p17 = make_p17(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        base = p17(tool, argmode, utterance, state, **kw)
        if tool == "set_navigation":
            v = base.values
            if base.complete and v.get("action") == "set" and ROUTE_END.search(str(v.get("destination", ""))):
                return A.Arguments(values={"action": "reroute"}, notes=[f"P21: 目的地 {v['destination']!r} は道の条件 → reroute"])
            if base.complete and v.get("action") == "reroute" and any(w in utterance for w in AGAIN) and PLACE_DEICTIC.search(utterance):
                return A.Arguments(values={"action": "set"}, missing=["destination"], notes=["P21: 指示語の場所がある。引き直しにせず確認へ"])
            if base.complete:
                return base
            if any(w in utterance for w in NAV_WORDS) and any(w in utterance for w in CANCEL_MORE):
                return A.Arguments(values={"action": "cancel"}, notes=["P21: ナビの語と中止の語 → cancel"])
            if ROUTE_COND.search(utterance) and not A.is_anaphoric_destination(utterance):
                return A.Arguments(values={"action": "reroute"}, notes=["P21: 道の条件 → reroute"])
            return base
        if base.complete:
            return base
        if tool == "control_hvac" and any(w in utterance for w in AUTO):
            return A.Arguments(values={"mode": "auto"}, notes=["P21: おまかせ・自動 → mode=auto"])
        if tool == "control_media" and any(w in utterance for w in DOWN_MEDIA):
            volume = state.get("media", {}).get("volume")
            if volume is not None:
                return A.Arguments(values={"action": "volume_set", "volume": int(A.clamp("volume", volume - A.STEPS["volume"]))},
                                   notes=["P21: 絞る・控える → 1 段下げる"])
        return base
    return resolve
