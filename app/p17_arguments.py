# -*- coding: utf-8 -*-
"""P17: 引数の取り出しを補う層の 2 つ目。`p9_arguments` の上に重ねる。EdgeSLM_MultiHead_Multi の `arguments.py` は変えない。

設計: DoC/設計書_P17_不要な聞き返しを減らす.md の 4 節。

`p9_arguments` の resolve をまず呼び、**それでも引数が足りないときだけ**、下の規則を試す。
評価セット v3 で「引数が足りない」と聞き返した対象の依頼（層 17 で 10.3 行。ほとんどナビ）から作った。

|ツール|足した規則|例|
|---|---|---|
|ナビ|ルートの引き直しの語を足す（別ルート・最短・案内し直し・近道）|別ルートを探して／最短のルートで案内し直して|
|ナビ|経由地に、種類の語に無い場所を入れる|経由地に道の駅を追加して|
|ナビ|目的地の言い方を足す（行き先は X／X まで道案内）|ナビを開始、行き先は市民病院／ホテルまで道案内して|
|ナビ|目的の前置き（〜を替えに・〜を買いに）を落として目的地を取る|エンジンオイルを替えにガソリンスタンドを目的地にして|
|メディア|小さめ・控えめ → 1 段下げる／大きめ → 1 段上げる|音を小さめにして|

足さないもの（聞き返すのが正しい、のまま）:
- 「最寄りの駅まで」「一番近いガソリンスタンドへ」… 周辺の検索の結果が要る（EdgeSLM_MultiHead_Multi の設計どおり）
- 「送風だけにして」「顔に風が当たらないようにして」… ツールの引数に無い指定
"""
from __future__ import annotations

import re

from .p9_arguments import DEST, NAV_WORDS, make_resolve as make_p9

REROUTE_MORE = ("別ルート", "最短", "案内し直", "ルートを探", "近道", "早い道", "速い道")
PURPOSE = re.compile(r"^.*を[ぁ-ん一-龥]{1,6}に(?=[^にをは]+$)")          # 「エンジンオイルを替えに」「シートを受け取りに」を落とす
DEST_MORE = (
    re.compile(rf"(?:{DEST})は(.+?)(?:で|に|へ|まで)?(?:お願い|頼む|して)?$"),          # 行き先は市民病院
    re.compile(r"(.+?)(?:まで|へ)(?:道案内|ナビゲート|ガイド)"),                       # ホテルまで道案内して
    re.compile(rf"(.+?)を(?:{DEST})に"),                                            # （前置きを落としてから）ガソリンスタンドを目的地にして
)
WAYPOINT = (
    re.compile(r"経由地[にへ](.+?)を(?:追加|入れ|設定|セット|足)"),                     # 経由地に道の駅を追加して
    re.compile(r"(.+?)を経由地[にへ]"),                                              # 道の駅を経由地にして
    re.compile(r"(?:途中で|途中に)(.+?)[にへ]寄"),                                    # 途中で道の駅に寄って
)
SOFTER = ("小さめ", "控えめ", "静かめ", "低め")
LOUDER = ("大きめ",)


def _ok_dest(dest: str, A) -> bool:
    return bool(dest) and len(dest) <= 15 and not A.is_anaphoric_destination(dest) and not any(w in dest for w in NAV_WORDS)


def _navigation(utterance: str, base, A):
    if any(w in utterance for w in REROUTE_MORE):
        return A.Arguments(values={"action": "reroute"}, notes=["P17: ルートの引き直し → reroute"])
    for pat in WAYPOINT:
        m = pat.search(utterance)
        if m:
            dest = A.clean_destination(m.group(1).split("、")[-1].strip())
            if _ok_dest(dest, A):
                return A.Arguments(values={"action": "add_waypoint", "destination": dest}, notes=[f"P17: 経由地 {dest!r}"])
    text = utterance.split("、")[-1].strip()
    for cand in (text, PURPOSE.sub("", text)):
        for pat in DEST_MORE:
            m = pat.search(cand)
            if not m:
                continue
            dest = A.clean_destination(PURPOSE.sub("", m.group(1)).strip("、 をはに"))
            if _ok_dest(dest, A):
                return A.Arguments(values={"action": "set", "destination": dest}, notes=[f"P17: 目的地として {dest!r} を抽出"])
    return base


def _media(utterance: str, state: dict, base, A):
    volume = state.get("media", {}).get("volume")
    if volume is None:
        return base
    sign = -1 if any(w in utterance for w in SOFTER) else +1 if any(w in utterance for w in LOUDER) else 0
    if sign:
        return A.Arguments(values={"action": "volume_set", "volume": int(A.clamp("volume", volume + sign * A.STEPS["volume"]))},
                           notes=[f"P17: 小さめ・大きめ → {sign:+d} 段"])
    return base


def make_resolve(A):
    p9 = make_p9(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        base = p9(tool, argmode, utterance, state, **kw)
        if base.complete:
            return base                                              # P9 までで決まったものは変えない
        if tool == "set_navigation":
            return _navigation(utterance, base, A)
        if tool == "control_media":
            return _media(utterance, state, base, A)
        return base
    return resolve
