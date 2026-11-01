# -*- coding: utf-8 -*-
"""P35: 引数の仕上げの 3 回目。`p33_arguments` の上に重ねる（P33 までのソースは変えない）。

設計: DoC/設計書_P35_重大の余裕と返すものの広げ.md の 3・4 節。

1. **E3 止める語を、後ろに別の操作があるときは見ない**: 「ライトを**消して**夜景を見ながら聴く音楽を流して」「さっき**止めた**曲を再生して」が一時停止になっていた（v10 の値まで見た重大な誤り 2 行）。
   止める語（止め・停止・ポーズ・ストップ・消し）の**後ろに再生の語**（かけ・流し・再生・聴か…）があれば、その止める語は前置き・連体修飾とみなして伏せ、もう一度取り出す。
   空調も同じ（切っ・消し・止め・オフ の後ろに入れる・強める・温める などがあれば伏せる）。「曇り止め」は伏せない
2. **返すもので絞るのを 4 つの読み出しの機能に広げる**（`p35_scope`。車両の状態・曲の情報を足し、施設の語の末尾 1 文字の不具合を直した）。
   P33 の判断は残し、その上に p35_scope でもう一度見る（P35 の方が厳しいので、P33 が通して P35 が断る行だけ増える）

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

import re

from .p33_arguments import make_resolve as make_p33
from .p35_scope import explains

M_STOP = ("止め", "とめ", "停止", "ポーズ", "ストップ", "消し")
M_PLAY = ("かけ", "流し", "流す", "再生", "聴か", "聞か", "聴き", "聞き")
H_STOP = ("切っ", "消し", "止め", "とめ", "オフ", "停止")
H_ON = ("入れ", "つけ", "上げ", "下げ", "強め", "弱め", "強く", "弱く", "暖め", "温め", "あたため", "冷や", "涼しく", "暖かく")     # 「オフにして」の「にして」は止める語と同じ述語なので入れない
KEEP = re.compile(r"(?:曇り|くもり)止め")


def masked_stop(tool: str, u: str) -> str | None:
    """止める語の後ろに別の操作の語があれば、止める語を伏せた発話（伏せる所が無ければ None）"""
    stops, goes = (M_STOP, M_PLAY) if tool == "control_media" else (H_STOP, H_ON)
    keep = {m.start() + k for m in KEEP.finditer(u) for k in range(len(m.group(0)))}
    pos = [(m.start(), w) for w in stops for m in re.finditer(re.escape(w), u) if m.start() not in keep]
    if not pos:
        return None
    last_go = max((m.start() for w in goes for m in re.finditer(re.escape(w), u)), default=-1)
    hide = [(i, w) for i, w in pos if i < last_go]
    if not hide or len(hide) < len(pos):          # 後ろに操作が無い止める語が 1 つでもあれば、伏せない（止めるのが本意）
        return None
    v = list(u)
    for i, w in hide:
        for k in range(i, i + len(w)):
            v[k] = "〇"
    return "".join(v)


def make_resolve(A):
    p33 = make_p33(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        if tool in ("control_media", "control_hvac") and not kw.get("recent_poi"):
            m = masked_stop(tool, utterance)
            if m is not None:
                out = p33(tool, argmode, m, state, **kw)
                return A.Arguments(values=dict(out.values), missing=list(out.missing), notes=["P35: E3 止める語の後ろに別の操作 → 止める語を伏せて取り出した", *out.notes])
        out = p33(tool, argmode, utterance, state, **kw)
        if kw.get("recent_poi"):
            return out
        if out.complete and "__scope__" not in out.values:
            ok, why = explains(tool, utterance)
            if not ok:
                return A.Arguments(values={**out.values, "__scope__": True}, notes=[f"P35: {tool} は {why} → 断る（その機能はありません）", *out.notes])
        return out
    return resolve
