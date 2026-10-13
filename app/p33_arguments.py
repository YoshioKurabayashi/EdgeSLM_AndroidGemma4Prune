# -*- coding: utf-8 -*-
"""P33: 引数の仕上げの 2 回目。`p31_arguments` の上に重ねる（P31 までのソースは変えない）。

設計: DoC/設計書_P33_何を返すかで絞る.md の 3・4 節。

1. **機能が何を返すかで絞る**（`p33_scope`・利用者の決定 2026-10-13）: 周辺の検索・説明書で、聞いているものが返せるもので説明がつかなければ、
   P31 と同じく値に `__scope__` を付けて返す（検査が「その機能はありません」と断る）
2. **最大・最小の語を単語の切れ目で見る**（E1）: 元の引数の取り出しは「フル」「最高」などを部分一致で見るので、「シャッ**フル**」で音量が最大になる。
   最大・最小の語（`MAX_WORDS`・`MIN_WORDS`）が単語の切れ目に合わない所にしか無ければ、その部分を伏せてもう一度取り出す
3. **経由地の前の「途中に」**（E2）: 目的地・経由地が「〜途中に／途中で／ついでに ＋ 名詞」なら後ろの名詞だけ（行く途中にコンビニ → コンビニ）

E1・E2 は v9（直す用になった）の対象の行で見つけた。合否は新しい評価セット v10 で。

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

import re

from .p30_words import aligned
from .p31_arguments import make_resolve as make_p31
from .p33_scope import explains

ON_THE_WAY = re.compile(r"^.*(?:途中に|途中で|ついでに)(.+)$")


def masked(u: str, A) -> str | None:
    """最大・最小の語が部分一致でしか無いとき、その部分を伏せた発話（伏せる所が無ければ None）"""
    terms = tuple(A.MAX_WORDS) + tuple(A.MIN_WORDS)
    try:
        hit = [t for t in terms if t in u]
        if not hit or any(aligned(t, u, stem=True) for t in hit):
            return None
    except SystemExit:
        return None
    v = u
    for t in sorted(hit, key=len, reverse=True):
        v = v.replace(t, "〇" * len(t))
    return v


def make_resolve(A):
    p31 = make_p31(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        if tool in ("control_media", "control_hvac"):
            m = masked(utterance, A)
            if m is not None:
                out = p31(tool, argmode, m, state, **kw)
                return A.Arguments(values=dict(out.values), missing=list(out.missing), notes=["P33: E1 最大・最小の語が部分一致だけ → 伏せて取り出した", *out.notes])
        out = p31(tool, argmode, utterance, state, **kw)
        if kw.get("recent_poi"):
            return out
        if tool == "set_navigation" and out.complete and out.values.get("action") in ("set", "add_waypoint"):
            d = out.values.get("destination") or ""
            mm = ON_THE_WAY.match(d)
            if mm and mm.group(1):
                return A.Arguments(values={**out.values, "destination": mm.group(1)}, notes=[f"P33: E2 途中に {d} → {mm.group(1)}", *out.notes])
        if out.complete and "__scope__" not in out.values:
            ok, why = explains(tool, utterance)
            if not ok:
                return A.Arguments(values={**out.values, "__scope__": True}, notes=[f"P33: {tool} は {why} → 断る（その機能はありません）", *out.notes])
        return out
    return resolve
