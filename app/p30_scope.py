# -*- coding: utf-8 -*-
"""P30: P29 の「機能の定義で絞る」を、**部分一致ではなく単語の切れ目に合わせた一致**にする。語の一覧は P29 のまま（＋対象の言い方 4 語）。

設計: DoC/設計書_P30_語単位で一致させる.md の 3 節。

P29 は語の一覧を発話の部分文字列で探したので、「宿題」の「宿」・「スキー場」の「キー」・「名前」の「前」が根拠になった。
P30 では、発話を端末の BERT と同じ分け方（NFKC ＋ MeCab ＋ unidic-lite・`p30_words`）で単語に分け、語が**単語の切れ目から始まる**ときだけ根拠にする。

|語の長さ|一致の決まり|ねらい|
|---|---|---|
|2 文字以上|単語の始まりに合えばよい（語幹・複合語の頭: 「走れる」の「走れ」・「前方」の「前方」）|動詞の活用・複合語を落とさない|
|1 文字で、向き・曲の語（前・後・左・右・横・曲・歌）|単語の始まりに合えばよい（「右側」「左前」「曲名」「歌って」）|向きの複合語・「歌う」の活用|
|1 文字で、それ以外（宿・屋・店・鍵・窓・駅・局 など）|単語そのものと同じときだけ（「ラーメン 屋」は通る・「屋根」「宿題」は通らない）|接尾の語の部分一致を防ぐ|

語の一覧に足した対象の言い方（v6〜v8 と学習の**対象の**行で、単語の一致にしたら落ちた行を見て足した）: 真後ろ・タコメーター（カメラ・状態）、後輪・前輪（車の部品）。

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

from . import p29_scope as P29
from .p26_arguments import make_resolve as make_p26
from .p30_words import aligned

PREFIX_1 = set("前後左右横曲歌")
CAR_PARTS = P29.CAR_PARTS + ("後輪", "前輪", "タコメーター")
STATUS_ITEM, STATUS_ALL = P29.STATUS_ITEM, P29.STATUS_ALL
CAMERA = P29.CAMERA + ("真後ろ",)
MEDIA_INFO, POI_KIND, POI_PLACE, READ = P29.MEDIA_INFO, P29.POI_KIND, P29.POI_PLACE, P29.READ


def has(words, u: str) -> list[str]:
    return [w for w in words if aligned(w, u, stem=len(w) >= 2 or w in PREFIX_1)]


def evidence(tool: str, u: str) -> list[str]:
    """その機能の定義で説明がつく語（単語の切れ目に合うものだけ。空なら説明がつかない）"""
    if tool == "get_vehicle_status":
        return has(STATUS_ITEM, u) or has(STATUS_ALL, u) or has(CAR_PARTS, u)
    if tool == "capture_camera":
        return has(CAMERA, u)
    if tool == "get_media_info":
        return has(MEDIA_INFO, u)
    if tool == "search_vehicle_manual":
        return has(CAR_PARTS, u)
    if tool == "search_poi":
        return has(POI_KIND, u) or has(POI_PLACE, u)
    return ["(操作の機能は見ない)"]


def make_resolve(A):
    p26 = make_p26(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        out = p26(tool, argmode, utterance, state, **kw)
        if tool not in READ or not out.complete or kw.get("recent_poi"):
            return out
        if evidence(tool, utterance):
            return out
        return A.Arguments(values=dict(out.values), missing=["scope"], notes=[f"P30: {tool} の定義で説明がつく語が無い（単語の一致）→ 聞き返す", *out.notes])
    return resolve
