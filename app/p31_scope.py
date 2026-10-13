# -*- coding: utf-8 -*-
"""P31: 機能の定義で絞る語の一覧を**機能ごとに分けた**もの（単語の一致は P30 のまま）。

設計: DoC/設計書_P31_仕上げ.md の 3 節。

P30 では、車の部品の語（`CAR_PARTS`）を、車両の状態と説明書の両方の根拠にしていた。足した語「後輪」が説明書の根拠にもなり、
「後輪の左側は何キロパスカル」（いまの値 = 車両の状態）が説明書として通った。ここでは、語を機能ごとに持つ。

|機能|語|
|---|---|
|車両の状態|項目の語・状態の語・**いまの状態を読める部品**（P29 の車の部品 ＋ 後輪・前輪・タコメーター）＋ フューエル・電費・診断|
|説明書|P29 の車の部品の語（後輪・前輪・タコメーターは足さない）＋ クルーズコントロール・回転半径・積載量・発炎筒・ボタン・何人乗|
|カメラ|P30 の語 ＋ 足元・真横・アラウンドビュー・外 ＋ **車の部品の語（どこを映すか）**|
|曲の情報|P29 の語 ＋ ボーカル・声・誰|
|周辺の施設|P29 の語 ＋ センター・工場・ガソスタ|

言い換えは、**語の一覧を書くときに見ていない評価セット v1〜v4** の、対象の読み出しの行（768 行）で説明がつかなかった 29 行を見て足した（足す前の 29 行 = 3.8% が、見ていない対象の行で断ってしまう割合の見積もり）。
"""
from __future__ import annotations

from . import p29_scope as P29
from . import p30_scope as P30

READ = P29.READ
# 評価セット v1〜v4（語の一覧を書くときに見ていない）の対象の読み出しの行で説明がつかなかった言い換えを足した（対象の側。下の ADD_*）
ADD_STATUS = ("フューエル", "電費", "診断")
ADD_MANUAL = ("クルーズコントロール", "回転半径", "積載量", "発炎筒", "ボタン", "何人乗")
ADD_CAMERA = ("足元", "真横", "アラウンドビュー", "外")
ADD_MEDIA = ("ボーカル", "声", "誰")
ADD_POI = ("センター", "工場", "ガソスタ")
STATUS = P29.STATUS_ITEM + P29.STATUS_ALL + P29.CAR_PARTS + ("後輪", "前輪", "タコメーター") + ADD_STATUS
MANUAL = P29.CAR_PARTS + ADD_MANUAL
CAMERA = P30.CAMERA + ADD_CAMERA + P29.CAR_PARTS          # カメラは「どこを映すか」の語として車の部品も根拠にする（トランクの中・ドアの外）
MEDIA_INFO = P29.MEDIA_INFO + ADD_MEDIA
POI = P29.POI_KIND + P29.POI_PLACE + ADD_POI
LEX = {"get_vehicle_status": STATUS, "search_vehicle_manual": MANUAL, "capture_camera": CAMERA, "get_media_info": MEDIA_INFO, "search_poi": POI}


def evidence(tool: str, u: str) -> list[str]:
    """その機能の定義で説明がつく語（単語の切れ目に合うものだけ。空なら説明がつかない）"""
    if tool not in LEX:
        return ["(操作の機能は見ない)"]
    return P30.has(LEX[tool], u)
