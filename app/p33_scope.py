# -*- coding: utf-8 -*-
"""P33: 機能が**何を返すか**で絞る（周辺の検索・説明書）。P31 の語の一覧（何について）の上に重ねる。

設計: DoC/設計書_P33_何を返すかで絞る.md の 3 節。

利用者の決定（2026-10-13）:
- 周辺の検索（search_poi）が返すもの = **施設の名前・場所・距離・営業時間**
- 説明書（search_vehicle_manual）が返すもの = **操作方法・仕様・意味**（費用や在庫は返さない）

P31 は「何についての依頼か」（店・車検などの語）だけを見ていた。ここでは「**何を聞いているか**」（教えて・調べて・探して の目的語）を取り出し、
それが返せるものの閉じた一覧で説明がつくかを見る。説明がつかなければ範囲外（断る）。開いた外側（値段・在庫・メニュー…）は列挙しない。

|取り出し方|例|聞いているもの|
|---|---|---|
|〜を／は／が／って ＋ 教えて・調べて・探して など|近くのガソリンの**値段**を教えて|値段|
|名詞 ＋ 教えて（助詞なし）|最寄りの**スタンド**教えて|スタンド|
|〜か ＋ 教えて・調べて（周辺の検索だけ）|この店に牛乳が**ある**か調べて|「ある」の主語（牛乳）|
|「いくら」|車検にいくらかかるか教えて|お金の額（どちらの機能も返さない）|
|頼む動詞が無い「〜は（何・いくつ…）」（P34 の前に足した）|この駐車場の**料金**は|料金（「〜はどこ」は場所を聞いているので通す）|

取り出せないとき（「近くにコンビニある」など）は P31 のまま（何についての語で判断）。
周辺の検索は、探す条件（安い・24 時間・〜してくれる店・犬と泊まれる）を付けて**施設を探す**依頼は通す（返すのは施設の名前・場所）。学習の対象の行（最安のスタンド・鍵を作ってくれる店）と同じ扱い。

    asked(tool, u) -> (kind, chunk)   # 聞いているもの（取り出せなければ None）
    explains(tool, u) -> (ok, why)
"""
from __future__ import annotations

import re

from . import p31_scope as P31
from .p30_words import words

REQ = ("教え", "調べ", "探し", "探す", "検索", "知り", "知る", "見せ", "出し", "リストアップ", "確認", "説明", "比べ", "チェック", "表示")
PART = ("を", "は", "が", "って", "も", "について", "とか")
FUNC = set("の に で へ と が は を も や か な だ ね よ さ わ て た る い ん 、 。 ？ ! ? ・".split()) | {
    "から", "まで", "より", "って", "です", "ます", "でし", "ください", "下さい", "ほしい", "欲しい", "たい", "ない", "なく", "だけ", "など", "について", "とか", "けど", "ので", "のに"}
HIRA_NOUN = {"お", "ご", "おすすめ", "オススメ", "ところ", "とこ", "もの", "こと", "いくつ", "どこ", "なに", "何"}
VERBISH = re.compile(r"[ぁ-ん]*(?:る|た|て|だ|い|ない|れる|せる|たい|ます|う|く|す|ず|っ|ぬ|れ|ろ|ば)$")
EXIST = ("ある", "ない", "あっ", "あり", "無い", "無かっ", "なかっ", "あれ")
Q_TAIL = {"何", "なに", "どこ", "いくつ", "どれ", "くらい", "ぐらい", "どの", "どのくらい", "どれくらい", "何時", "いつ", "誰", "です", "でしょう", "か", "の", "？", "?", "。", "、", "ある", "あり", "ます"}
HOURS = ("開い", "開く", "開け", "やっ", "営業", "空い", "閉まっ", "閉まる", "開店", "閉店")

POI_RET = ("ところ", "とこ", "場所", "名前", "住所", "位置", "所在地", "距離", "道のり", "何キロ", "営業", "時間", "何時", "開店", "閉店", "定休", "休み", "一覧", "リスト", "候補", "どこ",
           "ショップ", "医者", "交番", "警察")   # 施設の言い方（v1〜v9 の対象の行で足した）
POI_SUFFIX = ("屋", "店", "場", "館", "所", "局", "院", "園", "駅", "器", "機", "地", "宿")
MANUAL_RET = ("方", "方法", "手順", "仕方", "対処", "注意", "意味", "説明", "見方", "マーク", "ランプ", "警告", "表示", "容量", "規格", "サイズ", "型番", "型式", "種類",
              "位置", "場所", "範囲", "上限", "下限", "限度", "推奨", "値", "寿命", "目安", "時期", "期間", "条件", "項目", "内容", "燃費", "回転", "温度", "定員",
              "重さ", "重量", "寸法", "長さ", "幅", "高さ", "数", "性能", "仕様", "スペック", "機能", "設定", "モード", "理由", "原因", "違い", "使い分け", "保証",
              "頻度", "間隔", "タイミング", "基準", "適正", "規定", "交換", "点検", "整備", "操作", "使用", "取り扱い", "取扱", "何", "なに", "どこ", "いくつ", "メンテナンス", "コツ", "ポイント",
              "トルク", "粘度", "仕組み", "構造", "処置", "場面",   # v1〜v9 の対象の行で足した
              "油種", "対応", "記号", "スイッチ", "席")   # 「〜は何」の形を足したとき、v1〜v9 の対象の行で足した
MANUAL_SUFFIX = ("方", "法", "値", "点", "数", "量", "灯", "液", "時")
RET = {"search_poi": (P31.POI + POI_RET, POI_SUFFIX), "search_vehicle_manual": (P31.MANUAL + MANUAL_RET, MANUAL_SUFFIX)}


def nounish(t: str) -> bool:
    if t in HIRA_NOUN:
        return True
    if t in FUNC or not t.strip():
        return False
    if re.fullmatch(r"[ぁ-ん]+", t):
        return False
    return not (re.search(r"[ぁ-ん]$", t) and VERBISH.search(t))


def chunk_before(ws: list, j: int) -> list:
    """ws[j] で終わる、名詞らしい語の並び"""
    k = j
    while k >= 0 and nounish(ws[k]):
        k -= 1
    return ws[k + 1:j + 1]


def asked(tool: str, u: str):
    ws = list(words(u))
    if "いくら" in ws:
        return ("money", ["いくら"])
    for v, t in enumerate(ws):
        if not t.startswith(REQ) or v == 0:
            continue
        p = v - 1
        if ws[p] in ("お", "ご") and p > 0:          # お教えください
            p -= 1
        if ws[p] == "か":
            if tool != "search_poi":
                return None
            q = p - 1
            while q >= 0 and ws[q] in ("の", "ん", "て", "いる", "か"):
                q -= 1
            if q < 0:
                return None
            pred = ws[q]
            if pred.startswith(EXIST):
                ga = [i for i in range(q - 1, -1, -1) if ws[i] in ("が", "は", "も", "って")]
                s = ga[0] - 1 if ga else q - 1     # 「〜が ある」の主語（無ければ「ある」の直前）
                return ("exist", chunk_before(ws, s)) if s >= 0 else None
            return ("yn", [pred])
        if ws[p] in PART:
            p -= 1
        if p >= 0:
            c = chunk_before(ws, p)
            if c:
                return ("noun", c)
        return None
    # 頼む動詞が無い「〜は（何・どこ・いくつ…）」の形: 「は・って」の後が問いの語だけなら、その前が聞いているもの
    for p in range(len(ws) - 1, 0, -1):
        if ws[p] in ("は", "って"):
            if all(t in Q_TAIL for t in ws[p + 1:]):
                if "どこ" in ws[p + 1:]:
                    return ("where", ws[p + 1:])          # 場所を聞いている（周辺の検索 = 場所・説明書 = 部品の位置。どちらも返す）
                c = chunk_before(ws, p - 1)
                return ("noun", c) if c else None
            break
    return None


def _match(terms, suffix, c: list) -> list:
    s = "".join(c)
    starts, pos = set(), 0
    for t in c:
        starts.add(pos)
        pos += len(t)
    hit = []
    for w in terms:
        i = s.find(w)
        while i >= 0:
            if i in starts and (len(w) >= 2 or (i + 1) in starts | {len(s)}):
                hit.append(w)
                break
            i = s.find(w, i + 1)
    hit += [t for t in c if t.endswith(suffix) and len(t) >= 1 and t[-1] in suffix]
    return hit


def explains(tool: str, u: str):
    """返せるもので説明がつくか。(True/False, 理由)。取り出せなければ (True, None)＝P31 のまま"""
    if tool not in RET:
        return True, None
    a = asked(tool, u)
    if a is None:
        return True, None
    kind, c = a
    if kind == "money":
        return False, "お金の額（返さない）"
    if kind == "where":
        return True, "どこ（場所・位置は返す）"
    if kind == "yn":
        ok = c[0].startswith(HOURS)
        return ok, f"〜か: {c[0]}" + ("（営業時間）" if ok else "（返さない）")
    terms, suffix = RET[tool]
    if kind == "exist":
        hit = _match(P31.POI, POI_SUFFIX, c)
        return bool(hit), f"あるか: {''.join(c)}" + (f"（施設 {hit[0]}）" if hit else "（施設ではない）")
    hit = _match(terms, suffix, c)
    return bool(hit), f"聞いているもの: {''.join(c)}" + (f"（{hit[0]}）" if hit else "（返さない）")
