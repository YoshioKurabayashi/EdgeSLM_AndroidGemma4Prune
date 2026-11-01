# -*- coding: utf-8 -*-
"""P35: 「何を返すか」で絞る（P33）の 2 回目。車両の状態・曲の情報にも広げ、施設の語の末尾 1 文字の一致の不具合を直す。

設計: DoC/設計書_P35_重大の余裕と返すものの広げ.md の 3 節。`p33_scope` の取り出し方（`asked`）をそのまま使う（P33 のソースは変えない）。

利用者の決定（2026-10-14）:
- 車両の状態（get_vehicle_status）が返すもの = **燃料・航続距離・走行距離・空気圧・ドア・電池などの、車のいまの値**
- 曲の情報（get_media_info）が返すもの = **曲名・歌手・アルバム・長さ**（再生中の曲・番組のこと。作詞作曲・発売年・局名・番組名・プレイリスト名・次の曲も、
  これまでの評価セットで対象の行としてきたので、再生中のものの情報として含める）

直したもの:
- **施設の語の末尾 1 文字**（P33 の `POI_SUFFIX`）: 「相場」の「場」で施設とみなしていた（v10 で 1 行）。末尾の 1 文字だけで施設とみなすのは、屋・店・館・局・院・園・駅（施設の種類を表す字）に限る。
  場・所・地・器・機・宿 は、**独立した単語のとき**（「駐車 場」「洗車 場」「充電 器」）だけ
"""
from __future__ import annotations

from . import p31_scope as P31
from . import p33_scope as P33
from .p33_scope import asked

STRONG_SUFFIX = ("屋", "店", "館", "局", "院", "園", "駅")          # 語の末尾にあれば施設（花屋・飯屋・郵便局）
WEAK_SUFFIX = ("場", "所", "地", "器", "機", "宿")                  # 独立した単語のときだけ（駐車 場・充電 器）。「相場」「場所」の末尾では見ない
STATUS_RET = ("残り", "残量", "量", "値", "数値", "数字", "目盛り", "温度", "距離", "パーセント", "何割", "リットル", "キロ", "状況", "様子", "具合",
              "状態", "ところ", "燃費", "どれくらい", "いくつ", "何",
              # 学習の対象の行（車両の状態として学習している項目）で足した
              "圧", "情報", "設定", "速度", "スピード", "回転", "外気温", "気温", "サイズ", "規格", "期限", "日付", "全体", "今",
              # 評価セット v1〜v10 の対象の行で足した（足す前に断った 393 行中 3 行 = 0.8% が、見ていない言い方を断る割合の見積もり）
              "合計", "全部", "全て")
MINFO_RET = ("名前", "曲名", "名", "長さ", "時間", "何分", "残り", "年", "発売", "リリース", "情報", "者", "人", "何", "誰", "どこ",
             # 学習の対象の行（曲の情報として学習している項目）で足した
             "リスト", "状況", "値", "設定",
             # 評価セット v1〜v10 の対象の行で足した（足す前に断った 230 行中 4 行 = 1.7% が見積もり）
             "ジャンル", "題名")
RET = {"search_poi": (P31.POI + P33.POI_RET, STRONG_SUFFIX),
       "search_vehicle_manual": (P31.MANUAL + P33.MANUAL_RET, P33.MANUAL_SUFFIX),
       "get_vehicle_status": (P31.STATUS + STATUS_RET, ()),
       "get_media_info": (P31.MEDIA_INFO + MINFO_RET, ())}


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
    hit += [t for t in c if len(t) >= 2 and t[-1] in suffix]           # 花屋・郵便局
    hit += [t for t in c if t in suffix or t in WEAK_SUFFIX and suffix is STRONG_SUFFIX]   # 独立した「場」「店」
    return hit


def explains(tool: str, u: str):
    """返せるもので説明がつくか。(True/False, 理由)。取り出せなければ (True, None)"""
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
        ok = c[0].startswith(P33.HOURS)
        return ok, f"〜か: {c[0]}" + ("（営業時間）" if ok else "（返さない）")
    terms, suffix = RET[tool]
    if kind == "exist":
        hit = _match(P31.POI, STRONG_SUFFIX, c)
        return bool(hit), f"あるか: {''.join(c)}" + (f"（施設 {hit[0]}）" if hit else "（施設ではない）")
    hit = _match(terms, suffix, c)
    return bool(hit), f"聞いているもの: {''.join(c)}" + (f"（{hit[0]}）" if hit else "（返さない）")
