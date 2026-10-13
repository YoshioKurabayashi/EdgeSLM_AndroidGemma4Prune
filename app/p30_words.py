# -*- coding: utf-8 -*-
"""P30: 発話を単語に分ける（端末の BERT の入力作りと同じ分け方 = NFKC ＋ MeCab ＋ unidic-lite）。

設計: DoC/設計書_P30_語単位で一致させる.md の 3 節。

- fugashi と unidic-lite がある環境（EdgeSLM_MultiHead_Multi の venv）では、`BertJapaneseTokenizer` と同じやり方でその場で分ける
- 無い環境（VM の python3 など）では、`data/p30_words/words.tsv`（端末の `BertJa.java` で分けた表。`tools/p30_java/WordsDump.java`）を引く
- `BertJa.java` は P24 で、PC の分け方と 5,072 行すべてで同じことを確かめた（P30 でも同じ表で 0 行の食い違いを確かめた）。アプリでは `BertJa.words()` をそのまま使える

    words(u) -> list[str]            # NFKC 後の単語
    bounds(u) -> (norm, starts, ends) # NFKC 後の文字列と、単語の始まり・終わりの位置
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

from .common import PROJECT_ROOT

TABLE = PROJECT_ROOT / "data" / "p30_words" / "words.tsv"
_tagger = None
_table = None


def _load_table() -> dict:
    global _table
    if _table is None:
        _table = {}
        if TABLE.exists():
            for line in TABLE.read_text(encoding="utf-8").splitlines():
                u, _, ws = line.partition("\t")
                _table[u] = ws.split(" ") if ws else []
    return _table


def _fugashi():
    global _tagger
    if _tagger is None:
        try:
            import fugashi
            import unidic_lite
            _tagger = fugashi.GenericTagger(f'-d "{unidic_lite.DICDIR}"')
        except Exception:
            _tagger = False
    return _tagger


@lru_cache(maxsize=None)
def words(u: str) -> tuple:
    t = _load_table()
    if u in t:
        return tuple(t[u])
    tg = _fugashi()
    if tg:
        return tuple(w.surface for w in tg(unicodedata.normalize("NFKC", u)))
    raise SystemExit(f"単語に分けられない（fugashi が無く、{TABLE.name} にも無い）: {u}")


@lru_cache(maxsize=None)
def bounds(u: str):
    norm = unicodedata.normalize("NFKC", u)
    starts, ends, cur = set(), set(), 0
    for w in words(u):
        i = norm.find(w, cur)
        if i < 0:
            raise SystemExit(f"単語の位置が合わない: {u} / {w}")
        starts.add(i)
        ends.add(i + len(w))
        cur = i + len(w)
    return norm, frozenset(starts), frozenset(ends)


def aligned(term: str, u: str, stem: bool = False) -> bool:
    """term が、単語の切れ目から始まり、単語の切れ目で終わる位置にあるか（stem = True なら、始まりだけ合えばよい＝動詞の語幹）"""
    norm, starts, ends = bounds(u)
    t = unicodedata.normalize("NFKC", term)
    i = norm.find(t)
    while i >= 0:
        if i in starts and (stem or i + len(t) in ends):
            return True
        i = norm.find(t, i + 1)
    return False
