# -*- coding: utf-8 -*-
"""P23: 形の違う文に備えて、発話を**整えてから** `p22_arguments` に渡す層。`p22_arguments` は変えない。

設計: DoC/設計書_P22_引数の値を測って直す.md の 7 節。

P22 を、見ていない行（arg_eval_v1・240 行）で測ると、合っている 84%・聞き返す 5%・違う値で実行 10% だった。
違う値で実行した 25 行は、3 つの形に集まっていた。ここで、その形を「整える」ことで直す。

|整えること|例|整えた後|
|---|---|---|
|1 接続の語の後ろに読点を入れる（んで・ため・し・ければ・だら・ように・てから）|お腹がすいたんでファミレスに向かって|お腹がすいたんで、ファミレスに向かって|
|2 打ち消し・言い換えの前を落とす（A じゃなくて B・A はやめて B・A しないで B・A せずに B）|止めるんじゃなくて音を下げて|音を下げて|
|3 後ろに足した理由を落とす（依頼、理由）|デフロスターをつけて、前が見えない|デフロスターをつけて|
|4 漢数字・ひらがなの言い方をそろえる|温度を二十二度にして／風よわめで|温度を22度にして／風弱めで|

そのうえで、`p22_arguments` に足すもの:

|足すもの|例|
|---|---|
|空調で 2 つの値を言う（温度と風量）→ 分けて取り出し、1 回の実行にまとめる|温度を下げて風も強くして → 温度 24・風量 4|
|音量を倍に|音量を倍にして → 24|
|くだけたナビの言い方（名詞だけ＋お願い・よろしく・いって／家帰る／行き先、X）。**名詞が漢字・カタカナだけのとき**（「それでお願い」を目的地にしない）|東京駅お願い／空港いって|
|経由地を消す → 聞き返す（ツールに無い）|経由地だけ消して|
|ルートだけ変えて → 引き直し|目的地は変えずにルートだけ変えて|

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)
"""
from __future__ import annotations

import re

from .p22_arguments import PLACE_DEICTIC, VAGUE_PLACE, _hvac, make_resolve as make_p22, split_main

# ---------------------------------------------------------------- 整える
# 1. 接続の語（理由・条件）。後ろに読点が無ければ入れる
CONNECT = re.compile(r"((?<=[たいだるく])んで|(?<=[たるいの])ため(?:に)?|(?<=[いただ])し(?=[一-龥ァ-ヶ])|ければ|(?<=[きえけせてねへめれげべ])れば|(?<=[んい])だら|ように|てから|おかげで|せいで|もんで)(?![、。])")
# 2. 打ち消し・言い換え。前を落として、後ろだけにする
REPLACE = re.compile(r"^.*?(?:じゃなくて|ではなくて|じゃなく|ではなく|はやめて|をやめて|(?<=[ぁ-ん])ないで|(?<=[ぁ-ん])ずに)[、 　]*(?!ください|くれ|ほしい|欲しい|もらえ|いただ|おいて|ね|よ)(?=.{3,}$)")
# 3. 依頼の終わり方（これで終わる節は依頼。形容詞の「くて」は理由）
REQUEST_END = re.compile(r"(?:(?<!く)て|で|って|てよ|てね|てー|くれる|くれない|ほしい|ほしいな|ください|お願い|お願いします|頼む|たのむ|ますか|ませんか|よろしく|に|へ|まで)$")
KANJI_NUM = re.compile(r"(?<!もう)(?<!今)(?<!いま)([一二三四五六七八九十]+)(?=度|段階|段目|段|つ(?!ぎ)|目盛)|(?:(?<=音量を)|(?<=音量は)|(?<=ュームを)|(?<=風量を)|(?<=風量は))([一二三四五六七八九十]+)(?=に|まで|へ|で)")
DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
KANA = (("つよめ", "強め"), ("よわめ", "弱め"), ("つよく", "強く"), ("よわく", "弱く"), ("あげて", "上げて"), ("さげて", "下げて"))


def kanji_to_int(s: str) -> int:
    """十までの位だけ（二十二 → 22・十八 → 18・三 → 3）"""
    if "十" in s:
        a, b = s.split("十", 1)
        return (DIGIT.get(a, 1) if a else 1) * 10 + (DIGIT.get(b, 0) if b else 0)
    return DIGIT[s] if len(s) == 1 else int("".join(str(DIGIT[c]) for c in s))


def _num(m) -> str:
    s = m.group(1) or m.group(2)
    if s == "一" and m.group(1) and m.string[m.end():m.end() + 1] == "段":
        return s                                                     # 「一段強く」は数字にしない（1 段 = 強さの語として読む）
    return str(kanji_to_int(s))


def is_comment(part: str) -> bool:
    """後ろに足した理由・感想らしい節か（ひらがなで終わり、依頼の終わり方ではない）。「目的地は東京駅」のように名詞で終わる節は落とさない"""
    t = part.strip()
    return bool(re.search(r"[ぁ-ん]$", t)) and not REQUEST_END.search(t)


def normalize(u: str, replace: bool = True) -> str:
    v = KANJI_NUM.sub(_num, u)
    for a, b in KANA:
        v = v.replace(a, b)
    parts = [p for p in re.split(r"[、。]", v) if p.strip()]
    if len(parts) >= 2 and is_comment(parts[-1]) and any(REQUEST_END.search(p.strip()) for p in parts[:-1]):
        while len(parts) >= 2 and is_comment(parts[-1]):
            parts.pop()                                              # 3. 後ろに足した理由を落とす（打ち消しより先に見る）
        v = "、".join(parts)
    if replace:
        v = REPLACE.sub("", v)                                       # 2. 打ち消し・言い換えの前を落とす
    return CONNECT.sub(lambda m: m.group(1) + "、", v)


# ---------------------------------------------------------------- 足すもの
TWO = re.compile(r"(?<=[てで])(?=(?:設定)?温度|風量|風|冷房|暖房|窓|曇|デフロスタ|[0-9０-９]+度)")
NOUN = r"[一-龥ァ-ヶー々0-9０-９A-Za-z]{2,}"          # 漢字・カタカナ・数字だけ（「それ」「いつもの」を目的地にしない）
# 場所らしい終わり方。「お願い」「よろしく」だけで目的地にするのは、これで終わる名詞に限る（「全部お願い」「適当にお願い」を目的地にしない）
PLACE_END = (r"(?:駅|空港|港|店|屋|館|園|場|所|院|局|庁|城|寺|神社|橋|町|市|区|村|県|湖|山|浜|海岸|タワー|センター|ホテル|モール|ストア|ショップ|スタンド|パーキング|ランド|ドーム|ビル|ホール|カフェ"
             r"|コンビニ|スーパー|デパート|会社|自宅|家|実家|学校|大学|公園|銀行|インター|サービスエリア|ジャンクション)")
PLACE = rf"[一-龥ァ-ヶー々0-9０-９A-Za-z]*{PLACE_END}"
CASUAL = (re.compile(rf"^(?:行き先|目的地|次)[、はを]?\s*({PLACE})(?:な|ね|で|に)?$"),
          re.compile(rf"^({PLACE})(?:を|まで|へ|に)?(?:お願い|よろしく|頼む|たのむ)(?:します|ね|な)?$"),
          re.compile(rf"^({NOUN})(?:を|まで|へ|に)?(?:いって|行って|ナビして|ナビ)(?:ね|な)?$"),
          re.compile(r"^([一-龥ァ-ヶー々0-9０-９A-Za-z]+)(?:まで|へ)(?:お願い|よろしく|頼む|たのむ)(?:します|ね|な)?$"),
          re.compile(r"^(家|自宅|うち|実家|会社|職場)(?:に|へ|まで)?(?:帰る|帰りたい|戻る|帰ろう)$"),
          re.compile(rf"^({NOUN})(?:に)?寄って$"))
ROUTE_ONLY = re.compile(r"(?:ルート|経路|道順|行き方)(?:だけ|は|も)?(?:を|に)?(?:変|引き直|切り替|探)")
WAY_REMOVE = re.compile(r"経由地?.{0,4}(?:消|削除|外し|取り消|やめ|キャンセル|なし)")


def _hvac_two(u: str, state: dict, A):
    """温度と風量を 1 文で言う。分けて取り出し、まとめる（どれかが決まらなければ、やめる）"""
    segs = [s for s in TWO.split(split_main(u)[1]) if len(s.strip("、。 　")) >= 3]          # 主節の中だけで分ける
    if len(segs) < 2:
        return None
    values = {}
    for s in segs:
        x = _hvac(s, state, A)
        if x is None or not x.complete:
            return None
        values.update(x.values)
    return A.Arguments(values=values, notes=[f"P23: {len(segs)} つに分けて取り出し、まとめた"])


def make_resolve(A):
    p22 = make_p22(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        if tool not in ("control_hvac", "control_media", "set_navigation") or kw.get("recent_poi"):
            return p22(tool, argmode, utterance, state, **kw)
        u, plain = normalize(utterance), normalize(utterance, replace=False)
        out = one(tool, argmode, u, state, **kw)
        if not out.complete and plain != u:
            out = one(tool, argmode, plain, state, **kw)             # 打ち消しの前を落として決まらなければ、落とさずにもう一度（消音をやめて音を出して）
        return out

    def one(tool: str, argmode, u: str, state: dict, **kw):
        if tool == "control_hvac":
            return _hvac_two(u, state, A) or p22(tool, argmode, u, state, **kw)
        if tool == "control_media":
            vol = state.get("media", {}).get("volume")
            if "倍" in u and re.search(r"音量|ボリューム|音を", u) and vol is not None and not re.search(r"[0-9０-９]", u):
                return A.Arguments(values={"action": "volume_set", "volume": int(A.clamp("volume", vol * 2))}, notes=["P23: 倍 → いまの 2 倍"])
            return p22(tool, argmode, u, state, **kw)
        if WAY_REMOVE.search(u):
            return A.Arguments(values={"action": "set"}, missing=["destination"], notes=["P23: 経由地を消す指定はツールに無い → 確認へ"])
        out = p22(tool, argmode, u, state, **kw)
        if out.complete:
            return out
        for pat in CASUAL:
            m = pat.search(u.strip("、。 　"))
            if m and not PLACE_DEICTIC.search(m.group(1)) and not VAGUE_PLACE.search(m.group(1)) and not A.is_anaphoric_destination(m.group(1)):
                return A.Arguments(values={"action": "set", "destination": m.group(1)}, notes=[f"P23: くだけた言い方 → 目的地 {m.group(1)!r}"])
        if ROUTE_ONLY.search(u) and not PLACE_DEICTIC.search(u):
            return A.Arguments(values={"action": "reroute"}, notes=["P23: ルートを変える → reroute"])
        return out
    return resolve
