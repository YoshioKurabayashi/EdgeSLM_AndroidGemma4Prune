# -*- coding: utf-8 -*-
"""P36: アプリ（Java）の語の一覧 `Lex.java` を、Python の語の一覧から機械的に書き出す（手で写すと写し間違いが出るため）。

設計: DoC/設計書_P36_判定デモアプリv2.md の 4 節。書き出した `Lex.java` は `android_app_router2/app/src/main/java/org/edgeslm/router/` に置く。

    python -m app.p36_lex        # work/p36/Lex.java
"""
from __future__ import annotations

from .common import PROJECT_ROOT
from . import p29_scope as P29, p30_scope as P30, p31_scope as P31, p33_scope as P33, p35_scope as P35
from . import p35_arguments as A35

OUT = PROJECT_ROOT / "work" / "p36" / "Lex.java"


def arr(name: str, xs, comment: str) -> str:
    xs = list(xs)
    for x in xs:
        assert '"' not in x and "\\" not in x, x
    return f"    /** {comment} */\n    static final String[] {name} = {{" + ", ".join('"' + x + '"' for x in xs) + "};\n"


def main() -> None:
    out = ["package org.edgeslm.router;\n\n/**\n * 語の一覧（P29〜P35。Python の `app/p29_scope.py`〜`app/p35_scope.py`・`p35_arguments.py` から機械的に書き出したもの。手で直さない。\n"
           " * 書き出し: `python -m app.p36_lex`。順番も Python と同じ。集合（FUNC・HIRA_NOUN・Q_TAIL）は並べ替えてある（属すかだけを見る）。\n */\nfinal class Lex {\n    private Lex() {}\n\n"]
    items = [("P30_CAR_PARTS", P30.CAR_PARTS, "P30 の車の部品（P29 ＋ 後輪・前輪・タコメーター）"), ("P29_CAR_PARTS", P29.CAR_PARTS, "P29 の車の部品"),
             ("STATUS_ITEM", P29.STATUS_ITEM, "車両の状態: 項目の語"), ("STATUS_ALL", P29.STATUS_ALL, "車両の状態: 全体の語"),
             ("P31_STATUS", P31.STATUS, "P31 車両の状態"), ("P31_MANUAL", P31.MANUAL, "P31 説明書"), ("P31_CAMERA", P31.CAMERA, "P31 カメラ"),
             ("P31_MEDIA_INFO", P31.MEDIA_INFO, "P31 曲の情報"), ("P31_POI", P31.POI, "P31 周辺の施設"), ("READ", P31.READ, "読み出しの機能"),
             ("PREFIX_1", sorted(P30.PREFIX_1, key="前後左右横曲歌".index), "1 文字でも単語の始まりに合えばよい語（P30）"),
             ("P33_REQ", P33.REQ, "P33 頼む動詞"), ("P33_PART", P33.PART, "P33 目的語の後ろの助詞"), ("P33_FUNC", sorted(P33.FUNC), "P33 機能語"),
             ("P33_HIRA_NOUN", sorted(P33.HIRA_NOUN), "P33 ひらがなの名詞"), ("P33_EXIST", P33.EXIST, "P33 ある・ない"), ("P33_HOURS", P33.HOURS, "P33 営業時間の語"),
             ("P33_Q_TAIL", sorted(P33.Q_TAIL), "P33 問いの語"), ("P33_POI_RET", P33.POI_RET, "P33 周辺の検索が返すもの"), ("P33_POI_SUFFIX", P33.POI_SUFFIX, "P33 施設の末尾 1 文字"),
             ("P33_MANUAL_RET", P33.MANUAL_RET, "P33 説明書が返すもの"), ("P33_MANUAL_SUFFIX", P33.MANUAL_SUFFIX, "P33 説明書の末尾 1 文字"),
             ("P35_STRONG_SUFFIX", P35.STRONG_SUFFIX, "P35 末尾 1 文字で施設"), ("P35_WEAK_SUFFIX", P35.WEAK_SUFFIX, "P35 独立した単語のときだけ施設"),
             ("P35_STATUS_RET", P35.STATUS_RET, "P35 車両の状態が返すもの"), ("P35_MINFO_RET", P35.MINFO_RET, "P35 曲の情報が返すもの"),
             ("P35_M_STOP", A35.M_STOP, "E3 メディアの止める語"), ("P35_M_PLAY", A35.M_PLAY, "E3 メディアの再生の語"),
             ("P35_H_STOP", A35.H_STOP, "E3 空調の止める語"), ("P35_H_ON", A35.H_ON, "E3 空調の操作の語")]
    out += [arr(*x) for x in items]
    out.append("}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(out), encoding="utf-8", newline="\n")
    print(f"保存しました: {OUT}（{len(items)} 個の一覧）")


if __name__ == "__main__":
    main()
