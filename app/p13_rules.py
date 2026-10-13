# -*- coding: utf-8 -*-
"""P13: 決定的な規則を補う層。EdgeSLM_MultiHead_Multi の `dispatch.rules`（読むだけ）の上に重ねる。

設計: DoC/設計書_P13_評価セットを増やし重大な誤りを減らす.md の 5 節。

元の `apply_rules` を先に呼び、**元の規則が確認へ回さなかったときだけ**、下の 3 つを試す（元のファイルは変えない）。
どれも「履歴なし・ツールを実行しようとしている」ときだけ効く（雑談と判定された発話には触れない。元の規則と同じ範囲）。

|規則|中身|元の規則で抜けていた例（評価セット v2）|
|---|---|---|
|S1 言い直し|読点のあとに「いや」「やっぱ」が来て、その前にも中身がある（「〜して、いややっぱり〜」）|「流して、いややっぱり止めて」（元は逆向きの対の表に無い組を拾えない）|
|S2 取り消し|「忘れて」「今言った」「無かったことに」など。対象の語が無いとき|「今言ったのは忘れて」|
|S3 言いよどみ|助詞のあとに「、あの」「、えっと」で終わる|「冷房を、あの」|

    make_rules(rules) -> apply(row, predicted_action)      # `p8_gate.run(make_rules=...)` に渡す
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FILLER = r"(?:あのさ|あのー|あの|えっと|えーっと|えーと|えー|んーと|んー|うーんと|うーん|あー|ねえ|ごめん|ちょ|あ)"
# 読点・空白のあとに来る言い直しの印（「いらないや」のような語の中の「いや」は拾わない）
REDO = re.compile(r"[、。 　](?:あ、|あ |待って、)?(?:いいや|いや|やっぱ)")
LEAD = re.compile(rf"^(?:{FILLER}[、。 　ー]*)+")
TRAIL = re.compile(rf"[、 　]{FILLER}[ー…。、]*$|[、 　](?:その|そのー)[ー…。、]*$")
UNDO_MORE = ("忘れて", "今言った", "いま言った", "さっき言った", "取りやめ", "無かったことに", "無しで", "無しにして", "なしにして",
             "やらなくていい", "何もしないで", "言い間違え", "間違えた")
PARTICLES = ("は", "が", "を", "に", "で", "と", "の", "へ", "も")


@dataclass(frozen=True)
class Decision:
    action: str
    demote: bool
    rule: str | None


def redo_with_content(utterance: str) -> bool:
    """S1: 言い直しの印の前に、言いよどみ以外の中身があるか"""
    m = REDO.search(utterance)
    if not m:
        return False
    before = LEAD.sub("", utterance[:m.start()]).strip("、。 　")
    return len(before) >= 1


def trailing_hesitation(utterance: str) -> bool:
    """S3: 「〜を、あの」のように、助詞のあとの言いよどみで終わるか"""
    m = TRAIL.search(utterance)
    if not m:
        return False
    head = utterance[:m.start()].strip("、。 　")
    return bool(head) and head.endswith(PARTICLES)


def make_rules(rules):
    """元の `edge_router.dispatch.rules` を受け取り、補った apply(row, predicted_action) を返す"""
    def apply(row: dict, predicted_action: str):
        d = rules.apply_rules(row, predicted_action)
        if d.demote or d.rule is not None or not predicted_action.startswith("TOOL_") or row.get("history"):
            return d
        u = row.get("utterance", "")
        tool = predicted_action[len("TOOL_"):]
        if redo_with_content(u):
            return Decision(predicted_action, True, "S1_REDO")
        if any(m in u for m in UNDO_MORE) and not rules.has_target(u, tool):
            return Decision(predicted_action, True, "S2_UNDO")
        if trailing_hesitation(u):
            return Decision(predicted_action, True, "S3_HESITATION")
        return d
    return apply
