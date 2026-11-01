# -*- coding: utf-8 -*-
"""P36: 1 つの発話について、**いまの構成（P35）**の判定を返す参照の実装（Python）。アプリ v2（Java）は、これと同じ結果を返すように書く。

設計: DoC/設計書_P36_判定デモアプリv2.md の 3 節。順番は `p24_decide.decide` と同じ（部品だけ替える）:
BERT が断る → 規則（`p31_rules`）→ 棄却 → 確信度 →（雑談）→ 引数（`p35_arguments`。範囲外は値に `__scope__` → 検査が断る）→ 検査 → BERT のフラグ → 実行。

    parts = load_parts()
    d = decide(utterance, action_probs, classes, reject, thresholds, bert_prob, bert_hi, state, parts)    # p24_decide.decide
"""
from __future__ import annotations

import json

from .common import MULTI_ROOT, use_multi_library
from .p24_decide import BERT_LO, decide  # noqa: F401（呼ぶ側のために出す）


def load_parts() -> dict:
    """決定的な部品（規則 P31・引数 P35・検査・機能の表・車両の状態の見本）。EdgeSLM_MultiHead_Multi は読むだけ"""
    use_multi_library()
    from edge_router.dispatch import arguments, policy, rules
    from .p31_rules import make_rules
    from .p35_arguments import make_resolve
    tools = policy.load_tools()
    effect = {k: v.get("effect", "read") for k, v in tools.items()}
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    return {"rules": make_rules(rules), "resolve": make_resolve(arguments), "policy": policy, "effect": effect,
            "write": {k for k, v in effect.items() if v == "write"}, "state": state}
