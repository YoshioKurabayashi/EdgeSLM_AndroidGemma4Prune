# -*- coding: utf-8 -*-
"""P24: 1 つの発話について、いまの構成の判定を返す参照の実装（Python）。アプリ（Java）は、これと同じ結果を返すように書く。

設計: DoC/設計書_P24_判定デモアプリ.md の 3 節。

評価の流れ（`p8_gate.run` の中の `flow`）は、評価セットをまとめて流す形で、1 つの発話を渡して判定を受け取る入口が無い。
ここに、同じ順番を 1 発話ずつ返す形で書く。**順番も中身も `p8_gate.flow` と同じ**（`p24_fixtures` が、評価セットで行ごとに突き合わせる）。

いまの構成: BERT（案 F・seed 0）→ ヘッド（1,584 行・mlp・seed 7）→ 規則（`p21_rules`）→ 棄却 → 確信度 → 引数（`p23_arguments`）→ 検査 → BERT のフラグ。

    d = decide(utterance, action_probs, classes, reject, thresholds, bert_prob, bert_hi, state, parts)
    d["outcome"]   実行／雑談／聞き返し／拒否
    d["stage"]     止めた段（規則・棄却ヘッド・確信度・引数が足りない・検査・BERT のフラグ・BERT が断る）。止めていなければ None
    d["tool"]・d["args"]   実行する機能と引数
"""
from __future__ import annotations

import json

from .common import MULTI_ROOT, use_multi_library

BERT_LO = 0.5          # これ以上でフラグ（`p8_gate.BERT_LO` と同じ）


def load_parts() -> dict:
    """決定的な部品（規則・引数・検査・機能の表・車両の状態の見本）。EdgeSLM_MultiHead_Multi は読むだけ"""
    use_multi_library()
    from edge_router.dispatch import arguments, policy, rules
    from .p21_rules import make_rules
    from .p23_arguments import make_resolve
    tools = policy.load_tools()
    effect = {k: v.get("effect", "read") for k, v in tools.items()}
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    return {"rules": make_rules(rules), "resolve": make_resolve(arguments), "policy": policy, "effect": effect,
            "write": {k for k, v in effect.items() if v == "write"}, "state": state}


def decide(utterance: str, action_probs, classes: list[str], reject: float, th: dict, bert_prob: float, bert_hi: float, state: dict, parts: dict) -> dict:
    """`p8_gate.flow` と同じ順: BERT が断る → 規則 → 棄却 → 確信度 → （雑談）→ 引数 → 検査 → BERT のフラグ → 実行"""
    out = {"outcome": None, "stage": None, "action": None, "tool": None, "args": None, "rule": None, "confidence": None, "reject": float(reject), "bert": float(bert_prob)}
    if bert_prob >= bert_hi:
        return {**out, "outcome": "拒否", "stage": "BERT が断る"}
    k = max(range(len(classes)), key=lambda i: action_probs[i])
    act, conf = classes[k], float(action_probs[k])
    d = parts["rules"]({"utterance": utterance, "history": [], "last_tool": None}, act)
    act = d.action
    out.update(action=act, confidence=conf, rule=d.rule)
    if d.demote:
        return {**out, "outcome": "聞き返し", "stage": "規則"}
    if reject >= th["REJECT"]:
        return {**out, "outcome": "聞き返し", "stage": "棄却ヘッド"}
    tau = th["CHAT"] if act == "CHAT" else th["TOOL_write" if parts["effect"].get(act[5:]) == "write" else "TOOL_read"]
    if conf < tau:
        return {**out, "outcome": "聞き返し", "stage": "確信度"}
    if act == "CHAT":
        if bert_prob >= BERT_LO:
            return {**out, "outcome": "聞き返し", "stage": "BERT のフラグ"}
        return {**out, "outcome": "雑談"}
    tool = act[5:]
    out["tool"] = tool
    ar = parts["resolve"](tool, None, utterance, state)
    if not ar.complete:
        return {**out, "outcome": "聞き返し", "stage": "引数が足りない"}
    out["args"] = ar.values
    v = parts["policy"].inspect(tool, ar.values, state)
    if not v.ok:
        return {**out, "outcome": "聞き返し" if v.missing else "拒否", "stage": "検査"}
    if bert_prob >= BERT_LO:
        return {**out, "outcome": "聞き返し", "stage": "BERT のフラグ"}
    return {**out, "outcome": "実行"}
