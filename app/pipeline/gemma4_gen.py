# -*- coding: utf-8 -*-
"""案 C の出力のかたまり: 最後の正規化＋「指定した位置」の選択＋出力層（lm_head）。

設計: DoC/設計書_C_まとめ読みグラフで分類と応答生成.md の 7 節。

まとめ読みのグラフ（T=128・KV なし）の最後に付ける。出力は 2 本:
- `h` [1,T,H]: 最後の正規化の出力（全位置。改善版 B と同じもの。端末の CPU で分類ヘッドに使う）
- `logits` [1,1,V]: **`sel` で指定した 1 か所だけ**の出力層の値（V=262,144。全位置ぶん出すと 1 回 134MB になるので出さない）

位置の指定 `sel` [1,T,1] は one-hot（選ぶ位置だけ 1、ほかは 0）。`h` を転置して `sel` を右から掛ける
（[1,H,T]×[1,T,1] → [1,H,1]）。act×act の行列積は右辺が 8bit に落ちる（C-8・`downgrade_actact_rhs`）が、
右辺を `sel`（0 と 1 だけ）にしてあるので 8bit でも正確。`h` の側は 16bit のまま。
（Gather に位置の番号を入れる作りは、グラフ入力が Gather に直接入る形で C-1 に当たるおそれがあるので避けた）

重み・名前は `gemma4_graph.DecodeChunk`（層なし・with_head）と同じ（継承）。QAT の値の書き込み `qat_inject --lm-head`
がそのまま使える。出力の tanh ソフトキャップ（30）は掛けない（単調なので 1 位は変わらない。貪欲な生成では同じ）。
`gemma4_graph.py`・`gemma4_prefill.py` は変更しない。
"""
from __future__ import annotations

import numpy as np
import torch

from app.pipeline.gemma4_graph import DecodeChunk


class PrefillHead(DecodeChunk):
    """入力 hidden_in [1,T,H]（層 34 の出力）・sel [1,T,1] → 出力 h [1,T,H]・logits [1,1,V]"""

    def __init__(self, qf):
        super().__init__(qf, [], with_head=True)

    def input_names(self) -> list[str]:
        return ["hidden_in", "sel"]

    def output_names(self) -> list[str]:
        return ["h", "logits"]

    def forward(self, x, sel):
        hn = self.final_norm(x)                                   # [1,T,H]
        hs = torch.matmul(hn.transpose(1, 2), sel)                # [1,H,T]×[1,T,1] → [1,H,1]
        logits = self.lm_head(hs.transpose(1, 2))                 # [1,1,V]
        return hn, logits


def one_hot(pos: list[int] | np.ndarray, T: int) -> np.ndarray:
    """[n,T,1]。行ごとに pos の位置だけ 1"""
    s = np.zeros((len(pos), T, 1), np.float32)
    for k, p in enumerate(pos):
        s[k, int(p), 0] = 1.0
    return s
