# -*- coding: utf-8 -*-
"""分類専用の「まとめ読み」グラフ（改善版 B）: T トークンを 1 回で読み、KV キャッシュを持たない。

設計: DoC/設計書_B_分類専用のまとめ読みグラフ.md の 3 節。

改善前（G2・G3）の端末のグラフは、文章生成用の `gemma4_graph.DecodeChunk`（1 回 1 トークン・KV キャッシュ付き）に
`h` の出力を足したものだった。分類では生成しないので、プロンプト（最大 64 トークン）を**1 回で**読めばよい。

- 重み・配線は `gemma4_graph.DecoderLayer` / `DecodeChunk` と同じ（**継承して forward だけ替える**。パラメタの名前が
  同じなので、QAT の値の書き込み `qat_inject` がそのまま使える）。`gemma4_graph.py` は変更しない（改善前のまま）
- 注意は T 個のトークンの中だけ（KV キャッシュなし）。マスク `attn_mask` [1,T,1,T] は因果（自分より後ろを見ない）
- 詰め物（右詰め、ID 0）は後ろにあるので、因果マスクだけで本物のトークンからは見えない。詰め物の位置の出力は捨てる
- KV を使い回す層（15〜34）は、同じ種類の最後の層（局所 13・全体 14）の K/V（このグラフの中で作ったもの）を使う
- 出力層（lm_head）は持たない。最後の正規化の出力 `h` [1,T,H] を全位置ぶん出し、端末の CPU で「最後の本物の位置」と
  「平均」を取る

入出力（`with_ple_proj=True` のかたまり）:
    inputs_embeds [1,T,H]・ple_tok [1,T,35,256]・cos_s/sin_s [1,T,1,256]・cos_g/sin_g [1,T,1,512]・attn_mask [1,T,1,T]
    → hidden_out [1,T,H]・ple_all [1,T,35,256]（次のかたまりへ）・present_k_{13,14}/present_v_{13,14}（後ろのかたまりが使う K/V）

名前は改善前の繋ぎの道具 `merge_chunks` の約束に合わせる（`present_k_13` → 後ろのかたまりの `shared_k_13`）。
"""
from __future__ import annotations

import torch

from app.pipeline import gemma4_graph as M
from app.pipeline.gemma4_graph import DecodeChunk, DecoderLayer, gelu_tanh, rope

MASK_NEG = M.MASK_NEG
PAD_ID = 0                 # Gemma の <pad>。右詰めの詰め物（因果マスクで本物のトークンからは見えない）


class PrefillLayer(DecoderLayer):
    """DecoderLayer と同じ重み・配線で、KV キャッシュを持たずに T トークンをまとめて計算する。"""

    def forward(self, h, ple_i, cos, sin, attn_mask, shared_kv=None):
        """h [1,T,H]・ple_i [1,T,256]・cos/sin [1,T,1,hd]・attn_mask [1,T,1,T]。
        戻り値: (h, k, v)。k・v は [1,1,T,hd]（KV を使い回す層に渡す）。使い回す層は shared_kv=(k, v) を受け取る"""
        T = h.shape[1]
        res = h
        x = self.input_layernorm(h)
        q = self.q_proj(x).view(1, T, self.nh, self.hd)          # 8 ヘッドを 8 行に並べる（MQA。改善前と同じ）
        q = rope(self.q_norm(q), cos, sin)                      # [1,T,8,hd]
        if shared_kv is None:
            k = rope(self.k_norm(self.k_proj(x).view(1, T, 1, self.hd)), cos, sin).transpose(1, 2)   # [1,1,T,hd]
            v = self.v_norm(self.v_proj(x).view(1, T, 1, self.hd)).transpose(1, 2)
        else:
            k, v = shared_kv
        s = torch.matmul(q, k.transpose(2, 3)) + attn_mask       # [1,T,8,T]。スケール 1.0
        n = v.shape[2]
        # 確率×V は形を変えた V に掛ける（改善前と同じ理由: 右辺だけを 8bit に落とせるように。C-8）
        p = torch.softmax(s, dim=-1).reshape(1, T * self.nh, n)
        ctx = torch.matmul(p, v.reshape(1, n, self.hd))          # [1,T*8,hd]
        h = res + self.post_attention_layernorm(self.o_proj(ctx.reshape(1, T, self.nh * self.hd)))
        res = h
        x = self.pre_feedforward_layernorm(h)
        h = res + self.post_feedforward_layernorm(self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x)))
        res = h
        g = gelu_tanh(self.per_layer_input_gate(h)) * ple_i
        h = res + self.post_per_layer_input_norm(self.per_layer_projection(g))
        h = h * self.layer_scalar
        return h, k, v


class PrefillChunk(DecodeChunk):
    """入口（埋め込みの入力と PLE の射影）＋指定した層（＋最後の正規化）を、T トークンまとめて計算する。

    - `with_ple_proj=True`: 入口を含む（入力 inputs_embeds・ple_tok）。False: 前のかたまりの hidden_in・ple_all を受ける
    - `with_norm=True`: 最後の正規化まで含め、出力を `h`（最後の正規化の出力）にする。lm_head は持たない
    - K/V: このかたまりで作った層 13・14 の K/V は `present_k_i`/`present_v_i` として出す（後ろのかたまりが使う）。
      このかたまりに無い K/V（層 15 以降だけのかたまり）は `shared_k_i`/`shared_v_i` を入力で受ける
      （改善前の `merge_chunks` がこの名前で繋ぐ。まとめ読みでは K/V はキャッシュではなく、かたまりの間の受け渡しだけ）
    """

    def __init__(self, qf, layers: list[int], with_ple_proj: bool = False, with_norm: bool = False):
        # 親の __init__ で DecoderLayer が作られるので、作ったあと PrefillLayer へ差し替える（重みは同じもの）
        super().__init__(qf, layers, with_ple_proj=with_ple_proj, with_head=False)
        for ly in self.layers:
            ly.__class__ = PrefillLayer
        self.with_norm = with_norm
        if with_norm:
            self.final_norm = M.RMSNorm(qf.vec(f"{M.PFX}norm.weight"), qf.eps)
        # 後ろのかたまりへ渡す K/V: このかたまりで作り、かつ KV を使い回す層の出どころになっているもの（13・14）
        srcs = {qf.layer_info(i)["kv_src"] for i in range(qf.cfg["num_hidden_layers"]) if qf.layer_info(i)["shared"]}
        self.kv_out = [i for i in self.kv_layers if i in srcs and not with_norm]

    def input_names(self) -> list[str]:
        names = (["inputs_embeds", "ple_tok"] if self.with_ple_proj else ["hidden_in", "ple_all"]) + \
                ["cos_s", "sin_s", "cos_g", "sin_g", "attn_mask"]
        for i in self.ext_kv:
            names += [f"shared_k_{i}", f"shared_v_{i}"]
        return names

    def output_names(self) -> list[str]:
        if self.with_norm:
            return ["h"]
        names = ["hidden_out"] + (["ple_all"] if self.with_ple_proj else [])
        for i in self.kv_out:
            names += [f"present_k_{i}", f"present_v_{i}"]
        return names

    def forward(self, x0, ple_in, cos_s, sin_s, cos_g, sin_g, attn_mask, *ext):
        T = x0.shape[1]
        if self.with_ple_proj:
            proj = torch.nn.functional.linear(x0, self.ple_proj) * self.ple_in_scale
            proj = proj.reshape(1, T, self.n_layers, self.ple_dim)
            ple_all = (self.ple_norm(proj) + ple_in) * self.ple_combine
        else:
            ple_all = ple_in
        h = x0
        kv = {i: (ext[2 * k], ext[2 * k + 1]) for k, i in enumerate(self.ext_kv)}
        for ly in self.layers:
            info = ly.info
            cos, sin = (cos_g, sin_g) if info["full"] else (cos_s, sin_s)
            ple_i = ple_all[:, :, info["i"], :]
            if info["shared"]:
                h, _, _ = ly(h, ple_i, cos, sin, attn_mask, shared_kv=kv[info["kv_src"]])
            else:
                h, k, v = ly(h, ple_i, cos, sin, attn_mask)
                kv[info["i"]] = (k, v)
        if self.with_norm:
            return (self.final_norm(h),)
        outs = [h] + ([ple_all] if self.with_ple_proj else [])
        for i in self.kv_out:
            outs += [kv[i][0], kv[i][1]]
        return tuple(outs)


def causal_mask(T: int) -> torch.Tensor:
    """[1,T,1,T]。自分と前だけ 0、後ろは MASK_NEG（−inf は量子化できないので −60。改善前と同じ）"""
    m = torch.full((T, T), MASK_NEG)
    m = torch.triu(m, diagonal=1)
    return m.reshape(1, T, 1, T)
