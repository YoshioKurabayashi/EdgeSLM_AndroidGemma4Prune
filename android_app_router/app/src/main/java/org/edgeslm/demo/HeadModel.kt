package org.edgeslm.demo

import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp

/**
 * 分類ヘッド（PC の `edge_router.heads.model.MultiHead`、mlp・幅 256）を端末の CPU で動かす。
 *
 * 設計: EdgeSLM_AndroidHead/DoC/設計書_P3_端末で分類するデモアプリ.md の 4.3 節。
 *
 * ```
 * action  : z  = (h_last − μ) / σ              → Linear → GELU → Linear → softmax(·/T)
 * argmode : z（同上）                           → Linear → GELU → Linear → softmax(·/T)
 * reject  : zr = ([h_last, h_mean] − μr) / σr  → Linear → GELU → Linear → softmax → [1]
 * ```
 *
 * 重みは PC の `app/g3_assets.py` が書き出した `head/head.json`（形・位置・クラス名・温度・閾値）と
 * `head/head.bin`（float32 リトルエンディアン）。書き出し時に numpy で同じ式を計算し、torch と一致することを確かめてある。
 *
 * 最終判定は PC の `app/p2_host_h.decisions()` と同じ:
 *  1. 棄却の得点 ≥ REJECT の閾値 → REJECT（確認へ）
 *  2. 1 位の確率 < τ（CHAT は τ_CHAT、道具は read / write で分ける）→ CLARIFY（聞き返し）
 *  3. それ以外 → 1 位の行き先
 * 規則（フィラー・インジェクションなど）と Dispatcher は入れない。**ヘッドだけの判定**。
 */
class HeadModel private constructor(
    private val meta: JSONObject,
    private val w: FloatArray,
) {
    val hidden: Int = meta.getInt("hidden")
    val rejectHidden: Int = meta.getInt("reject_hidden")
    val temperature: Float = meta.getDouble("temperature").toFloat()
    val actionClasses: List<String> = meta.getJSONArray("action_classes").let { a -> (0 until a.length()).map { a.getString(it) } }
    val argmodeClasses: List<String> = meta.getJSONArray("argmode_classes").let { a -> (0 until a.length()).map { a.getString(it) } }
    private val thr = meta.getJSONObject("thresholds")
    val tauWrite = thr.getDouble("TOOL_write").toFloat()
    val tauRead = thr.getDouble("TOOL_read").toFloat()
    val tauChat = thr.getDouble("CHAT").toFloat()
    val tauReject = thr.getDouble("REJECT").toFloat()
    private val toolEffect = meta.getJSONObject("tool_effect")
    private val argmodeTools: Set<String> = meta.optJSONArray("argmode_tools")?.let { a -> (0 until a.length()).map { a.getString(it) }.toSet() } ?: emptySet()
    private val labelsJa = meta.optJSONObject("labels_ja") ?: JSONObject()

    /** 表示用の日本語名（無ければそのまま） */
    fun label(name: String): String = labelsJa.optString(name, name)

    /** 1 回の推論の結果 */
    data class Prediction(
        val actionProbs: FloatArray,       // actionClasses の順
        val argmodeProbs: FloatArray,      // argmodeClasses の順
        val reject: Float,                 // 棄却の得点（意図が不明な確率）
        val top1: String,
        val conf: Float,
        val tau: Float,                    // 1 位に使った閾値
        val final: String,                 // REJECT / CLARIFY / 1 位の行き先
        val argmode: String?,              // エアコン・メディアのときだけ
        val ms: Double,
    )

    fun predict(hLast: FloatArray, hMean: FloatArray): Prediction {
        require(hLast.size == hidden && hMean.size == hidden) { "h の長さが違う: ${hLast.size}/${hMean.size}（期待 $hidden）" }
        val t0 = System.nanoTime()
        val z = standardize(hLast, "standardizer")
        val xr = FloatArray(rejectHidden).also { System.arraycopy(hLast, 0, it, 0, hidden); System.arraycopy(hMean, 0, it, hidden, hidden) }
        val zr = standardize(xr, "reject_standardizer")

        val actionProbs = softmax(mlp(z, "heads.action"), temperature)
        val argmodeProbs = softmax(mlp(z, "heads.argmode"), temperature)
        val reject = softmax(mlp(zr, "heads.reject"), 1f)[1]

        var best = 0
        for (i in actionProbs.indices) if (actionProbs[i] > actionProbs[best]) best = i
        val top1 = actionClasses[best]
        val conf = actionProbs[best]
        val tau = when {
            top1 == "CHAT" -> tauChat
            toolEffect.optString(top1.removePrefix("TOOL_"), "read") == "write" -> tauWrite
            else -> tauRead
        }
        val final = when {
            reject >= tauReject -> "REJECT"
            conf < tau -> "CLARIFY"
            else -> top1
        }
        var am: String? = null
        if (top1.removePrefix("TOOL_") in argmodeTools && argmodeProbs.isNotEmpty()) {
            var b = 0
            for (i in argmodeProbs.indices) if (argmodeProbs[i] > argmodeProbs[b]) b = i
            am = argmodeClasses[b]
        }
        val ms = (System.nanoTime() - t0) / 1e6
        return Prediction(actionProbs, argmodeProbs, reject, top1, conf, tau, final, am, ms)
    }

    // ---------------------------------------------------------------- 部品

    private fun tensor(name: String): Pair<Int, IntArray> {
        val t = meta.getJSONObject("tensors").getJSONObject(name)
        val shp = t.getJSONArray("shape").let { a -> IntArray(a.length()) { a.getInt(it) } }
        return t.getInt("offset") to shp
    }

    private fun standardize(x: FloatArray, prefix: String): FloatArray {
        val (om, _) = tensor("$prefix.mean")
        val (os, _) = tensor("$prefix.std")
        return FloatArray(x.size) { (x[it] - w[om + it]) / w[os + it] }
    }

    /** Linear(in→width) → GELU → Linear(width→out)。重みは PyTorch と同じ [out][in] の並び */
    private fun mlp(x: FloatArray, prefix: String): FloatArray {
        val hid = linear(x, "$prefix.0")
        for (i in hid.indices) hid[i] = gelu(hid[i])
        return linear(hid, "$prefix.2")
    }

    private fun linear(x: FloatArray, prefix: String): FloatArray {
        val (ow, shp) = tensor("$prefix.weight")
        val (ob, _) = tensor("$prefix.bias")
        val out = shp[0]
        val inn = shp[1]
        require(inn == x.size) { "$prefix: 入力 ${x.size} と重み $inn が合わない" }
        val y = FloatArray(out)
        for (o in 0 until out) {
            var s = 0.0
            val base = ow + o * inn
            for (i in 0 until inn) s += w[base + i] * x[i]
            y[o] = (s + w[ob + o]).toFloat()
        }
        return y
    }

    companion object {
        /** PyTorch の既定の GELU（erf 版）: x·Φ(x) = 0.5·x·(1 + erf(x/√2)) */
        fun gelu(x: Float): Float = (0.5 * x * (1.0 + erf(x / 1.4142135623730951))).toFloat()

        /** erf の近似（Abramowitz–Stegun 7.1.26、最大誤差 1.5e-7） */
        fun erf(x: Double): Double {
            val t = 1.0 / (1.0 + 0.3275911 * abs(x))
            val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
            return if (x >= 0) y else -y
        }

        fun softmax(v: FloatArray, temp: Float): FloatArray {
            var m = Float.NEGATIVE_INFINITY
            for (a in v) if (a / temp > m) m = a / temp
            val e = DoubleArray(v.size) { exp((v[it] / temp - m).toDouble()) }
            val s = e.sum()
            return FloatArray(v.size) { (e[it] / s).toFloat() }
        }

        /** `dir/head.json` と `dir/head.bin` を読む */
        fun load(dir: File): HeadModel {
            val meta = JSONObject(File(dir, "head.json").readText())
            val bytes = File(dir, "head.bin").readBytes()
            val n = meta.getInt("total_floats")
            require(bytes.size == n * 4) { "head.bin の大きさが違う: ${bytes.size}（期待 ${n * 4}）" }
            val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val w = FloatArray(n)
            fb.get(w)
            return HeadModel(meta, w)
        }
    }
}
