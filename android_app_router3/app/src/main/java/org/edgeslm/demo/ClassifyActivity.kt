package org.edgeslm.demo

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

/**
 * 分類のデモ画面（Gemma 4 E2B・**層を減らした**分類専用のまとめ読みグラフ。お試し EdgeSLM_AndroidGemma4Prune P3・2026-10-02）。
 * 改善版 B のアプリ（EdgeSLM_AndroidGemma4Head/android_app_fast）の同名の画面を複製し、既定のモデルを層 17 のグラフに、見出しを
 * samples.json の `title`・`subtitle`・`model_label` で差し替えられるようにした（判定の流れ・ヘッドの計算は改善版 B と同じ）。
 * 以下は複製元の説明:
 * 分類のデモ画面（Gemma 4 E2B・改善版 B: 分類専用のまとめ読みグラフ）。**ヘッド付きの LLM が、スマホの NPU だけで発話を分類する**ところを見せる。
 *
 * 設計: EdgeSLM_AndroidGemma4Head/DoC/設計書_G2_hの出力を持つcontext_binary.md の 3.5 節。
 * 前段 EdgeSLM_AndroidHead の同名の画面（0.5B 版）を複製し、トークナイザ・テンプレート・入力（表）を Gemma 4 に替えた。
 *
 * ## 流れ（すべて端末の中）
 *
 * 1. 発話を F2 の文（状態＋履歴なし＋発話）にし、Gemma 4 のチャットテンプレート（system なし）で包む
 * 2. [Gemma4Tokenizer] で ID 列にする
 * 3. `generate_kv`（`collect_h`・`pos0_kv`）で **NPU（HTP）に 1 トークンずつ流し**、`h` の最後と平均を受け取る
 * 4. [HeadModel] で行き先・ArgMode・棄却を計算し、閾値で最終判定を出す
 *
 * ## 画面の作り
 *
 * レイアウト XML を持たず、コードで組む（デモアプリの ChatActivity と同じ）。
 * モデル（772MB）は画面を開いたときに 1 回だけ開き、開いたまま分類を繰り返す。
 * 推論は別スレッドで行い、その間はボタンを押せなくする。
 *
 * ## テスト・録画用の自動実行
 *
 * `am start -n org.edgeslm.gemma4prune/org.edgeslm.demo.ClassifyActivity --es prompts "発話1|発話2"` で起動すると、準備ができたあと
 * 発話を順に入力欄へ入れて分類する（手で押したときと同じ経路）。1 件ごとにログ `auto_classify item`、
 * 終わると `auto_classify done` を出し、結果を `out/classify_auto.jsonl` に書く。PC の `app/g3_record_demo.py` が使う。
 */
class ClassifyActivity : Activity() {

    // ---- 画面の部品 ----
    private lateinit var statusView: TextView
    private lateinit var inputEdit: EditText
    private lateinit var classifyButton: Button
    private lateinit var motionButtons: List<Button>
    private lateinit var resultBox: LinearLayout
    private lateinit var finalView: TextView
    private lateinit var detailView: TextView
    private lateinit var barsBox: LinearLayout
    private lateinit var timingView: TextView
    private lateinit var compareView: TextView
    private val sampleButtons = ArrayList<Button>()

    // ---- 状態 ----
    @Volatile private var handle: Long = 0L
    @Volatile private var tokenizer: Gemma4Tokenizer? = null
    @Volatile private var head: HeadModel? = null
    @Volatile private var busy = true
    private var motion = "stopped"
    private var samples: List<Sample> = emptyList()
    private var chatPrefix = DEFAULT_PREFIX
    private var chatSuffix = DEFAULT_SUFFIX
    private var jobExtra: JSONObject? = null
    private var strict = false
    private var loadMs = 0L
    // テスト・録画用の自動実行（インテント引数 `--es prompts "発話1|発話2|…"`。渡さなければ何もしない）
    private val autoQueue = ArrayDeque<String>()
    private var autoActive = false
    private var autoIndex = 0
    private val ui = Handler(Looper.getMainLooper())

    /** 比べる相手の表示（samples.json の ref_label・ref_short・note。無ければ既定） */
    private var refLabel = DEFAULT_REF_LABEL
    private var refShort = DEFAULT_REF_SHORT
    private var noteText = DEFAULT_NOTE
    /** 見出し（samples.json の title・subtitle・model_label。PC の app/p3_assets.py が書く。無ければ層 17 の既定） */
    private var titleText = DEFAULT_TITLE
    private var subtitleText = DEFAULT_SUBTITLE
    private var modelLabel = DEFAULT_MODEL_LABEL

    /** 例文（PC の `app/g3_assets.py` が書き出す `head/samples.json`） */
    data class Sample(
        val id: String, val utterance: String, val motion: String, val expected: String,
        val pcFinal: String, val pcTop1: String, val pcConf: Double, val pcIds: IntArray?,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        ensureDirs()
        loadSamples()
        intent?.getStringExtra("prompts")?.split("|")?.map { it.trim() }?.filter { it.isNotEmpty() }?.let {
            autoQueue.addAll(it)
            Log.i(TAG, "auto_classify queued ${it.size} prompts")
        }
        runCatching { File(getExternalFilesDir(null), "out/classify_auto.jsonl").delete() }
        buildUi()
        openModel()
    }

    override fun onDestroy() {
        val h = handle
        handle = 0L
        if (h != 0L) runCatching { NativeBridge.nativeDestroy(h) }
        super.onDestroy()
    }

    // ================================================================= 準備

    /** サブディレクトリはアプリが作る（adb で作ると所有者が shell になり読めない。デモアプリの注記） */
    private fun ensureDirs() {
        val b = getExternalFilesDir(null) ?: return
        listOf("models", "models/ops", "jobs", "inputs", "out", "head", "tokenizer").forEach {
            val d = File(b, it)
            if (!d.isDirectory && !d.mkdirs()) Log.w(TAG, "mkdir failed: ${d.absolutePath}")
        }
        Log.i(TAG, "externalFilesDir=${b.absolutePath}")
    }

    private fun loadSamples() {
        try {
            val f = File(getExternalFilesDir(null), "head/samples.json")
            if (!f.exists()) return
            val o = JSONObject(f.readText())
            chatPrefix = o.optString("chat_prefix", DEFAULT_PREFIX)
            chatSuffix = o.optString("chat_suffix", DEFAULT_SUFFIX)
            jobExtra = o.optJSONObject("job")      // 端末の表の場所など（PC の app/g3_assets.py が書く）
            refLabel = o.optString("ref_label", DEFAULT_REF_LABEL)
            refShort = o.optString("ref_short", DEFAULT_REF_SHORT)
            noteText = o.optString("note", DEFAULT_NOTE)
            titleText = o.optString("title", DEFAULT_TITLE)
            subtitleText = o.optString("subtitle", DEFAULT_SUBTITLE)
            modelLabel = o.optString("model_label", DEFAULT_MODEL_LABEL)
            val arr = o.getJSONArray("samples")
            samples = (0 until arr.length()).map { i ->
                val s = arr.getJSONObject(i)
                val ids = s.optJSONArray("ids")?.let { a -> IntArray(a.length()) { a.getInt(it) } }
                Sample(s.getString("id"), s.getString("utterance"), s.optString("motion", "stopped"),
                       s.optString("expected"), s.optString("pc_final"), s.optString("pc_top1"),
                       s.optDouble("pc_conf", 0.0), ids)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "samples.json ignored: ${t.message}")
        }
    }

    private fun jobBase(): JSONObject = JSONObject().apply {
        put("job_id", "classify")
        // ★改善版 B（2026-09-28）: 分類専用のまとめ読みグラフで、プロンプトを 1 回の推論で読む（job.cpp の classify_prefill）。
        //   改善前（android_app/）は generate_kv で 1 トークンずつ流していた（1 回の分類 約 3.6 秒 → 約 0.1 秒）
        put("task", "classify_prefill")
        put("model", MODEL)
        // Gemma 4 の入力（EdgeSLM_AndroidGemma4 の device_chat.job_common と同じ）: 埋め込み 2bit・PLE 4bit・位置の表 4 本。
        // 値は head/samples.json の "job" があればそちらで上書きする（PC の tables.json から作る）
        put("embed_table", "models/ops/embed_q2.bin")
        put("embed_dtype", "q2row")
        put("embed_scales", "models/ops/embed_scale.bin")
        put("embed_scale", 39.191835884530846)
        put("ple_table", "models/ops/ple_q4.bin")
        put("ple_scales", "models/ops/ple_scale.bin")
        put("ple_scale", 16.0)
        put("rope_tables", JSONObject().apply {
            put("cos_s", "models/ops/rope_cos_s.bin"); put("sin_s", "models/ops/rope_sin_s.bin")
            put("cos_g", "models/ops/rope_cos_g.bin"); put("sin_g", "models/ops/rope_sin_g.bin")
        })
        put("mask_neg", -60.0)
        put("pad_id", 0)
        jobExtra?.let { x -> for (k in x.keys()) put(k, x.get(k)) }
        put("ep", JSONObject().apply {
            put("provider", "QNN")
            put("backend", "libQnnHtp.so")
            put("options", JSONObject().apply {
                put("offload_graph_io_quantization", "0")
                put("qnn_context_priority", "high")
            })
            put("session", JSONObject().apply {
                put("intra_op_num_threads", 1)
                put("log_severity_level", 3)
            })
        })
    }

    private fun withPaths(job: JSONObject): String = JsonInject.addRuntimePaths(
        job.toString(), baseDir = getExternalFilesDir(null)!!.absolutePath,
        nativeLibDir = applicationInfo.nativeLibraryDir, overridesJson = null)

    private fun openModel() {
        setBusy(true)
        status("モデルを読み込み中…（数秒）", SUB)
        thread(name = "head-open") {
            try {
                val base = getExternalFilesDir(null)!!
                tokenizer = Gemma4Tokenizer.load(File(base, "tokenizer"))
                head = HeadModel.load(File(base, "head"))
                val t0 = System.currentTimeMillis()
                val h = NativeBridge.nativeCreate(withPaths(jobBase()))
                loadMs = System.currentTimeMillis() - t0
                val d = JSONObject(NativeBridge.nativeDescribe(h))
                if (h == 0L) {
                    ui.post {
                        status("モデルを開けませんでした。デモアプリなど、ほかのアプリでモデルを開いていないか確認してください。\n" +
                               d.optString("error"), ERR)
                    }
                    return@thread
                }
                handle = h
                strict = d.optBoolean("strict_no_cpu_fallback", false)
                val backend = d.optString("backend", d.optString("placement", "?"))
                Log.i(TAG, "ready load_ms=$loadMs strict=$strict backend=$backend")
                ui.post {
                    status("準備完了 · $modelLabel · " +
                           (if (strict) "全ノード NPU（HTP）" else "一部 CPU の可能性") +
                           " · 読み込み ${"%.1f".format(loadMs / 1000.0)} 秒", OK)
                    setBusy(false)
                    Log.i(TAG, "classify ready load_ms=$loadMs strict=$strict")
                    if (autoQueue.isNotEmpty()) { autoActive = true; ui.postDelayed({ runNextAuto() }, 1500) }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "open failed", t)
                ui.post { status("準備に失敗しました: ${t.javaClass.simpleName}: ${t.message}\n" +
                                 "資材（scripts/g3_deploy.ps1）を push したか確認してください。", ERR) }
            }
        }
    }

    // ================================================================= 分類

    /** F2 の文（`edge_router.data.build.render(row, "F2")` と同じ。履歴は常に「なし」） */
    private fun renderF2(utterance: String, motion: String): String =
        "[状態] 走行:$motion 権限:allowed\n[履歴] なし\n[発話] $utterance"

    private fun classify(utterance: String, sample: Sample?) {
        val tok = tokenizer
        val hd = head
        val h = handle
        if (busy || tok == null || hd == null || h == 0L || utterance.isBlank()) return
        hideKeyboard()
        setBusy(true)
        status("NPU で計算中…", SUB)
        showPending(utterance.trim())
        val text = chatPrefix + renderF2(utterance.trim(), motion) + chatSuffix
        thread(name = "head-classify") {
            val t0 = System.nanoTime()
            val ids = tok.encode(text)
            val tokMs = (System.nanoTime() - t0) / 1e6
            val job = jobBase().apply { put("prompt_ids", JSONArray(ids.toList())) }
            val raw = NativeBridge.nativeRunJob(h, withPaths(job))
            val r = JSONObject(raw)
            runCatching { File(getExternalFilesDir(null), "out/classify_last.json").writeText(raw) }
            if (!r.optBoolean("ok", false) || !r.has("h_last")) {
                ui.post {
                    status("推論に失敗しました: ${r.optString("error")}", ERR)
                    finalView.text = "推論に失敗しました"; finalView.setTextColor(ERR); resultBox.alpha = 1f
                    setBusy(false)
                }
                return@thread
            }
            val hl = toFloats(r.getJSONArray("h_last"))
            val hm = toFloats(r.getJSONArray("h_mean"))
            val p = hd.predict(hl, hm)
            val idsMatch = sample?.pcIds?.let { it.contentEquals(ids) }
            Log.i(TAG, "classify n=${ids.size} final=${p.final} top1=${p.top1} conf=${p.conf} reject=${p.reject} " +
                       "npu_ms=${r.optDouble("total_ms")} ids_match=$idsMatch")
            val rec = JSONObject().apply {
                put("utterance", utterance); put("motion", motion); put("n_tokens", ids.size)
                put("final", p.final); put("top1", p.top1); put("conf", p.conf.toDouble()); put("reject", p.reject.toDouble())
                put("argmode", p.argmode ?: JSONObject.NULL)
                put("npu_total_ms", r.optDouble("total_ms")); put("run_ms", r.optDouble("run_ms")); put("prep_ms", r.optDouble("prep_ms"))
                put("head_ms", p.ms); put("tokenize_ms", tokMs)
                put("sample_id", sample?.id ?: JSONObject.NULL); put("pc_final", sample?.pcFinal ?: JSONObject.NULL)
                put("ids_match", idsMatch ?: JSONObject.NULL)
            }
            runCatching { File(getExternalFilesDir(null), "out/classify_auto.jsonl").appendText(rec.toString() + "\n") }
            ui.post {
                showResult(utterance, p, ids.size, tokMs, r, sample, idsMatch)
                status("準備完了 · 別の発話も試せます", OK)
                setBusy(false)
                if (autoActive) {
                    Log.i(TAG, "auto_classify item ${autoIndex} ${rec}")
                    if (autoQueue.isNotEmpty()) ui.postDelayed({ runNextAuto() }, AUTO_GAP_MS)
                    else { autoActive = false; Log.i(TAG, "auto_classify done n=${autoIndex}") }
                }
            }
        }
    }

    /** 自動実行の 1 件。手で入力して「分類」を押したときと同じ経路（例文と同じ文なら PC の判定も出す） */
    private fun runNextAuto() {
        val u = autoQueue.removeFirstOrNull() ?: return
        autoIndex += 1
        val s = samples.firstOrNull { it.utterance == u }
        if (s != null) { motion = s.motion; refreshMotion() }
        inputEdit.setText(u)
        classify(u, s)
    }

    /**
     * 計算中の表示。前の結果を残したままだと、入力欄（次の発話）と結果（前の発話）が食い違って見える
     * （2026-09-27 の 1 回目の録画）。結果の欄を「計算中」に切り替え、薄くする。
     */
    private fun showPending(utt: String) {
        resultBox.visibility = View.VISIBLE
        resultBox.alpha = 0.55f
        finalView.text = "NPU で計算中…"
        finalView.setTextColor(SUB)
        detailView.text = "「$utt」"
        barsBox.removeAllViews()
        timingView.text = ""
        compareView.visibility = View.GONE
    }

    private fun toFloats(a: JSONArray) = FloatArray(a.length()) { a.getDouble(it).toFloat() }

    private fun showResult(utt: String, p: HeadModel.Prediction, nTok: Int, tokMs: Double, r: JSONObject,
                           sample: Sample?, idsMatch: Boolean?) {
        val hd = head ?: return
        resultBox.visibility = View.VISIBLE
        resultBox.alpha = 1f
        val (title, color) = when (p.final) {
            "REJECT" -> "確認へ回す（意図が不明）" to WARN
            "CLARIFY" -> "聞き返す（確信度が低い）" to WARN
            "CHAT" -> "雑談として応答" to ACCENT
            else -> "${hd.label(p.final)} を実行" to OK
        }
        finalView.text = title
        finalView.setTextColor(color)

        val sb = StringBuilder()
        sb.append("「$utt」\n")
        sb.append("1 位: ${hd.label(p.top1)}  確信度 ${"%.2f".format(p.conf)}（閾値 ${"%.2f".format(p.tau)}）\n")
        if (p.argmode != null) sb.append("ArgMode: ${hd.label(p.argmode)}\n")
        sb.append("棄却の得点: ${"%.3f".format(p.reject)}（閾値 ${"%.3f".format(hd.tauReject)}）")
        detailView.text = sb.toString()

        barsBox.removeAllViews()
        val order = p.actionProbs.indices.sortedByDescending { p.actionProbs[it] }.take(3)
        for (i in order) barsBox.addView(bar(hd.label(hd.actionClasses[i]), p.actionProbs[i]))

        // 改善版: プロンプトを 1 回の推論で読む。NPU は ORT の Run、準備は表引き・入力の組み立て（CPU）
        val npuMs = r.optDouble("run_ms", 0.0)
        val prepMs = r.optDouble("prep_ms", 0.0)
        timingView.text = "トークン $nTok 個を 1 回で · NPU ${"%.0f".format(npuMs)} ms · 準備 ${"%.0f".format(prepMs)} ms" +
                          " · ヘッド ${"%.1f".format(p.ms)} ms · トークン化 ${"%.0f".format(tokMs)} ms"

        if (sample != null) {
            val same = sample.pcFinal == p.final
            compareView.visibility = View.VISIBLE
            // ★改善版（2026-09-28）: 比べる相手の名前は samples.json の ref_label（PC の app/g3_assets.py が書く）。
            //   端末の h で学習したヘッド（fast_all など）では、相手は「PC の基準」ではなく一括収集（B4）での端末の判定
            compareView.text = "$refLabel: ${hd.label(sample.pcFinal)}（1 位 ${hd.label(sample.pcTop1)} " +
                "${"%.2f".format(sample.pcConf)}）→ " + (if (same) "一致" else "不一致") +
                (if (sample.expected.isNotEmpty()) " · 正解ラベル: ${hd.label(sample.expected)}" else "") +
                (when (idsMatch) { true -> " · トークン列は PC と同じ"; false -> " · ★トークン列が PC と違う"; null -> "" })
            compareView.setTextColor(if (same) OK else WARN)
        } else {
            compareView.visibility = View.GONE
        }
    }

    // ================================================================= 画面

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(BG)
            setPadding(dp(16f), dp(16f), dp(16f), dp(16f))
        }
        root.addView(text(titleText, 20f, TEXT, bold = true))
        root.addView(text(subtitleText, 13f, SUB).apply { setPadding(0, dp(4f), 0, dp(8f)) })
        statusView = text("", 13f, SUB).apply { setPadding(0, 0, 0, dp(8f)) }
        root.addView(statusView)

        // 状態（走行）
        val motionRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        motionRow.addView(text("車の状態: ", 14f, TEXT))
        motionButtons = listOf("stopped" to "停車中", "moving" to "走行中").map { (key, label) ->
            Button(this).apply {
                text = label
                isAllCaps = false
                setOnClickListener { motion = key; refreshMotion() }
                motionRow.addView(this)
            }
        }
        root.addView(motionRow)

        // 入力
        val inputRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        inputEdit = EditText(this).apply {
            hint = "例: エアコンを25度にして"
            setSingleLine(true)
            imeOptions = EditorInfo.IME_ACTION_DONE
            setOnEditorActionListener { _, _, _ -> classify(text.toString(), null); true }
        }
        inputRow.addView(inputEdit, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        classifyButton = Button(this).apply {
            text = "分類"
            setOnClickListener { classify(inputEdit.text.toString(), null) }
        }
        inputRow.addView(classifyButton)
        root.addView(inputRow)

        // 例文
        if (samples.isNotEmpty()) {
            root.addView(text("例文（タップで分類。${refShort}と並べて表示）", 13f, SUB).apply { setPadding(0, dp(8f), 0, dp(4f)) })
            val rows = samples.chunked((samples.size + 2) / 3)
            for (row in rows) {
                val line = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                for (s in row) {
                    line.addView(Button(this).apply {
                        text = s.utterance
                        isAllCaps = false
                        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                        setOnClickListener {
                            inputEdit.setText(s.utterance)
                            motion = s.motion; refreshMotion()
                            classify(s.utterance, s)
                        }
                    }.also { sampleButtons.add(it) })
                }
                root.addView(HorizontalScrollView(this).apply { addView(line) })
            }
        }

        // 結果
        resultBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12f), dp(12f), dp(12f), dp(12f))
            background = GradientDrawable().apply { setColor(CARD); cornerRadius = dp(10f).toFloat() }
            visibility = View.GONE
        }
        finalView = text("", 22f, OK, bold = true)
        detailView = text("", 14f, TEXT).apply { setPadding(0, dp(6f), 0, dp(6f)) }
        barsBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        timingView = text("", 12f, SUB).apply { setPadding(0, dp(6f), 0, 0) }
        compareView = text("", 12f, SUB).apply { setPadding(0, dp(4f), 0, 0) }
        listOf(finalView, detailView, barsBox, timingView, compareView).forEach { resultBox.addView(it) }
        root.addView(resultBox, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(12f) })

        root.addView(text("注: 判定は分類ヘッドだけで出しています（規則・Dispatcher なし）。$noteText", 11f, SUB)
            .apply { setPadding(0, dp(12f), 0, 0) })

        setContentView(ScrollView(this).apply { addView(root) })
        refreshMotion()
    }

    private fun bar(label: String, prob: Float): View {
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        row.addView(text(label, 13f, TEXT), LinearLayout.LayoutParams(dp(130f), WRAP_CONTENT))
        val track = LinearLayout(this).apply {
            background = GradientDrawable().apply { setColor(TRACK); cornerRadius = dp(4f).toFloat() }
        }
        val fill = View(this).apply {
            background = GradientDrawable().apply { setColor(ACCENT); cornerRadius = dp(4f).toFloat() }
        }
        track.addView(fill, LinearLayout.LayoutParams(0, dp(10f), prob.coerceIn(0f, 1f)))
        track.addView(View(this), LinearLayout.LayoutParams(0, dp(10f), (1f - prob).coerceIn(0f, 1f)))
        row.addView(track, LinearLayout.LayoutParams(0, dp(10f), 1f))
        row.addView(text(" ${"%.2f".format(prob)}", 13f, TEXT))
        return row
    }

    private fun refreshMotion() {
        motionButtons.forEachIndexed { i, b ->
            val on = (i == 0 && motion == "stopped") || (i == 1 && motion == "moving")
            b.setTextColor(if (on) Color.WHITE else TEXT)
            b.background = GradientDrawable().apply { setColor(if (on) ACCENT else TRACK); cornerRadius = dp(6f).toFloat() }
        }
    }

    private fun setBusy(b: Boolean) {
        busy = b
        ui.post {
            classifyButton.isEnabled = !b
            sampleButtons.forEach { it.isEnabled = !b }
        }
    }

    private fun status(s: String, color: Int) {
        statusView.text = s
        statusView.setTextColor(color)
    }

    private fun hideKeyboard() = runOnUiThread {
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(inputEdit.windowToken, 0)
    }

    private fun text(s: String, sp: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        text = s
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        setTextColor(color)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun dp(v: Float): Int = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    companion object {
        const val TAG = "EdgeSLMG4Prune"
        // 既定は層 17（層 0〜16＋最後の正規化）の分類専用まとめ読み（T=64・KV なし・出力 h）。samples.json の job.model で差し替えられる
        const val MODEL = "models/ops/p2_L17_epctx.onnx"
        const val DEFAULT_TITLE = "Gemma 4 E2B ヘッド分類（前半 17 層・スマホの NPU）"
        const val DEFAULT_SUBTITLE = "Gemma 4 E2B の 35 層のうち前半 17 層だけを残したモデルが、発話の意味を表す h を NPU で作り、" +
            "小さな分類ヘッドが行き先を決めます。通信は使いません。"
        const val DEFAULT_MODEL_LABEL = "Gemma 4 E2B（前半 17 層・まとめ読み）"
        const val AUTO_GAP_MS = 2500L       // 自動実行で結果を見せておく時間
        // Gemma 4 のチャットテンプレート（system なし。PC の ref_qat.first_turn_ids・G1 の抽出と同じ）。
        // samples.json に PC で作った値があればそちらを使う
        // 比べる相手の既定の表示（改善版のアプリに載せるのは端末の h で学習したヘッドなので、相手は一括収集での端末の判定）
        const val DEFAULT_REF_LABEL = "一括収集での判定（同じ端末・同じヘッド）"
        const val DEFAULT_REF_SHORT = "一括収集での判定"
        const val DEFAULT_NOTE = "ヘッドは端末の h で学習しています。"
        const val DEFAULT_PREFIX = "<bos><|turn>user\n"
        const val DEFAULT_SUFFIX = "<turn|>\n<|turn>model\n"
        val BG = Color.parseColor("#F7F7F5")
        val CARD = Color.WHITE
        val TEXT = Color.parseColor("#1F2328")
        val SUB = Color.parseColor("#667085")
        val OK = Color.parseColor("#1E7B45")
        val WARN = Color.parseColor("#B54708")
        val ERR = Color.parseColor("#B42318")
        val ACCENT = Color.parseColor("#2F6FEB")
        val TRACK = Color.parseColor("#E4E7EC")
    }
}
