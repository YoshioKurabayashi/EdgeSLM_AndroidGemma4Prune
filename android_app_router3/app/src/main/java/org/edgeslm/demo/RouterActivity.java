package org.edgeslm.demo;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.edgeslm.router.Args;
import org.edgeslm.router.BertJa;
import org.edgeslm.router.Flow;
import org.edgeslm.router.Head;
import org.edgeslm.router.Policy;
import org.edgeslm.router.Say;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * ★P39（判定デモ v3・測定、2026-10-15）: v2 の複製に、時間の内訳の記録（Java・JNI・C++）と、測定の流し方 {@code --es bench} を足した。判定は v2 と同じ。
 * 設計: DoC/設計書_P39_時間の内訳を測る.md
 * <ul>
 * <li>{@code --es bench "17,26,35"}: 確かめ用の行の先頭 {@code bench_rows}（既定 300）行を、層ごとに、{@code bench_modes}（既定 "par,serial"）の順に流す。
 *     par = v2 と同じ（BERT を NPU と並行）、serial = BERT を先に 1 本で動かしてから Gemma（それぞれ単独の時間）。
 *     層を変えるときは、ほかの層を閉じてから開く。各モードの前に {@code bench_warmup}（既定 10）行を流して捨てる。結果は {@code out/bench_L<n>_<mode>.jsonl}</li>
 * <li>{@code --es qnnprof "detailed"}（bench と一緒に）: 層ごとの測定のあと、QNN の計測（profiling_level）を付けて開き直し、{@code qnnprof_rows}（既定 30）行を流す。
 *     計測の記録は {@code out/qnnprof_L<n>.csv}、そのときの判定は {@code out/qnnprof_L<n>.jsonl}。開けなければ飛ばす</li>
 * <li>終わるとログ {@code router_bench done}</li>
 * </ul>
 *
 * 判定デモの画面（お試し EdgeSLM_AndroidGemma4Prune P24・2026-10-05。P36・2026-10-14 で v2: 規則 P31・引数 P35・ヘッド 1,656 行・機能の定義で絞る段）。**発話 1 つについて、いまの構成の最終の判定までを、スマホの中だけで出す**。
 *
 * 設計: DoC/設計書_P24_判定デモアプリ.md。
 *
 * <h2>流れ（すべて端末の中）</h2>
 * <ol>
 * <li>発話を F2 の文（状態＋履歴なし＋発話）にし、Gemma 4 のチャットの形で包んで ID 列にする（{@code Gemma4Tokenizer}）</li>
 * <li>層を減らした Gemma 4 E2B（前半 17 層 か 26 層。NPU）に 1 回で読ませ、{@code h}（最後・平均）を受け取る（Runner の {@code classify_prefill}）</li>
 * <li>{@link Head} で、行き先の確率と棄却の得点を出す（CPU）</li>
 * <li>並行して、発話を日本語 BERT の ID 列にし（{@link BertJa}）、BERT（ONNX・CPU）で「機能にない依頼」の確率を出す（{@link BertScorer}）</li>
 * <li>{@link Flow#decide} が、BERT が断る → 規則 → 棄却ヘッド → 確信度 → 引数 → 検査 → BERT のフラグ の順に見て、実行／雑談／聞き返し／拒否 を決める</li>
 * </ol>
 * 操作は模擬（何を実行するかを表示するだけ）。聞き返した後の 2 回目のやり取りは無い。
 *
 * <h2>層の切り替え</h2>
 * 画面のボタンで 17 層と 26 層を切り替える。グラフは、初めて選んだときに開き、開いたままにする（2 つで約 1,346MiB。1 プロセスの NPU の上限 約 1,710MiB の内）。
 * 2 つ目が開けなければ、先に開いていた方を閉じてから開き直す。
 *
 * <h2>確かめ・録画用の自動実行（adb の am start の引数）</h2>
 * <ul>
 * <li>{@code --es prompts "発話1|発話2"}: 順に判定する（手で押したときと同じ経路）。ログ {@code router_auto item}・{@code router_auto done}、結果は {@code out/router_auto.jsonl}</li>
 * <li>{@code --es eval "17,26"}: 確かめ用の行（router.json の eval。v2 は {@code router/eval_v10.jsonl}）を、書いた層の順に全部流し、{@code out/router_eval_L<n>.jsonl} に書く。終わるとログ {@code router_eval done}。
 *     PC の {@code app/p24_device.py check} が、PC の判定と突き合わせる</li>
 * </ul>
 */
public class RouterActivity extends Activity {
    static final String TAG = "EdgeSLMRouter";
    static final int MAX_PROMPT_TOKENS = 64;        // グラフが 1 回で読める長さ（T=64）
    static final long AUTO_GAP_MS = 2500L;          // 自動実行で、結果を見せておく時間
    static final int BG = Color.parseColor("#F7F7F5"), CARD = Color.WHITE, TEXT = Color.parseColor("#1F2328"), SUB = Color.parseColor("#667085"),
            OK = Color.parseColor("#1E7B45"), WARN = Color.parseColor("#B54708"), ERR = Color.parseColor("#B42318"), ACCENT = Color.parseColor("#2F6FEB"),
            TRACK = Color.parseColor("#E4E7EC");

    /** 層の数ごとの設定と、開いたもの */
    static final class Layer {
        int n, contextMib;
        String label, detail, model, headDir;
        volatile long handle = 0L;
        Head head;
        long loadMs;
        boolean strict;
    }

    /** 画面の例文（PC の判定つき） */
    static final class Sample {
        String id, utterance, motion;
        JSONObject pc;          // 層の数（文字列）→ PC の outcome・stage・tool・args
    }

    /** 1 回の判定の結果（画面と記録の両方に使う） */
    static final class Result {
        String utterance, motion, error;
        Layer layer;
        Flow.Decision d;
        Head.Out head;
        int[] gemmaIds, bertIds;
        double bertProb, gemmaTokMs, bertTokMs, bertMs, npuRunMs, npuPrepMs, npuTotalMs, ruleMs, wallMs;
        // P39: 時間の内訳（Java 側）。job の JSON を作る・JNI の呼び出し全体・結果の JSON を読む・h を float にする・BERT を待った時間
        double jobBuildMs, jniMs, parseMs, floatsMs, bertWaitMs, gemmaPathMs, bertPathMs;
        boolean serial;
        JSONObject timing, jniTiming;       // P39: C++ の内訳（classify_prefill の "timing"）と JNI の中の時間
    }

    // ---- 画面の部品 ----
    private TextView statusView, finalView, whatView, detailView, timingView, compareView;
    private LinearLayout resultBox, stagesBox, barsBox;
    private EditText inputEdit;
    private Button runButton;
    private final List<Button> layerButtons = new ArrayList<Button>(), motionButtons = new ArrayList<Button>(), sampleButtons = new ArrayList<Button>();

    // ---- 状態 ----
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();     // NPU・判定（1 本ずつ）
    private final ExecutorService bertPool = Executors.newSingleThreadExecutor();   // BERT（NPU と並行）
    private volatile boolean busy = true;
    private String motion = "stopped";
    private JSONObject cfg;
    private String configError;
    private final List<Layer> layers = new ArrayList<Layer>();
    private final List<Sample> samples = new ArrayList<Sample>();
    private volatile Layer current;
    private Gemma4Tokenizer gemma;
    private BertJa bertTok;
    private BertScorer bert;
    private double bertHi = 1.0;
    private int bertMaxLength = 64;
    private String chatPrefix = "<bos><|turn>user\n", chatSuffix = "<turn|>\n<|turn>model\n";
    private final ArrayDeque<String> autoQueue = new ArrayDeque<String>();
    private boolean autoActive = false;
    private int autoIndex = 0;
    private String evalLayers;
    // P39: 測定の流し方（--es bench ほか）
    private String benchLayers, benchModes = "par,serial", qnnProfLevel;
    private int benchRows = 300, benchWarmup = 10, qnnProfRows = 30;
    private volatile String profFile;       // QNN の計測を付けて開くときだけ（jobBase が ep の options に足す）

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ensureDirs();
        loadConfig();
        String prompts = getIntent() == null ? null : getIntent().getStringExtra("prompts");
        if (prompts != null) {
            for (String p : prompts.split("\\|")) {
                if (p.trim().length() > 0) autoQueue.add(p.trim());
            }
            Log.i(TAG, "router_auto queued " + autoQueue.size());
        }
        evalLayers = getIntent() == null ? null : getIntent().getStringExtra("eval");
        if (getIntent() != null) {          // P39
            benchLayers = getIntent().getStringExtra("bench");
            if (getIntent().getStringExtra("bench_modes") != null) benchModes = getIntent().getStringExtra("bench_modes");
            qnnProfLevel = getIntent().getStringExtra("qnnprof");
            benchRows = intExtra("bench_rows", benchRows);
            benchWarmup = intExtra("bench_warmup", benchWarmup);
            qnnProfRows = intExtra("qnnprof_rows", qnnProfRows);
        }
        buildUi();
        if (cfg == null) {
            Log.e(TAG, "router_fatal config: " + configError);
            status("資材がありません（" + configError + "）。PC で .\\scripts\\run_p24.ps1 -Stage deploy を実行してください。", ERR);
            return;
        }
        open();
    }

    @Override
    protected void onDestroy() {
        worker.shutdownNow();
        bertPool.shutdownNow();
        for (Layer l : layers) {
            long h = l.handle;
            l.handle = 0L;
            if (h != 0L) {
                try {
                    NativeBridge.INSTANCE.nativeDestroy(h);
                } catch (Throwable ignored) {
                }
            }
        }
        if (bert != null) bert.close();
        super.onDestroy();
    }

    // ================================================================= 準備

    private File base() {
        return getExternalFilesDir(null);
    }

    /** フォルダはアプリが作る（adb で作ると持ち主が shell になり、アプリから読めない） */
    private void ensureDirs() {
        File b = base();
        if (b == null) return;
        for (String s : new String[]{"models", "models/ops", "jobs", "inputs", "out", "tokenizer", "router", "router/L17", "router/L26", "router/bert"}) {
            File d = new File(b, s);
            if (!d.isDirectory() && !d.mkdirs()) Log.w(TAG, "mkdir failed: " + d.getAbsolutePath());
        }
        Log.i(TAG, "externalFilesDir=" + b.getAbsolutePath());
    }

    private static String readText(File f) throws Exception {
        BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
        try {
            StringBuilder b = new StringBuilder();
            char[] buf = new char[8192];
            for (int n; (n = r.read(buf)) > 0; ) b.append(buf, 0, n);
            return b.toString();
        } finally {
            r.close();
        }
    }

    /** `router/router.json`（PC の app/p24_assets.py が作る）。画面を組む前に読む */
    private void loadConfig() {
        try {
            File f = new File(base(), "router/router.json");
            if (!f.exists()) {
                configError = "router/router.json が無い";
                return;
            }
            JSONObject o = new JSONObject(readText(f));
            chatPrefix = o.optString("chat_prefix", chatPrefix);
            chatSuffix = o.optString("chat_suffix", chatSuffix);
            JSONArray ls = o.getJSONArray("layers");
            for (int i = 0; i < ls.length(); i++) {
                JSONObject x = ls.getJSONObject(i);
                Layer l = new Layer();
                l.n = x.getInt("n");
                l.label = x.optString("label", "前半 " + l.n + " 層");
                l.detail = x.optString("detail", "");
                l.model = x.getString("model");
                l.headDir = x.getString("head");
                l.contextMib = x.optInt("context_mib", 0);
                layers.add(l);
            }
            JSONObject b = o.getJSONObject("bert");
            bertHi = b.getDouble("hi");
            bertMaxLength = b.optInt("max_length", 64);
            JSONArray ss = o.optJSONArray("samples");
            for (int i = 0; ss != null && i < ss.length(); i++) {
                JSONObject x = ss.getJSONObject(i);
                Sample s = new Sample();
                s.id = x.optString("id");
                s.utterance = x.getString("utterance");
                s.motion = x.optString("motion", "stopped");
                s.pc = x.optJSONObject("pc");
                samples.add(s);
            }
            int want = o.optInt("default_layer", layers.get(0).n);
            current = layers.get(0);
            for (Layer l : layers) {
                if (l.n == want) current = l;
            }
            cfg = o;
        } catch (Throwable t) {
            Log.w(TAG, "router.json ignored", t);
            configError = "router/router.json が読めない: " + t.getMessage();
            cfg = null;
        }
    }

    private JSONObject jobBase(Layer l) throws Exception {
        JSONObject j = new JSONObject();
        j.put("job_id", "router");
        j.put("task", "classify_prefill");      // プロンプトを 1 回の推論で読み、h の最後と平均を返す（既存の Runner の機能）
        j.put("model", l.model);
        j.put("mask_neg", -60.0);
        j.put("pad_id", 0);
        JSONObject x = cfg.optJSONObject("job");            // 埋め込み・PLE・位置の表の場所（PC の tables.json から）
        if (x != null) {
            for (Iterator<String> it = x.keys(); it.hasNext(); ) {
                String k = it.next();
                j.put(k, x.get(k));
            }
        }
        JSONObject ep = new JSONObject();
        ep.put("provider", "QNN");
        ep.put("backend", "libQnnHtp.so");
        JSONObject eo = new JSONObject().put("offload_graph_io_quantization", "0").put("qnn_context_priority", "high");
        if (profFile != null && qnnProfLevel != null) {     // P39: QNN の計測（開くときだけ効く）
            eo.put("profiling_level", qnnProfLevel);
            eo.put("profiling_file_path", profFile);
        }
        ep.put("options", eo);
        ep.put("session", new JSONObject().put("intra_op_num_threads", 1).put("log_severity_level", 3));
        j.put("ep", ep);
        return j;
    }

    private String withPaths(JSONObject job) {
        return JsonInject.INSTANCE.addRuntimePaths(job.toString(), base().getAbsolutePath(), getApplicationInfo().nativeLibraryDir, null);
    }

    /** 画面を開いたとき: トークナイザ・BERT・既定の層のグラフを順に開く */
    private void open() {
        setBusy(true);
        final Layer first = current;
        worker.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    post("準備中… Gemma 4 のトークナイザ", SUB);
                    gemma = Gemma4Tokenizer.Companion.load(new File(base(), "tokenizer"));
                    post("準備中… 日本語 BERT（CPU）", SUB);
                    JSONObject b = cfg.getJSONObject("bert");
                    bertTok = BertJa.load(new File(base(), b.optString("dir", "router/bert")));
                    org.edgeslm.router.Words.tok = bertTok;          // P36: 機能の定義で絞る段（Scope）も、BERT と同じ単語の分け方を使う
                    bert = new BertScorer(new File(base(), b.optString("model", "router/bert/model.onnx")), b.optInt("threads", 4));
                    Log.i(TAG, "bert ready load_ms=" + bert.loadMs);
                    post("準備中… " + first.label + " のグラフ（NPU）", SUB);
                    String err = ensureOpen(first);
                    if (err != null) {
                        Log.e(TAG, "router_fatal layer " + first.n + " open: " + err);
                        post("グラフを開けませんでした。ほかのアプリでモデルを開いていないか確認してください。\n" + err, ERR);
                        return;
                    }
                    ui.post(new Runnable() {
                        @Override
                        public void run() {
                            ready();
                            Log.i(TAG, "router ready layer=" + first.n);
                            if (benchLayers != null) runBench(benchLayers);      // P39
                            else if (evalLayers != null) runEval(evalLayers);
                            else if (!autoQueue.isEmpty()) {
                                autoActive = true;
                                new File(base(), "out/router_auto.jsonl").delete();
                                ui.postDelayed(new Runnable() {
                                    @Override
                                    public void run() {
                                        runNextAuto();
                                    }
                                }, 1500);
                            }
                        }
                    });
                } catch (Throwable t) {
                    Log.e(TAG, "router_fatal open: " + t, t);
                    post("準備に失敗しました: " + t.getClass().getSimpleName() + ": " + t.getMessage() + "\n資材（.\\scripts\\run_p24.ps1 -Stage deploy）を送ったか確認してください。", ERR);
                }
            }
        });
    }

    /** 層のグラフとヘッドを開く（開いていれば何もしない）。失敗したら理由を返す。worker のスレッドで呼ぶ */
    private String ensureOpen(Layer l) {
        try {
            if (l.head == null) {
                l.head = Head.load(new File(base(), l.headDir));
                l.head.thresholds.bertHi = bertHi;
            }
            if (l.handle != 0L) return null;
            String why = "";
            for (int attempt = 0; attempt < 2; attempt++) {
                long t0 = System.currentTimeMillis();
                long h = NativeBridge.INSTANCE.nativeCreate(withPaths(jobBase(l)));
                JSONObject d = new JSONObject(NativeBridge.INSTANCE.nativeDescribe(h));
                if (h != 0L) {
                    l.handle = h;
                    l.loadMs = System.currentTimeMillis() - t0;
                    l.strict = d.optBoolean("strict_no_cpu_fallback", false);
                    Log.i(TAG, "layer " + l.n + " ready load_ms=" + l.loadMs + " strict=" + l.strict + " backend=" + d.optString("backend", "?"));
                    return null;
                }
                why = d.optString("error", "理由は不明");
                Log.w(TAG, "layer " + l.n + " open failed (" + attempt + "): " + why);
                boolean closed = false;         // ほかの層を開いたままだと入らないことがある。閉じてからもう 1 回だけ試す
                for (Layer o : layers) {
                    if (o != l && o.handle != 0L) {
                        long oh = o.handle;
                        o.handle = 0L;
                        NativeBridge.INSTANCE.nativeDestroy(oh);
                        closed = true;
                    }
                }
                if (!closed) break;
            }
            return why;
        } catch (Throwable t) {
            Log.e(TAG, "ensureOpen", t);
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    private void ready() {
        Layer l = current;
        status("準備完了 · " + l.label + "（" + l.detail + "）· " + (l.strict ? "全ノード NPU（HTP）" : "一部 CPU の可能性")
                + " · グラフの読み込み " + String.format(Locale.US, "%.1f", l.loadMs / 1000.0) + " 秒", OK);
        refreshToggles();
        setBusy(false);
    }

    private void selectLayer(final Layer l) {
        if (busy || l == current || cfg == null) return;
        setBusy(true);
        status(l.label + " のグラフを開いています…（初回は数秒）", SUB);
        worker.execute(new Runnable() {
            @Override
            public void run() {
                final String err = ensureOpen(l);
                ui.post(new Runnable() {
                    @Override
                    public void run() {
                        if (err != null) {
                            status(l.label + " のグラフを開けませんでした: " + err, ERR);
                            // 開き直しで元の層を閉じたかもしれない。いま開いている層があれば、そちらに合わせる
                            for (Layer o : layers) {
                                if (o.handle != 0L) current = o;
                            }
                            refreshToggles();
                            setBusy(current.handle == 0L);
                            return;
                        }
                        current = l;
                        resultBox.setVisibility(View.GONE);
                        ready();
                    }
                });
            }
        });
    }

    // ================================================================= 判定

    /** F2 の文（PC の `edge_router.data.build.render(row, "F2")` と同じ。履歴は常に「なし」） */
    private static String renderF2(String utterance, String motion) {
        return "[状態] 走行:" + motion + " 権限:allowed\n[履歴] なし\n[発話] " + utterance;
    }

    /** 1 つの発話を判定する（同期）。worker のスレッドで呼ぶ */
    private Result compute(final String utterance, String motion, Layer l) {
        return compute(utterance, motion, l, false);
    }

    /**
     * P39: serial = true なら、BERT を先に（このスレッドで）動かしてから Gemma を動かす（それぞれ単独の時間を見る）。
     * false は v2 と同じ（BERT を別スレッドで NPU と並行）。判定の計算は同じ
     */
    private Result compute(final String utterance, String motion, Layer l, boolean serial) {
        final Result r = new Result();
        r.utterance = utterance;
        r.motion = motion;
        r.layer = l;
        r.serial = serial;
        long t0 = System.nanoTime();
        try {
            Callable<Object> bertJob = new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                    long a = System.nanoTime();
                    r.bertIds = bertTok.encode(utterance, bertMaxLength);
                    long b = System.nanoTime();
                    r.bertProb = bert.prob(r.bertIds);
                    r.bertTokMs = (b - a) / 1e6;
                    r.bertMs = (System.nanoTime() - b) / 1e6;
                    return null;
                }
            };
            Future<Object> bf = null;
            if (serial) {
                long a = System.nanoTime();
                bertJob.call();
                r.bertPathMs = (System.nanoTime() - a) / 1e6;
            } else {
                // BERT は CPU の別スレッドで、NPU と並行に動かす
                bf = bertPool.submit(bertJob);
            }
            long g0 = System.nanoTime();
            long a = System.nanoTime();
            r.gemmaIds = gemma.encode(chatPrefix + renderF2(utterance, motion) + chatSuffix);
            r.gemmaTokMs = (System.nanoTime() - a) / 1e6;
            if (r.gemmaIds.length > MAX_PROMPT_TOKENS) {
                if (bf != null) bf.get();
                r.error = "発話が長すぎます（Gemma 4 のトークン " + r.gemmaIds.length + " 個。1 回で読めるのは " + MAX_PROMPT_TOKENS + " 個まで）";
                return r;
            }
            long jb0 = System.nanoTime();
            JSONArray ids = new JSONArray();
            for (int v : r.gemmaIds) ids.put(v);
            String jobJson = withPaths(jobBase(l).put("prompt_ids", ids));
            long jb1 = System.nanoTime();
            String raw = NativeBridge.INSTANCE.nativeRunJob(l.handle, jobJson);
            long jb2 = System.nanoTime();
            JSONObject j = new JSONObject(raw);
            long jb3 = System.nanoTime();
            r.jobBuildMs = (jb1 - jb0) / 1e6;
            r.jniMs = (jb2 - jb1) / 1e6;
            r.parseMs = (jb3 - jb2) / 1e6;
            r.timing = j.optJSONObject("timing");
            r.jniTiming = j.optJSONObject("jni_timing");
            if (!j.optBoolean("ok", false) || !j.has("h_last")) {
                if (bf != null) bf.get();
                r.error = "NPU の推論に失敗しました: " + j.optString("error");
                return r;
            }
            r.npuRunMs = j.optDouble("run_ms", 0.0);
            r.npuPrepMs = j.optDouble("prep_ms", 0.0);
            r.npuTotalMs = j.optDouble("total_ms", 0.0);
            long f0 = System.nanoTime();
            float[] hl = floats(j.getJSONArray("h_last"));
            float[] hm = floats(j.getJSONArray("h_mean"));
            r.floatsMs = (System.nanoTime() - f0) / 1e6;
            r.head = l.head.predict(hl, hm);
            r.gemmaPathMs = (System.nanoTime() - g0) / 1e6;
            long w0 = System.nanoTime();
            if (bf != null) bf.get();
            r.bertWaitMs = (System.nanoTime() - w0) / 1e6;
            long c = System.nanoTime();
            Args.State st = new Args.State();
            st.motion = motion;
            r.d = Flow.decide(utterance, r.head.action, l.head.classes, r.head.reject, l.head.thresholds, r.bertProb, st);
            r.ruleMs = (System.nanoTime() - c) / 1e6;
        } catch (Throwable t) {
            Log.e(TAG, "compute", t);
            r.error = t.getClass().getSimpleName() + ": " + t.getMessage();
        }
        r.wallMs = (System.nanoTime() - t0) / 1e6;
        return r;
    }

    private static float[] floats(JSONArray a) throws Exception {
        float[] f = new float[a.length()];
        for (int i = 0; i < f.length; i++) f[i] = (float) a.getDouble(i);
        return f;
    }

    private static String orDash(String s) {
        return s == null ? "-" : s;
    }

    /** PC の判定（outcome・stage・tool・args）と同じか。PC の判定が無ければ null */
    private static Boolean samePc(Result r, JSONObject pc) {
        if (pc == null || r.d == null) return null;
        JSONObject p = pc.optJSONObject(String.valueOf(r.layer.n));
        if (p == null) return null;
        String want = p.optString("outcome") + "|" + (p.isNull("stage") ? "-" : p.optString("stage")) + "|" + (p.isNull("tool") ? "-" : p.optString("tool")) + "|" + (p.isNull("args") ? "-" : p.optString("args"));
        String got = r.d.outcome + "|" + orDash(r.d.stage) + "|" + orDash(r.d.tool) + "|" + Args.canon(r.d.args);
        return want.equals(got);
    }

    /** 記録の 1 行（PC の app/p24_device.py が読む） */
    private JSONObject record(Result r, String id, JSONObject pc, JSONArray pcGemmaIds) throws Exception {
        JSONObject o = new JSONObject();
        o.put("id", id == null ? JSONObject.NULL : id);
        o.put("utterance", r.utterance);
        o.put("motion", r.motion);
        o.put("layer", r.layer.n);
        o.put("error", r.error == null ? JSONObject.NULL : r.error);
        o.put("wall_ms", r.wallMs);
        o.put("gemma_tok_ms", r.gemmaTokMs);
        o.put("n_tokens", r.gemmaIds == null ? 0 : r.gemmaIds.length);
        if (pcGemmaIds != null && r.gemmaIds != null) {
            boolean same = pcGemmaIds.length() == r.gemmaIds.length;
            for (int i = 0; same && i < r.gemmaIds.length; i++) same = pcGemmaIds.getInt(i) == r.gemmaIds[i];
            o.put("gemma_ids_match", same);
        }
        if (r.bertIds != null) {
            JSONArray b = new JSONArray();
            for (int v : r.bertIds) b.put(v);
            o.put("bert_ids", b);
            o.put("bert", r.bertProb);
            o.put("bert_ms", r.bertMs);
            o.put("bert_tok_ms", r.bertTokMs);
        }
        if (r.d == null) return o;
        Flow.Decision d = r.d;
        o.put("outcome", d.outcome);
        o.put("stage", d.stage == null ? JSONObject.NULL : d.stage);
        o.put("tool", d.tool == null ? JSONObject.NULL : d.tool);
        o.put("args", d.args == null ? JSONObject.NULL : Args.canon(d.args));
        o.put("rule", d.rule == null ? JSONObject.NULL : d.rule);
        o.put("head_action", d.headAction == null ? JSONObject.NULL : d.headAction);
        int k = 0;
        for (int i = 1; i < r.head.action.length; i++) {
            if (r.head.action[i] > r.head.action[k]) k = i;
        }
        o.put("top1", r.layer.head.classes[k]);
        o.put("conf", r.head.action[k]);
        o.put("reject", r.head.reject);
        o.put("npu_run_ms", r.npuRunMs);
        o.put("npu_prep_ms", r.npuPrepMs);
        o.put("npu_total_ms", r.npuTotalMs);
        o.put("head_ms", r.head.ms);
        o.put("rule_ms", r.ruleMs);
        // P39: 時間の内訳
        o.put("serial", r.serial);
        o.put("job_build_ms", r.jobBuildMs);
        o.put("jni_ms", r.jniMs);
        o.put("parse_ms", r.parseMs);
        o.put("floats_ms", r.floatsMs);
        o.put("bert_wait_ms", r.bertWaitMs);
        o.put("gemma_path_ms", r.gemmaPathMs);
        o.put("bert_path_ms", r.bertPathMs);
        if (r.timing != null) o.put("timing", r.timing);
        if (r.jniTiming != null) o.put("jni_timing", r.jniTiming);
        Boolean same = samePc(r, pc);
        o.put("pc_same", same == null ? JSONObject.NULL : same);
        return o;
    }

    private void judge(String text, final Sample sample) {
        final String utterance = text == null ? "" : text.trim();
        final Layer l = current;
        if (busy || cfg == null || l == null || l.handle == 0L || utterance.length() == 0) return;
        hideKeyboard();
        setBusy(true);
        status("計算中…（NPU と BERT）", SUB);
        showPending(utterance);
        final String m = motion;
        worker.execute(new Runnable() {
            @Override
            public void run() {
                final Result r = compute(utterance, m, l);
                JSONObject rec = null;
                try {
                    rec = record(r, sample == null ? null : sample.id, sample == null ? null : sample.pc, null);
                    append(new File(base(), "out/router_auto.jsonl"), rec.toString() + "\n");
                } catch (Throwable t) {
                    Log.w(TAG, "record", t);
                }
                final String line = String.valueOf(rec);
                Log.i(TAG, "judge layer=" + l.n + " " + line);
                ui.post(new Runnable() {
                    @Override
                    public void run() {
                        show(r, sample);
                        status("準備完了 · " + l.label + " · 別の発話も試せます", OK);
                        setBusy(false);
                        if (autoActive) {
                            Log.i(TAG, "router_auto item " + autoIndex + " " + line);
                            if (!autoQueue.isEmpty()) {
                                ui.postDelayed(new Runnable() {
                                    @Override
                                    public void run() {
                                        runNextAuto();
                                    }
                                }, AUTO_GAP_MS);
                            } else {
                                autoActive = false;
                                Log.i(TAG, "router_auto done n=" + autoIndex);
                            }
                        }
                    }
                });
            }
        });
    }

    private static void append(File f, String s) throws Exception {
        BufferedWriter w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f, true), "UTF-8"));
        try {
            w.write(s);
        } finally {
            w.close();
        }
    }

    /** 自動実行の 1 件。「17:」「26:」で始まる項目は、その層に切り替えてから続ける */
    private void runNextAuto() {
        String u = autoQueue.pollFirst();
        if (u == null) return;
        if (u.matches("\\d+:")) {
            int n = Integer.parseInt(u.substring(0, u.length() - 1));
            for (final Layer l : layers) {
                if (l.n == n && l != current) {
                    selectLayer(l);
                    waitReadyThenNext();
                    return;
                }
            }
            runNextAuto();
            return;
        }
        autoIndex++;
        Sample s = null;
        for (Sample x : samples) {
            if (x.utterance.equals(u)) s = x;
        }
        if (s != null) {
            motion = s.motion;
            refreshToggles();
        }
        inputEdit.setText(u);
        judge(u, s);
    }

    private void waitReadyThenNext() {
        ui.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (busy) waitReadyThenNext();
                else if (!autoQueue.isEmpty()) runNextAuto();
                else {
                    autoActive = false;
                    Log.i(TAG, "router_auto done n=" + autoIndex);
                }
            }
        }, 500);
    }

    private int intExtra(String k, int dflt) {
        try {
            String v = getIntent().getStringExtra(k);
            return v == null ? dflt : Integer.parseInt(v.trim());
        } catch (Throwable t) {
            return dflt;
        }
    }

    /** P39: 開いているほかの層を閉じる（35 層は 1 つで約 1,273MiB。ほかと一緒には開かない） */
    private void closeOthers(Layer keep) {
        for (Layer o : layers) {
            if (o != keep && o.handle != 0L) {
                long oh = o.handle;
                o.handle = 0L;
                NativeBridge.INSTANCE.nativeDestroy(oh);
                Log.i(TAG, "router_bench closed layer " + o.n);
            }
        }
    }

    private void writeRows(Layer l, List<JSONObject> rows, int from, int to, boolean serial, File out, String what) throws Exception {
        BufferedWriter bw = out == null ? null : new BufferedWriter(new OutputStreamWriter(new FileOutputStream(out), "UTF-8"));
        try {
            for (int i = from; i < to; i++) {
                JSONObject row = rows.get(i);
                Result r = compute(row.getString("utterance"), row.optString("motion", "stopped"), l, serial);
                if (bw != null) {
                    JSONObject rec = record(r, row.optString("id"), row.optJSONObject("pc"), row.optJSONArray("gemma_ids"));
                    rec.remove("bert_ids");
                    bw.write(rec.toString());
                    bw.write("\n");
                }
                if ((i - from + 1) % 50 == 0) post("測定中… " + l.label + " " + what + " " + (i - from + 1) + "/" + (to - from), SUB);
            }
        } finally {
            if (bw != null) bw.close();
        }
    }

    /** P39: 時間の内訳を測る（層ごと・モードごと。最後に QNN の計測も） */
    private void runBench(final String which) {
        setBusy(true);
        worker.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    List<JSONObject> rows = new ArrayList<JSONObject>();
                    BufferedReader rd = new BufferedReader(new InputStreamReader(new FileInputStream(new File(base(), cfg.optString("eval", "router/eval_v11.jsonl"))), "UTF-8"));
                    try {
                        for (String line; (line = rd.readLine()) != null; ) {
                            if (line.length() > 0) rows.add(new JSONObject(line));
                        }
                    } finally {
                        rd.close();
                    }
                    int n = Math.min(benchRows, rows.size());
                    int w = Math.min(benchWarmup, rows.size());
                    Log.i(TAG, "router_bench start layers=" + which + " modes=" + benchModes + " rows=" + n + " warmup=" + w + " qnnprof=" + qnnProfLevel);
                    for (String ws : which.split(",")) {
                        Layer l = null;
                        for (Layer x : layers) {
                            if (String.valueOf(x.n).equals(ws.trim())) l = x;
                        }
                        if (l == null) {
                            Log.w(TAG, "router_bench layer " + ws + " is not in router.json");
                            continue;
                        }
                        closeOthers(l);
                        String err = ensureOpen(l);
                        if (err != null) {
                            Log.e(TAG, "router_bench layer " + l.n + " open failed: " + err);
                            continue;
                        }
                        current = l;
                        Log.i(TAG, "router_bench layer " + l.n + " open load_ms=" + l.loadMs);
                        for (String m : benchModes.split(",")) {
                            boolean serial = m.trim().equals("serial");
                            writeRows(l, rows, 0, w, serial, null, "準備運転");
                            long t0 = System.nanoTime();
                            writeRows(l, rows, 0, n, serial, new File(base(), "out/bench_L" + l.n + "_" + m.trim() + ".jsonl"), m.trim());
                            Log.i(TAG, "router_bench layer " + l.n + " mode " + m.trim() + " rows=" + n + " sec=" + String.format(Locale.US, "%.1f", (System.nanoTime() - t0) / 1e9));
                        }
                        if (qnnProfLevel != null) {
                            long oh = l.handle;                 // 計測を付けて開き直す
                            l.handle = 0L;
                            NativeBridge.INSTANCE.nativeDestroy(oh);
                            File csv = new File(base(), "out/qnnprof_L" + l.n + ".csv");
                            csv.delete();
                            profFile = csv.getAbsolutePath();
                            String perr = ensureOpen(l);
                            if (perr != null) {
                                Log.e(TAG, "router_bench qnnprof layer " + l.n + " open failed: " + perr);
                            } else {
                                writeRows(l, rows, 0, Math.min(qnnProfRows, rows.size()), false, new File(base(), "out/qnnprof_L" + l.n + ".jsonl"), "QNN の計測");
                                long ph = l.handle;
                                l.handle = 0L;
                                NativeBridge.INSTANCE.nativeDestroy(ph);    // 計測の記録は閉じるときにも書かれる
                                Log.i(TAG, "router_bench qnnprof layer " + l.n + " csv_bytes=" + csv.length());
                            }
                            profFile = null;
                        }
                    }
                    Log.i(TAG, "router_bench done");
                    ui.post(new Runnable() {
                        @Override
                        public void run() {
                            setBusy(false);
                            status("測定が終わりました（結果は out/bench_L<n>_<mode>.jsonl）", OK);
                        }
                    });
                } catch (Throwable t) {
                    Log.e(TAG, "router_fatal bench: " + t, t);
                    post("測定に失敗しました: " + t.getMessage(), ERR);
                }
            }
        });
    }

    /** 確かめ用の行を、書いた層の順に全部流す（画面には進み具合だけ出す） */
    private void runEval(final String which) {
        setBusy(true);
        worker.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    List<JSONObject> rows = new ArrayList<JSONObject>();
                    BufferedReader rd = new BufferedReader(new InputStreamReader(new FileInputStream(new File(base(), cfg.optString("eval", "router/eval_v6.jsonl"))), "UTF-8"));
                    try {
                        for (String line; (line = rd.readLine()) != null; ) {
                            if (line.length() > 0) rows.add(new JSONObject(line));
                        }
                    } finally {
                        rd.close();
                    }
                    for (String w : which.split(",")) {
                        Layer l = null;
                        for (Layer x : layers) {
                            if (String.valueOf(x.n).equals(w.trim())) l = x;
                        }
                        if (l == null) continue;
                        String err = ensureOpen(l);
                        if (err != null) {
                            Log.e(TAG, "router_eval layer " + l.n + " open failed: " + err);
                            continue;
                        }
                        current = l;
                        File out = new File(base(), "out/router_eval_L" + l.n + ".jsonl");
                        out.delete();
                        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(out), "UTF-8"));
                        int same = 0, done = 0;
                        try {
                            for (JSONObject row : rows) {
                                Result r = compute(row.getString("utterance"), row.optString("motion", "stopped"), l);
                                JSONObject rec = record(r, row.optString("id"), row.optJSONObject("pc"), row.optJSONArray("gemma_ids"));
                                bw.write(rec.toString());
                                bw.write("\n");
                                if (rec.optBoolean("pc_same", false)) same++;
                                done++;
                                if (done % 20 == 0 || done == rows.size()) post("確かめ中… " + l.label + " " + done + "/" + rows.size() + " 行（PC と同じ判定 " + same + "）", SUB);
                            }
                        } finally {
                            bw.close();
                        }
                        Log.i(TAG, "router_eval layer " + l.n + " rows=" + done + " pc_same=" + same);
                    }
                    Log.i(TAG, "router_eval done");
                    ui.post(new Runnable() {
                        @Override
                        public void run() {
                            ready();
                            status("確かめが終わりました（結果は out/router_eval_L<n>.jsonl）", OK);
                        }
                    });
                } catch (Throwable t) {
                    Log.e(TAG, "router_fatal eval: " + t, t);
                    post("確かめに失敗しました: " + t.getMessage(), ERR);
                }
            }
        });
    }

    // ================================================================= 表示

    private static String f2(double v) {
        return String.format(Locale.US, "%.2f", v);
    }

    private static String f3(double v) {
        return v > 0.9995 || v < 0.0005 ? String.format(Locale.US, "%.4f", v) : String.format(Locale.US, "%.3f", v);
    }

    private static String ms(double v) {
        return String.format(Locale.US, v < 10 ? "%.1f" : "%.0f", v);
    }

    private void showPending(String utt) {
        resultBox.setVisibility(View.VISIBLE);
        resultBox.setAlpha(0.55f);
        finalView.setText("計算中…");
        finalView.setTextColor(SUB);
        whatView.setText("「" + utt + "」");
        detailView.setText("");
        stagesBox.removeAllViews();
        barsBox.removeAllViews();
        timingView.setText("");
        compareView.setVisibility(View.GONE);
    }

    /** 段の 1 行。state: 0 = 通った、1 = ここで止めた、2 = 見ていない */
    private void stageRow(String name, String value, int state, int stopColor) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0, dp(2f), 0, dp(2f));
        int color = state == 0 ? TEXT : state == 1 ? stopColor : SUB;
        TextView mark = text(state == 0 ? "✓" : state == 1 ? "■" : "–", 13f, state == 0 ? OK : color, true);
        row.addView(mark, new LinearLayout.LayoutParams(dp(18f), ViewGroup.LayoutParams.WRAP_CONTENT));
        row.addView(text(name, 13f, color, state == 1), new LinearLayout.LayoutParams(dp(104f), ViewGroup.LayoutParams.WRAP_CONTENT));
        row.addView(text(value, 13f, color, false), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        stagesBox.addView(row);
    }

    private void show(Result r, Sample sample) {
        resultBox.setVisibility(View.VISIBLE);
        resultBox.setAlpha(1f);
        stagesBox.removeAllViews();
        barsBox.removeAllViews();
        whatView.setText("「" + r.utterance + "」（" + ("moving".equals(r.motion) ? "走行中" : "停車中") + "・" + r.layer.label + "）");
        if (r.d == null) {
            finalView.setText("判定できませんでした");
            finalView.setTextColor(ERR);
            detailView.setText(r.error == null ? "" : r.error);
            timingView.setText("");
            compareView.setVisibility(View.GONE);
            return;
        }
        Flow.Decision d = r.d;
        Flow.Thresholds th = r.layer.head.thresholds;
        int color;
        String title, detail;
        if (Flow.EXECUTE.equals(d.outcome)) {
            color = OK;
            title = "実行: " + Say.tool(d.tool);
            detail = Say.args(d.tool, d.args) + "\n" + d.tool + " " + Args.canon(d.args) + "\n（模擬です。実際の操作はしません" + (Policy.isWrite(d.tool) ? "" : "。読むだけの機能") + "）";
        } else if (Flow.CHAT.equals(d.outcome)) {
            color = ACCENT;
            title = "雑談として応答";
            detail = "機能の呼び出しはしません（応答文の生成は、このデモには入っていません）";
        } else if (Flow.ASK.equals(d.outcome)) {
            color = WARN;
            title = "聞き返す";
            if ("規則".equals(d.stage)) detail = "規則で止めました: " + Say.rule(d.rule);
            else if ("棄却ヘッド".equals(d.stage)) detail = "意図がはっきりしません（棄却の得点 " + f3(d.reject) + " ≥ " + f3(th.reject) + "）";
            else if ("確信度".equals(d.stage)) detail = "行き先に自信がありません（確信度 " + f2(d.confidence) + " < " + f2(d.tau) + "）";
            else if ("引数が足りない".equals(d.stage)) detail = Say.tool(d.tool) + " と判定しましたが、" + Say.missing(d.missing) + " が分かりません";
            else if ("検査".equals(d.stage)) detail = Say.tool(d.tool) + " と判定しましたが、必要な引数がありません";
            else detail = "機能にない依頼かもしれません（BERT " + f3(d.bert) + " ≥ " + f2(Flow.BERT_LO) + "）";
        } else {
            color = ERR;
            title = "断る";
            if ("検査".equals(d.stage)) detail = (d.response == null ? "実行できない指定です" : d.response) + "\n（" + d.tool + " " + Args.canon(d.args) + "）";
            else detail = "この車の機能にない依頼です（BERT " + f3(d.bert) + " ≥ " + f3(th.bertHi) + "）";
        }
        finalView.setText(title);
        finalView.setTextColor(color);
        detailView.setText(detail);

        // ---- 段ごとの値（止めた段より後ろは、見ていない）
        String[] order = "CHAT".equals(d.action) || d.action == null
                ? new String[]{"BERT が断る", "規則", "棄却ヘッド", "確信度", "BERT のフラグ"}
                : new String[]{"BERT が断る", "規則", "棄却ヘッド", "確信度", "引数が足りない", "検査", "BERT のフラグ"};
        int stop = order.length;
        for (int i = 0; i < order.length; i++) {
            if (order[i].equals(d.stage)) stop = i;
        }
        int k = 0;
        for (int i = 1; i < r.head.action.length; i++) {
            if (r.head.action[i] > r.head.action[k]) k = i;
        }
        stageRow("ヘッド", "1 位 " + Say.action(r.layer.head.classes[k]) + "（" + f2(r.head.action[k]) + "）· Gemma 4 " + r.layer.label + " の h から", 0, color);
        for (int i = 0; i < order.length; i++) {
            int state = i < stop ? 0 : i == stop ? 1 : 2;
            String name = order[i], v;
            if (name.equals("BERT が断る")) {
                name = "BERT";
                v = "機能にない依頼の確率 " + f3(d.bert) + "（断る ≥ " + f3(th.bertHi) + "）";
            } else if (name.equals("規則")) {
                v = state == 2 ? "" : d.rule == null ? "当たりなし" : d.rule + (d.action != null && !d.action.equals(d.headAction) ? " → " + Say.action(d.action) + " に変更" : "");
            } else if (name.equals("棄却ヘッド")) {
                v = "得点 " + f3(d.reject) + "（聞き返す ≥ " + f3(th.reject) + "）";
            } else if (name.equals("確信度")) {
                v = state == 2 ? "" : f2(d.confidence) + "（聞き返す < " + f2(d.tau) + "）";
            } else if (name.equals("引数が足りない")) {
                name = "引数";
                v = state == 2 ? "" : state == 1 ? Say.missing(d.missing) + " が無い" : Args.canon(d.args);
            } else if (name.equals("検査")) {
                v = state == 2 ? "" : state == 1 ? "通らない" : "型・範囲・走行状態の決まりを通過";
            } else {
                name = "BERT のフラグ";
                v = state == 2 ? "" : f3(d.bert) + "（聞き返す ≥ " + f2(Flow.BERT_LO) + "）";
            }
            stageRow(name, v, state, color);
        }

        // ---- ヘッドの確率（上位 3 つ）
        Integer[] idx = new Integer[r.head.action.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        final double[] p = r.head.action;
        java.util.Arrays.sort(idx, new java.util.Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return Double.compare(p[b], p[a]);
            }
        });
        for (int i = 0; i < 3 && i < idx.length; i++) barsBox.addView(bar(Say.action(r.layer.head.classes[idx[i]]), (float) p[idx[i]]));

        timingView.setText("合計 " + ms(r.wallMs) + " ms（NPU " + ms(r.npuRunMs) + "・入力の準備 " + ms(r.npuPrepMs) + "・ヘッド " + ms(r.head.ms) + " ／ BERT " + ms(r.bertMs)
                + "・BERT の入力 " + ms(r.bertTokMs) + " ／ 規則・引数・検査 " + ms(r.ruleMs) + "・Gemma の入力 " + ms(r.gemmaTokMs) + "）· Gemma のトークン " + r.gemmaIds.length + " 個");

        Boolean same = sample == null || !sample.motion.equals(r.motion) ? null : samePc(r, sample.pc);
        if (same == null) {
            compareView.setVisibility(View.GONE);
        } else {
            JSONObject pc = sample.pc.optJSONObject(String.valueOf(r.layer.n));
            compareView.setVisibility(View.VISIBLE);
            compareView.setText("PC での判定（評価の流れ・同じ層）: " + pc.optString("outcome") + (pc.isNull("stage") ? "" : "（" + pc.optString("stage") + "）")
                    + (pc.isNull("tool") ? "" : " " + pc.optString("tool")) + (pc.isNull("args") ? "" : " " + pc.optString("args")) + " → " + (same ? "一致" : "不一致"));
            compareView.setTextColor(same ? OK : WARN);
        }
    }

    // ================================================================= 画面

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setPadding(dp(16f), dp(16f), dp(16f), dp(16f));
        root.addView(text(cfg == null ? "車載ルーター 判定デモ" : cfg.optString("title", "車載ルーター 判定デモ"), 20f, TEXT, true));
        TextView sub = text(cfg == null ? "" : cfg.optString("subtitle", ""), 13f, SUB, false);
        sub.setPadding(0, dp(4f), 0, dp(8f));
        root.addView(sub);
        statusView = text("", 13f, SUB, false);
        statusView.setPadding(0, 0, 0, dp(8f));
        root.addView(statusView);

        // 層の切り替え
        LinearLayout layerRow = new LinearLayout(this);
        layerRow.setOrientation(LinearLayout.HORIZONTAL);
        layerRow.setGravity(Gravity.CENTER_VERTICAL);
        layerRow.addView(text("モデル: ", 14f, TEXT, false));
        for (final Layer l : layers) {
            Button b = pill(l.label);
            b.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    selectLayer(l);
                }
            });
            layerButtons.add(b);
            layerRow.addView(b, gap());
        }
        root.addView(layerRow);

        // 車の状態
        LinearLayout motionRow = new LinearLayout(this);
        motionRow.setOrientation(LinearLayout.HORIZONTAL);
        motionRow.setGravity(Gravity.CENTER_VERTICAL);
        motionRow.addView(text("車の状態: ", 14f, TEXT, false));
        final String[][] motions = {{"stopped", "停車中"}, {"moving", "走行中"}};
        for (final String[] m : motions) {
            Button b = pill(m[1]);
            b.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    motion = m[0];
                    refreshToggles();
                }
            });
            motionButtons.add(b);
            motionRow.addView(b, gap());
        }
        root.addView(motionRow);

        // 入力
        LinearLayout inputRow = new LinearLayout(this);
        inputRow.setOrientation(LinearLayout.HORIZONTAL);
        inputRow.setGravity(Gravity.CENTER_VERTICAL);
        inputEdit = new EditText(this);
        inputEdit.setHint("例: エアコンを25度にして");
        inputEdit.setSingleLine(true);
        inputEdit.setImeOptions(EditorInfo.IME_ACTION_DONE);
        inputEdit.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, android.view.KeyEvent event) {
                judge(inputEdit.getText().toString(), null);
                return true;
            }
        });
        inputRow.addView(inputEdit, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        runButton = new Button(this);
        runButton.setText("判定");
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                judge(inputEdit.getText().toString(), null);
            }
        });
        inputRow.addView(runButton);
        root.addView(inputRow);

        // 例文
        if (!samples.isEmpty()) {
            TextView cap = text("例文（タップで判定。PC での判定と並べて表示）", 13f, SUB, false);
            cap.setPadding(0, dp(8f), 0, dp(4f));
            root.addView(cap);
            int per = (samples.size() + 2) / 3;
            for (int s = 0; s < samples.size(); s += per) {
                LinearLayout line = new LinearLayout(this);
                line.setOrientation(LinearLayout.HORIZONTAL);
                for (int i = s; i < Math.min(s + per, samples.size()); i++) {
                    final Sample x = samples.get(i);
                    Button b = new Button(this);
                    b.setText(x.utterance);
                    b.setAllCaps(false);
                    b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f);
                    b.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            inputEdit.setText(x.utterance);
                            motion = x.motion;
                            refreshToggles();
                            judge(x.utterance, x);
                        }
                    });
                    sampleButtons.add(b);
                    line.addView(b);
                }
                HorizontalScrollView hs = new HorizontalScrollView(this);
                hs.addView(line);
                root.addView(hs);
            }
        }

        // 結果
        resultBox = new LinearLayout(this);
        resultBox.setOrientation(LinearLayout.VERTICAL);
        resultBox.setPadding(dp(12f), dp(12f), dp(12f), dp(12f));
        GradientDrawable card = new GradientDrawable();
        card.setColor(CARD);
        card.setCornerRadius(dp(10f));
        resultBox.setBackground(card);
        resultBox.setVisibility(View.GONE);
        whatView = text("", 13f, SUB, false);
        finalView = text("", 24f, OK, true);
        finalView.setPadding(0, dp(4f), 0, 0);
        detailView = text("", 15f, TEXT, false);
        detailView.setPadding(0, dp(4f), 0, dp(8f));
        stagesBox = new LinearLayout(this);
        stagesBox.setOrientation(LinearLayout.VERTICAL);
        barsBox = new LinearLayout(this);
        barsBox.setOrientation(LinearLayout.VERTICAL);
        barsBox.setPadding(0, dp(8f), 0, 0);
        timingView = text("", 12f, SUB, false);
        timingView.setPadding(0, dp(6f), 0, 0);
        compareView = text("", 12f, SUB, false);
        compareView.setPadding(0, dp(4f), 0, 0);
        for (View v : new View[]{whatView, finalView, detailView, stagesBox, barsBox, timingView, compareView}) resultBox.addView(v);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(12f);
        root.addView(resultBox, lp);

        TextView note = text(cfg == null ? "" : "注: " + cfg.optString("note", ""), 11f, SUB, false);
        note.setPadding(0, dp(12f), 0, 0);
        root.addView(note);

        ScrollView sv = new ScrollView(this);
        sv.addView(root);
        setContentView(sv);
        refreshToggles();
        setBusy(true);
    }

    private LinearLayout.LayoutParams gap() {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(38f));
        lp.rightMargin = dp(8f);
        return lp;
    }

    private Button pill(String label) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        b.setPadding(dp(12f), 0, dp(12f), 0);
        return b;
    }

    private void paint(Button b, boolean on) {
        b.setTextColor(on ? Color.WHITE : TEXT);
        GradientDrawable g = new GradientDrawable();
        g.setColor(on ? ACCENT : TRACK);
        g.setCornerRadius(dp(6f));
        b.setBackground(g);
    }

    private void refreshToggles() {
        for (int i = 0; i < layerButtons.size(); i++) paint(layerButtons.get(i), layers.get(i) == current);
        for (int i = 0; i < motionButtons.size(); i++) paint(motionButtons.get(i), (i == 0) == "stopped".equals(motion));
    }

    private View bar(String label, float prob) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.addView(text(label, 12f, SUB, false), new LinearLayout.LayoutParams(dp(130f), ViewGroup.LayoutParams.WRAP_CONTENT));
        LinearLayout track = new LinearLayout(this);
        GradientDrawable tg = new GradientDrawable();
        tg.setColor(TRACK);
        tg.setCornerRadius(dp(4f));
        track.setBackground(tg);
        View fill = new View(this);
        GradientDrawable fg = new GradientDrawable();
        fg.setColor(ACCENT);
        fg.setCornerRadius(dp(4f));
        fill.setBackground(fg);
        float p = Math.max(0f, Math.min(1f, prob));
        track.addView(fill, new LinearLayout.LayoutParams(0, dp(8f), p));
        track.addView(new View(this), new LinearLayout.LayoutParams(0, dp(8f), 1f - p));
        row.addView(track, new LinearLayout.LayoutParams(0, dp(8f), 1f));
        row.addView(text(" " + f2(prob), 12f, SUB, false));
        return row;
    }

    private void setBusy(final boolean b) {
        busy = b;
        ui.post(new Runnable() {
            @Override
            public void run() {
                boolean on = !b && cfg != null;
                runButton.setEnabled(on);
                for (Button x : sampleButtons) x.setEnabled(on);
                for (Button x : layerButtons) x.setEnabled(on);
            }
        });
    }

    private void status(String s, int color) {
        statusView.setText(s);
        statusView.setTextColor(color);
    }

    private void post(final String s, final int color) {
        ui.post(new Runnable() {
            @Override
            public void run() {
                status(s, color);
            }
        });
    }

    private void hideKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null) imm.hideSoftInputFromWindow(inputEdit.getWindowToken(), 0);
    }

    private TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private int dp(float v) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }
}
