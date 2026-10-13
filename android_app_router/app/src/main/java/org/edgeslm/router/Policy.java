package org.edgeslm.router;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 実行前の検査（P24）。EdgeSLM_MultiHead_Multi の `dispatch/policy.py`・`configs/tools.json`・`configs/policy.yaml` を写したもの。
 *
 * 3 段: 1) 型と列挙（schema） 2) 値の範囲 3) 走行状態と権限の表（操作系の機能だけ）。最初に落ちた段で止める。
 * 表は、ここに直接書く（アプリは YAML を読まない）。元の表を変えたら、ここも変える。`tools/p24_java` の `ParityTest` が食い違いを見つける。
 */
public final class Policy {
    private Policy() {}

    /** 検査の結果。ok でなければ、missing があれば聞き返し、無ければ断る */
    public static final class Verdict {
        public final boolean ok;
        public final String stage;       // schema・range・policy
        public final String reason;      // 開発者向け
        public final String response;    // 利用者向け
        public final boolean missing;

        Verdict(boolean ok, String stage, String reason, String response, boolean missing) {
            this.ok = ok;
            this.stage = stage;
            this.reason = reason;
            this.response = response;
            this.missing = missing;
        }

        /** 突き合わせ用（`p24_fixtures.verdict` と同じ書き方） */
        public String code() {
            return ok ? "ok" : (missing ? "ask" : "refuse");
        }
    }

    static final Verdict OK = new Verdict(true, null, null, null, false);

    /** 引数 1 つの決まり */
    static final class Prop {
        final String type;          // string・integer・number
        final List<String> enums;   // 無ければ null
        final Double lo, hi;

        Prop(String type, String[] enums, Double lo, Double hi) {
            this.type = type;
            this.enums = enums == null ? null : Arrays.asList(enums);
            this.lo = lo;
            this.hi = hi;
        }
    }

    static final class Tool {
        final String effect;
        final String label;
        final String[] required;
        final Map<String, Prop> props = new HashMap<String, Prop>();

        Tool(String label, String effect, String... required) {
            this.label = label;
            this.effect = effect;
            this.required = required;
        }

        Tool str(String k, String... enums) {
            props.put(k, new Prop("string", enums.length == 0 ? null : enums, null, null));
            return this;
        }

        Tool num(String k, String type, double lo, double hi) {
            props.put(k, new Prop(type, null, lo, hi));
            return this;
        }
    }

    static final Map<String, Tool> TOOLS = new HashMap<String, Tool>();

    static {
        TOOLS.put("get_vehicle_status", new Tool("車両情報取得", "read", "item").str("item", "fuel", "range", "odometer", "tire_pressure", "door", "battery", "all"));
        TOOLS.put("search_poi", new Tool("POI・QA", "read", "query").str("query").num("radius_km", "number", 0.5, 50));
        TOOLS.put("control_hvac", new Tool("空調", "write").num("temperature", "number", 16, 30).num("fan_level", "integer", 0, 7)
                .str("mode", "auto", "cool", "heat", "defrost", "off").str("zone", "driver", "passenger", "rear", "all"));
        TOOLS.put("control_media", new Tool("メディア", "write", "action").str("action", "play", "pause", "next", "previous", "volume_set", "mute")
                .num("volume", "integer", 0, 40).str("target"));
        TOOLS.put("set_navigation", new Tool("ナビゲーション", "write", "action").str("destination").str("action", "set", "add_waypoint", "cancel", "reroute"));
        TOOLS.put("capture_camera", new Tool("カメラ利用", "read", "view").str("view", "front", "rear", "left", "right", "surround", "cabin"));
        TOOLS.put("get_media_info", new Tool("メディア情報QA", "read", "item").str("item", "now_playing", "artist", "album", "source", "queue"));
        TOOLS.put("search_vehicle_manual", new Tool("車両知識QA", "read", "query").str("query").num("top_k", "integer", 1, 10));
    }

    /** 操作系（車の状態を変える）か */
    public static boolean isWrite(String tool) {
        Tool t = TOOLS.get(tool);
        return t != null && "write".equals(t.effect);
    }

    /** 画面に出す機能の名前 */
    public static String label(String tool) {
        Tool t = TOOLS.get(tool);
        return t == null ? tool : t.label;
    }

    static Verdict checkSchema(String tool, Map<String, Object> values) {
        Tool spec = TOOLS.get(tool);
        if (spec == null) return new Verdict(false, "schema", "未知のツール: " + tool, "その操作には対応していません。", false);
        for (String k : spec.required) {
            if (!values.containsKey(k)) return new Verdict(false, "schema", "required が足りない: " + k, null, true);
        }
        for (Map.Entry<String, Object> e : values.entrySet()) {
            String key = e.getKey();
            Object value = e.getValue();
            Prop rule = spec.props.get(key);
            if (rule == null) return new Verdict(false, "schema", "schema に無い引数: " + key, "その指定には対応していません。", false);
            if ("string".equals(rule.type) && !(value instanceof String)) return new Verdict(false, "schema", key + " は文字列のはず", "指定を解釈できませんでした。", false);
            if ("integer".equals(rule.type) && !(value instanceof Integer)) return new Verdict(false, "schema", key + " は整数のはず", "指定を解釈できませんでした。", false);
            if ("number".equals(rule.type) && !(value instanceof Number)) return new Verdict(false, "schema", key + " は数値のはず", "指定を解釈できませんでした。", false);
            if (rule.enums != null && !rule.enums.contains(value)) return new Verdict(false, "schema", key + " が列挙に無い: " + value, "その指定には対応していません。", false);
        }
        return OK;
    }

    static String labelOf(String key) {
        if ("temperature".equals(key)) return "温度";
        if ("fan_level".equals(key)) return "風量";
        if ("volume".equals(key)) return "音量";
        if ("radius_km".equals(key)) return "検索範囲";
        if ("top_k".equals(key)) return "件数";
        return key;
    }

    static String fmt(Double v) {
        if (v == null) return "—";
        return v == Math.rint(v) ? String.valueOf((long) (double) v) : String.valueOf(v);
    }

    static Verdict checkRange(String tool, Map<String, Object> values) {
        Tool spec = TOOLS.get(tool);
        for (Map.Entry<String, Object> e : values.entrySet()) {
            if (!(e.getValue() instanceof Number)) continue;
            Prop rule = spec.props.get(e.getKey());
            if (rule == null) continue;
            double v = ((Number) e.getValue()).doubleValue();
            String say = labelOf(e.getKey()) + "は" + fmt(rule.lo) + "から" + fmt(rule.hi) + "の範囲で指定できます。";
            if (rule.lo != null && v < rule.lo) return new Verdict(false, "range", e.getKey() + " が下限未満", say, false);
            if (rule.hi != null && v > rule.hi) return new Verdict(false, "range", e.getKey() + " が上限超過", say, false);
        }
        return OK;
    }

    static Verdict checkPolicy(String tool, Map<String, Object> values, Args.State st) {
        if (!isWrite(tool)) return OK;
        if (!"allowed".equals(st.permission)) {
            // 権限が無いときは、操作系の 3 つとも止める（条件なし）
            String why = "set_navigation".equals(tool) ? "目的地の設定には操作権限が要る" : "control_hvac".equals(tool) ? "空調の変更には操作権限が要る" : "メディアの操作には権限が要る";
            return new Verdict(false, "policy", why, "この操作には権限が必要です。", false);
        }
        if ("moving".equals(st.motion)) {
            if ("set_navigation".equals(tool) && "set".equals(values.get("action"))) {
                return new Verdict(false, "policy", "走行中の目的地入力は前方不注意になる", "走行中は目的地の設定ができません。停車してからお試しください。", false);
            }
            if ("control_media".equals(tool) && "volume_set".equals(values.get("action"))) {
                Object v = values.get("volume");
                if (v instanceof Number && ((Number) v).doubleValue() >= 30) {
                    return new Verdict(false, "policy", "走行中の大音量は周囲の音が聞こえなくなる", "走行中は音量を30以上にできません。", false);
                }
            }
        }
        return OK;
    }

    /** 3 段の検査を順に通す。最初に落ちた段で止める */
    public static Verdict inspect(String tool, Map<String, Object> values, Args.State st) {
        Verdict v = checkSchema(tool, values);
        if (!v.ok) return v;
        v = checkRange(tool, values);
        if (!v.ok) return v;
        return checkPolicy(tool, values, st);
    }
}
