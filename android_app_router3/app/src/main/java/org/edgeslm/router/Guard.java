package org.edgeslm.router;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 引数の歯止め（P36 でアプリに入れた。Python の `p25_arguments`（G1〜G4）・`p26_arguments`（G5〜G10））。
 * 取り出した値に発話の中の根拠が無い・別の値の根拠があるときは、実行せずに聞き返す（`missing = ["value_check"]`）。
 *
 * 返す値は、止めた歯止めの印（{@link Args.Arguments#guard} に入れる）。止めなければ null。
 */
final class Guard {
    private Guard() {}

    static final String HVAC = "control_hvac", MEDIA = "control_media", NAV = "set_navigation";
    static final String WS = "[" + Py.WS + "]";

    // ---------------------------------------------------------------- P25
    static final Pattern SUPPORT_HVAC = Pattern.compile("つけ|付け|点け|入れ|動か|オン|スイッチ|起動|かけ");
    static final Pattern SUPPORT_MEDIA = Pattern.compile("かけ|掛け|流し|流す|流そ|再生|スタート|プレイ|ラジオ|FM|AM|リピート|聴|聞きたい|BGM|音楽に|曲に|曲を変え|曲変え|にして|(?:音楽|曲).{0,4}(?:ほしい|欲しい)");
    static final Pattern OTHER_HVAC = Pattern.compile("切|消|止|オフ|強|弱|上げ|下げ|高く|低く|最大|最小|暖か|涼し|冷や");
    static final Pattern OTHER_MEDIA = Pattern.compile("次|前|戻|飛ば|スキップ|音量|ボリューム|大き|小さ|上げ|下げ|止|消|ミュート|最初");
    static final Pattern NUM = Pattern.compile("[0-9０-９]+(?:[.．][0-9０-９]+)?");
    static final Pattern NOT_AMOUNT = Pattern.compile("^" + WS + "*(?:曲|回|分|秒|時|号|キロ|km|人|個|枚|件|番|日|年|月|週|メートル|m(?![a-z])|%|パーセント)");
    static final Pattern TRACK = Pattern.compile("^" + WS + "*つ" + WS + "*(?:前|後|先|戻)");
    static final Pattern KEEP = Pattern.compile("^" + WS + "*(?:度|段階|段目|段|つ)?" + WS + "*(?:でいい|のまま|はそのまま|で大丈夫|でよい|で十分)");
    static final Pattern ABSOLUTE = Pattern.compile("^" + WS + "*(?:度|段階|段目|段|目盛り?|つ)?" + WS + "*(?:くらい|ぐらい|程度|位|ほど)?" + WS + "*(?:に|まで|へ)");
    static final Pattern TIMES = Pattern.compile("^" + WS + "*倍");
    static final Pattern NEGATED = Pattern.compile("(?:止め|切ら|消さ|とめ)(?:ないで|ずに)");
    static final Pattern HVAC_STOP = Pattern.compile("(?:エアコン|冷房|暖房|空調|ヒーター|クーラー|デフロスター?|曇り止め|曇り取り|送風|風)(?:を|は|も)?(?:もう)?(?:切|消|止め|とめ|オフ)");
    static final Pattern GO_WORD = Pattern.compile("入れ|つけ|付け|オン|強|弱|上げ|下げ");
    static final Pattern DEST_NOISE = Pattern.compile("できるだけ|なるべく|早く|急いで|すぐ|とりあえず|ひとまず|いったん|先に|まずは|もう一度|もう一回|そのまま|ゆっくり|今から|いまから|これから"
            + "|して|って|たい|ない|ので|から|けど|ください|ほしい");
    static final int DEST_MAX = 20;
    static final String[] NUMERIC = {"temperature", "fan_level", "volume"};

    static Double state(Args.State st, String key) {
        if ("temperature".equals(key)) return st.temperature;
        if ("fan_level".equals(key)) return st.fanLevel == null ? null : st.fanLevel.doubleValue();
        if ("volume".equals(key)) return st.volume == null ? null : st.volume.doubleValue();
        return null;
    }

    static double num(Object v) {
        return ((Number) v).doubleValue();
    }

    static boolean near(double a, double b) {
        return Math.abs(a - b) < 1e-9;
    }

    static boolean lookingAt(Pattern p, String s) {
        return p.matcher(s).lookingAt();
    }

    /** P25 の歯止め。first = 元の処理の最初のメモ（{@link Args.Arguments#note}） */
    static String check25(String tool, String utterance, Map<String, Object> values, String first, Args.State st) {
        String u = Args.kanjiNumbers(utterance);
        if (first == null) first = "";
        if (HVAC.equals(tool) && first.contains("既定値")) {
            if (!Py.find(SUPPORT_HVAC, u) || Py.find(OTHER_HVAC, u)) return "P25:G1:auto";
        }
        if (MEDIA.equals(tool) && first.startsWith("指定が無いので再生")) {
            if (!Py.find(SUPPORT_MEDIA, u) || Py.find(OTHER_MEDIA, u)) return "P25:G1:play";
        }
        if (HVAC.equals(tool) || MEDIA.equals(tool)) {
            List<String> keys = new ArrayList<String>();
            List<Double> vals = new ArrayList<Double>();
            for (String k : NUMERIC) {
                if (values.containsKey(k) && values.get(k) != null) {
                    keys.add(k);
                    vals.add(num(values.get(k)));
                }
            }
            Matcher m = NUM.matcher(u);
            while (m.find()) {
                if (u.substring(Math.max(0, m.start() - 2), m.start()).equals("国道")) continue;
                double n = Double.parseDouble(Py.zenToHan(m.group()).replace('．', '.'));
                String rest = u.substring(m.end());
                if (lookingAt(NOT_AMOUNT, rest) || lookingAt(KEEP, rest) || lookingAt(TRACK, rest)) continue;
                boolean ok = false;
                for (int i = 0; i < keys.size(); i++) {
                    double v = vals.get(i);
                    Double now = state(st, keys.get(i));
                    if (lookingAt(TIMES, rest)) ok |= now != null && near(v, now * n);
                    else if (lookingAt(ABSOLUTE, rest)) ok |= near(v, n);
                    else ok |= near(v, n) || (now != null && near(Math.abs(v - now), n));
                }
                if (!ok) return "P25:G2";
            }
        }
        if (HVAC.equals(tool)) {
            String v = NEGATED.matcher(u).replaceAll("");
            Object mode = values.get("mode");
            if (("off".equals(mode) && Py.find(GO_WORD, v)) || (!"off".equals(mode) && Py.find(HVAC_STOP, v) && Py.find(GO_WORD, v))) return "P25:G3";
        }
        if (NAV.equals(tool) && values.get("destination") != null && !String.valueOf(values.get("destination")).isEmpty()) {
            String d = String.valueOf(values.get("destination"));
            if (Py.find(DEST_NOISE, d) || d.codePointCount(0, d.length()) > DEST_MAX) return "P25:G4";
        }
        return null;
    }

    // ---------------------------------------------------------------- P26
    static final Pattern HVAC_STOP2 = Pattern.compile("(エアコン|冷房|暖房|空調|ヒーター|クーラー|デフロスター?|曇り止め|曇り取り|送風|風)(?:を|は|も|だけ)?(?:もう)?(?:切|消|止め|とめ|オフ)");
    static final Pattern MEDIA_STOP = Pattern.compile("止め|とめて|とめと|ストップ|一時停止|停止|ポーズ");
    static final Pattern MEDIA_NEGATED = Pattern.compile("(?:止め|とめ|停止し)(?:ないで|ずに)|止めるんじゃなく|停止じゃなく|ストップじゃなく|止めた|止めてた|止まった|止まって");
    static final Pattern EXTREME = Pattern.compile("(?:音量|ボリューム|音|風量|風の量|風の強さ|風|温度)(?:を|は|の|も)?[^、。]{0,4}?(?:いっぱい|目いっぱい|最大|マックス|最高|全開|最小|最低|ゼロ|一番|いちばん|フル)");
    static final Pattern HALF_OF = Pattern.compile("半分");
    static final Pattern HALF = Pattern.compile("([0-9]+(?:\\.[0-9]+)?)" + WS + "*度" + WS + "*半");
    static final String[] KANA_W = {"ひとつ", "ふたつ", "みっつ", "よっつ", "いつつ"};
    static final int[] KANA_N = {1, 2, 3, 4, 5};
    static final Pattern PLACE_MOVE = Pattern.compile("([一-龥ァ-ヶー々A-Za-z0-9]{1,15})(?:に|へ|まで)(?:寄|向か|行|着|戻|帰)");
    static final Pattern VIA_THEN = Pattern.compile("寄ってから|寄った後|寄ったあと");
    static final String[] PURPOSE = {"修理", "買い物", "食事", "散歩", "仕事", "迎え", "送り", "トイレ", "休憩", "用事", "給油", "充電", "お参り", "見学", "診察"};
    static final Pattern MODE_ON = Pattern.compile("(?:暖房|冷房|クーラー|ヒーター|デフロスター?|曇り止め|曇り取り)(?:を|は)?(?:つけ|付け|入れ|オン)");

    /** STOP_TARGET: 止める語の対象ごとに、食い違いとみなす値（null = mode が無い） */
    static boolean stopTarget(String noun, Object mode) {
        if (noun.equals("冷房") || noun.equals("クーラー")) return "cool".equals(mode);
        if (noun.equals("暖房") || noun.equals("ヒーター")) return "heat".equals(mode);
        if (noun.equals("デフロスタ") || noun.equals("デフロスター") || noun.equals("曇り止め") || noun.equals("曇り取り")) return "defrost".equals(mode);
        if (noun.equals("エアコン") || noun.equals("空調")) return mode == null || "auto".equals(mode) || "cool".equals(mode) || "heat".equals(mode) || "defrost".equals(mode);
        return false;
    }

    static String check26(String tool, String utterance, Map<String, Object> values, Args.State st) {
        String u = Args.kanjiNumbers(utterance);
        if (HVAC.equals(tool)) {
            Object mode = values.get("mode");
            String v = NEGATED.matcher(u).replaceAll("");
            Matcher m = HVAC_STOP2.matcher(v);
            while (m.find()) {
                String noun = m.group(1);
                boolean bad;
                if (noun.equals("風") || noun.equals("送風")) {
                    Object f = values.get("fan_level");
                    bad = !"off".equals(mode) && !(f == null || num(f) == 0);
                } else {
                    bad = stopTarget(noun, mode);
                }
                if (bad) return "P26:G5";
            }
            if ("off".equals(mode) && Py.find(NEGATED, u)) return "P26:G5";
            if (!values.containsKey("mode") && Py.find(MODE_ON, u)) return "P26:G10";
        }
        if (MEDIA.equals(tool) && "play".equals(values.get("action"))) {
            if (Py.find(MEDIA_STOP, MEDIA_NEGATED.matcher(u).replaceAll(""))) return "P26:G6";
        }
        if (HVAC.equals(tool) || MEDIA.equals(tool)) {
            for (String key : NUMERIC) {
                if (!values.containsKey(key) || values.get(key) == null || !tool.equals(key.equals("volume") ? MEDIA : HVAC)) continue;
                Double now = state(st, key);
                double got = num(values.get(key));
                double step = key.equals("volume") ? 2.0 : 1.0;
                if (now != null && Py.find(EXTREME, u) && near(Math.abs(got - now), step)) return "P26:G7";
                if (now != null && Py.find(HALF_OF, u) && Math.abs(got - now / 2) > step / 2 + 1e-9) return "P26:G7";
                Matcher h = HALF.matcher(u);
                if (key.equals("temperature") && h.find() && Math.abs(got - (Double.parseDouble(h.group(1)) + 0.5)) > 1e-9) return "P26:G7";
                for (int i = 0; i < KANA_W.length; i++) {
                    if (u.contains(KANA_W[i]) && now != null && Math.abs(Math.abs(got - now) - KANA_N[i]) > 1e-9 && Math.abs(got - KANA_N[i]) > 1e-9) return "P26:G8";
                }
            }
        }
        if (NAV.equals(tool) && "set".equals(values.get("action")) && values.get("destination") != null && !String.valueOf(values.get("destination")).isEmpty()
                && !Py.find(VIA_THEN, u)) {
            String d = String.valueOf(values.get("destination"));
            Matcher m = PLACE_MOVE.matcher(u);
            while (m.find()) {
                String g = m.group(1);
                if (g.contains(d) || d.contains(g)) continue;
                boolean purpose = false;
                for (String w : PURPOSE) purpose |= g.contains(w);
                if (!purpose) return "P26:G9";
            }
        }
        return null;
    }
}
