package org.edgeslm.router;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * いまの引数の処理（P36 でアプリに入れた）。{@link Args}（P23 まで）の上に、Python と同じ順で重ねる:
 *
 * <ol>
 * <li>`p25_arguments`・`p26_arguments`: 歯止め G1〜G10（{@link Guard}）。止めたら「引数が足りない（value_check）」</li>
 * <li>`p31_arguments`: 対象の言い方の取りこぼし F1〜F6・**何についての語で説明がつかない読み出しは断る**（値に `__scope__`）</li>
 * <li>`p33_arguments`: E1 最大・最小の語を単語の切れ目で見る・E2「途中に」・**何を聞いているかで断る**（周辺の検索・説明書）</li>
 * <li>`p35_arguments`: E3 止める語の後ろに別の操作があれば止める語を伏せる・**何を聞いているかで断る**（4 つの読み出しの機能）</li>
 * </ol>
 * 断るとき（`__scope__`）は、検査（{@link Policy#inspect}）が「その指定には対応していません」と断る。
 * Python と同じ結果になるかは、`tools/p36_java` の `ParityTest2` が `work/p36/fixtures/args.tsv` で確かめる。
 */
public final class Args2 {
    private Args2() {}

    static final String HVAC = "control_hvac", MEDIA = "control_media", NAV = "set_navigation";
    static final String WS = "[" + Py.WS + "]";

    static boolean write(String tool) {
        return HVAC.equals(tool) || MEDIA.equals(tool) || NAV.equals(tool);
    }

    static Args.Arguments copy(Args.Arguments a) {
        Args.Arguments b = new Args.Arguments();
        b.values.putAll(a.values);
        b.missing.addAll(a.missing);
        b.note = a.note;
        b.guard = a.guard;
        return b;
    }

    static Args.Arguments ask(Args.Arguments a, String guard) {
        Args.Arguments b = new Args.Arguments();
        b.values.putAll(a.values);
        b.missing.add("value_check");
        b.guard = guard;
        return b;
    }

    static Args.Arguments scope(Args.Arguments a) {
        Args.Arguments b = new Args.Arguments();
        b.values.putAll(a.values);
        b.values.put("__scope__", Boolean.TRUE);
        return b;
    }

    // ---------------------------------------------------------------- P25・P26

    static Args.Arguments p25(String tool, String u, Args.State st) {
        Args.Arguments out = Args.resolve(tool, u, st);
        if (!write(tool) || !out.complete()) return out;
        String g = Guard.check25(tool, u, out.values, out.note, st);
        return g == null ? out : ask(out, g);
    }

    static Args.Arguments p26(String tool, String u, Args.State st) {
        Args.Arguments out = p25(tool, u, st);
        if (!write(tool) || !out.complete()) return out;
        String g = Guard.check26(tool, u, out.values, st);
        return g == null ? out : ask(out, g);
    }

    // ---------------------------------------------------------------- P31

    static final Pattern ZERO_VOL = Pattern.compile("(?:音量|ボリューム|音)(?:を|は|も)?(?:ゼロ|0|０|零)");
    static final Pattern REROUTE = Pattern.compile("(?:早く着|はやく着|空いて|すいて|混んでな|近い|速い|短い)[^、。]{0,6}(?:道|ルート|経路)|(?:道|ルート|経路)(?:に|へ)(?:切り替え|切りかえ|変え)|(?:使わ|通ら|乗ら)ず|避けて");
    static final Pattern VERB_FRAG = Pattern.compile("(?:ず|ずに|て|って|れて|せて)$");
    static final Pattern TE_NOUN = Pattern.compile("^.*[ぁ-ん][てで]([一-龥ァ-ヴーA-Za-z][^てで]*)$");
    static final Pattern DIGITS = Pattern.compile("\\p{Nd}+");
    static final Pattern COUNT_ONLY = Pattern.compile("(\\p{Nd}+)" + WS + "*(?:曲|回|つ前|つ先|つ後|つ戻|分|秒)");
    static final Pattern REPLAY = Pattern.compile("(?:止め|停止|ポーズ|ストップ)[^、。]{0,8}(?:再び|再開|続き|流して|流す|かけて|かける|もう一度流)");
    static final Pattern G2_TRACK = Pattern.compile("\\p{Nd}+" + WS + "*つ" + WS + "*(?:飛ばし|先|進め|戻)");
    static final Pattern ONCE = Pattern.compile("一度|一回|1度|1回");

    static boolean valueCheckOnly(Args.Arguments a) {
        return a.missing.size() == 1 && "value_check".equals(a.missing.get(0));
    }

    /** P31 の直し F1〜F6。直さないなら null */
    static Args.Arguments fix31(String tool, String u, Args.Arguments out) {
        if (MEDIA.equals(tool)) {
            Object vol = out.values.get("volume");
            if (Py.find(ZERO_VOL, u) && (!out.complete() || !(vol == null || Guard.num(vol) == 0))) {
                Args.Arguments b = new Args.Arguments();
                b.values.put("action", "volume_set");
                b.values.put("volume", Integer.valueOf(0));
                return b;
            }
            if (valueCheckOnly(out) && "P25:G2".equals(out.guard)
                    && (Py.find(G2_TRACK, u) || (!"volume_set".equals(out.values.get("action")) && Py.find(ONCE, u)))) {
                Args.Arguments b = new Args.Arguments();
                b.values.putAll(out.values);
                return b;
            }
            if (out.complete() && "volume_set".equals(out.values.get("action"))) {
                List<String> nums = new ArrayList<String>();
                Matcher m = DIGITS.matcher(u);
                while (m.find()) nums.add(m.group());
                List<String> counted = new ArrayList<String>();
                Matcher c = COUNT_ONLY.matcher(u);
                while (c.find()) counted.add(c.group(1));
                if (!nums.isEmpty() && counted.containsAll(nums) && counted.contains(String.valueOf(out.values.get("volume")))) {
                    Args.Arguments b = new Args.Arguments();
                    b.values.putAll(out.values);
                    b.missing.add("value_check");
                    return b;
                }
            }
            if (out.complete() && "pause".equals(out.values.get("action")) && Py.find(REPLAY, u)) {
                Args.Arguments b = new Args.Arguments();
                b.values.putAll(out.values);
                b.missing.add("value_check");
                return b;
            }
        }
        if (HVAC.equals(tool)) {
            if (valueCheckOnly(out) && "P25:G1:auto".equals(out.guard) && u.contains("自動")) {
                Args.Arguments b = new Args.Arguments();
                b.values.putAll(out.values);
                return b;
            }
        }
        if (NAV.equals(tool)) {
            Object dv = out.values.get("destination");
            String dest = dv == null ? "" : String.valueOf(dv);
            if (Py.find(REROUTE, u) && (!out.complete() || Py.find(VERB_FRAG, dest) || dest.isEmpty())) {
                Args.Arguments b = new Args.Arguments();
                b.values.put("action", "reroute");
                return b;
            }
            Object act = out.values.get("action");
            if (out.complete() && ("set".equals(act) || "add_waypoint".equals(act)) && !dest.isEmpty()) {
                Matcher m = TE_NOUN.matcher(dest);
                if (m.matches() && m.group(1).length() >= 1) {
                    Args.Arguments b = new Args.Arguments();
                    b.values.putAll(out.values);
                    b.values.put("destination", m.group(1));
                    return b;
                }
            }
        }
        return null;
    }

    static Args.Arguments p31(String tool, String u, Args.State st) {
        Args.Arguments out = p26(tool, u, st);
        Args.Arguments fixed = fix31(tool, u, out);
        if (fixed != null) out = fixed;
        if (Scope.READ.contains(tool) && out.complete() && !Scope.evidence31(tool, u)) return scope(out);
        return out;
    }

    // ---------------------------------------------------------------- P33

    static final Pattern ON_THE_WAY = Pattern.compile("^.*(?:途中に|途中で|ついでに)(.+)$");

    /** E1: 最大・最小の語が部分一致でしか無いとき、その部分を伏せた発話（伏せる所が無い・単語に分けられないなら null） */
    static String mask1(String u) {
        List<String> hit = new ArrayList<String>();
        for (String t : Args.MAX_WORDS) if (u.contains(t)) hit.add(t);
        for (String t : Args.MIN_WORDS) if (u.contains(t)) hit.add(t);
        if (hit.isEmpty()) return null;
        try {
            for (String t : hit) if (Words.aligned(t, u, true)) return null;
        } catch (IllegalStateException e) {
            return null;                                  // Python は単語に分けられないと伏せない（SystemExit）
        }
        List<String> order = new ArrayList<String>(hit);
        Collections.sort(order, new Comparator<String>() {   // 長い順（同じ長さは元の順。Python の sorted と同じく安定）
            @Override
            public int compare(String a, String b) {
                return b.length() - a.length();
            }
        });
        String v = u;
        for (String t : order) v = v.replace(t, repeat(t.length()));
        return v;
    }

    static String repeat(int n) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < n; i++) b.append('〇');
        return b.toString();
    }

    static Args.Arguments p33(String tool, String u, Args.State st) {
        if (MEDIA.equals(tool) || HVAC.equals(tool)) {
            String m = mask1(u);
            if (m != null) return copy(p31(tool, m, st));
        }
        Args.Arguments out = p31(tool, u, st);
        Object act = out.values.get("action");
        if (NAV.equals(tool) && out.complete() && ("set".equals(act) || "add_waypoint".equals(act))) {
            Object dv = out.values.get("destination");
            String d = dv == null ? "" : String.valueOf(dv);
            Matcher mm = ON_THE_WAY.matcher(d);
            if (mm.matches() && !mm.group(1).isEmpty()) {
                Args.Arguments b = copy(out);
                b.values.put("destination", mm.group(1));
                return b;
            }
        }
        if (out.complete() && !out.values.containsKey("__scope__") && !Scope.explains33(tool, u)) return scope(out);
        return out;
    }

    // ---------------------------------------------------------------- P35

    static final Pattern KEEP = Pattern.compile("(?:曇り|くもり)止め");

    /** E3: 止める語の後ろに別の操作の語があれば、止める語を伏せた発話（伏せる所が無ければ null） */
    static String maskStop(String tool, String u) {
        String[] stops = MEDIA.equals(tool) ? Lex.P35_M_STOP : Lex.P35_H_STOP;
        String[] goes = MEDIA.equals(tool) ? Lex.P35_M_PLAY : Lex.P35_H_ON;
        boolean[] keep = new boolean[u.length() + 1];
        Matcher km = KEEP.matcher(u);
        while (km.find()) for (int k = km.start(); k < km.end(); k++) keep[k] = true;
        List<int[]> pos = new ArrayList<int[]>();
        for (String w : stops) {
            int i = u.indexOf(w);
            while (i >= 0) {
                if (!keep[i]) pos.add(new int[]{i, w.length()});
                i = u.indexOf(w, i + w.length());
            }
        }
        if (pos.isEmpty()) return null;
        int lastGo = -1;
        for (String w : goes) {
            int i = u.indexOf(w);
            while (i >= 0) {
                lastGo = Math.max(lastGo, i);
                i = u.indexOf(w, i + w.length());
            }
        }
        for (int[] p : pos) if (p[0] >= lastGo) return null;     // 後ろに操作が無い止める語が 1 つでもあれば伏せない
        char[] v = u.toCharArray();
        for (int[] p : pos) for (int k = p[0]; k < p[0] + p[1]; k++) v[k] = '〇';
        return new String(v);
    }

    static Args.Arguments p35(String tool, String u, Args.State st) {
        if (MEDIA.equals(tool) || HVAC.equals(tool)) {
            String m = maskStop(tool, u);
            if (m != null) return copy(p33(tool, m, st));
        }
        Args.Arguments out = p33(tool, u, st);
        if (out.complete() && !out.values.containsKey("__scope__") && !Scope.explains35(tool, u)) return scope(out);
        return out;
    }

    /** いまの引数の処理（P35）。履歴の無い 1 回目の発話に使う */
    public static Args.Arguments resolve(String tool, String utterance, Args.State st) {
        return p35(tool, utterance, st);
    }

    static final List<String> UNUSED = Arrays.asList();     // （Java 8 の範囲の機能だけを使う）
}
