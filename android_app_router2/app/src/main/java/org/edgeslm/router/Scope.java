package org.edgeslm.router;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * 読み出しの機能を、機能の定義で説明がつく依頼だけに絞る（P36 でアプリに入れた。Python の `p30_scope`・`p31_scope`・`p33_scope`・`p35_scope`）。
 *
 * <ul>
 * <li>{@link #evidence31}: **何について**の依頼か（機能ごとの語の一覧を、単語の切れ目に合わせて見る）</li>
 * <li>{@link #explains33}・{@link #explains35}: **何を聞いているか**（教えて・調べて の目的語・「〜があるか」の主語・「いくら」）が、機能の返すもので説明がつくか</li>
 * </ul>
 * 語の一覧は {@link Lex}（Python から機械的に書き出したもの）。
 */
public final class Scope {
    private Scope() {}

    static Set<String> set(String... xs) {
        return new HashSet<String>(Arrays.asList(xs));
    }

    static final Set<String> PREFIX_1 = set(Lex.PREFIX_1);
    static final Set<String> READ = set(Lex.READ);

    static boolean in(String x, String[] xs) {
        for (String y : xs) if (y.equals(x)) return true;
        return false;
    }

    static boolean startsWithAny(String t, String[] xs) {
        for (String y : xs) if (t.startsWith(y)) return true;
        return false;
    }

    /** `p30_scope.has` が空でないか（2 文字以上と PREFIX_1 は単語の始まりに合えばよい・ほかの 1 文字は単語そのもの） */
    static boolean has30(String[] terms, String u) {
        for (String w : terms) {
            if (Words.aligned(w, u, w.length() >= 2 || PREFIX_1.contains(w))) return true;
        }
        return false;
    }

    // ---------------------------------------------------------------- P31（何について）

    static String[] lex31(String tool) {
        if ("get_vehicle_status".equals(tool)) return Lex.P31_STATUS;
        if ("search_vehicle_manual".equals(tool)) return Lex.P31_MANUAL;
        if ("capture_camera".equals(tool)) return Lex.P31_CAMERA;
        if ("get_media_info".equals(tool)) return Lex.P31_MEDIA_INFO;
        if ("search_poi".equals(tool)) return Lex.P31_POI;
        return null;
    }

    /** `p31_scope.evidence` が空でないか（操作の機能は見ないので true） */
    public static boolean evidence31(String tool, String u) {
        String[] lex = lex31(tool);
        return lex == null || has30(lex, u);
    }

    // ---------------------------------------------------------------- P33（何を聞いているか）

    static final Set<String> FUNC = set(Lex.P33_FUNC), HIRA_NOUN = set(Lex.P33_HIRA_NOUN), Q_TAIL = set(Lex.P33_Q_TAIL), PART = set(Lex.P33_PART);
    static final Pattern VERBISH = Pattern.compile("(?:る|た|て|だ|い|ない|れる|せる|たい|ます|う|く|す|ず|っ|ぬ|れ|ろ|ば)$");
    static final Pattern HIRA_ONLY = Pattern.compile("[ぁ-ん]+");
    static final Pattern HIRA_END = Pattern.compile("[ぁ-ん]$");

    static boolean nounish(String t) {
        if (HIRA_NOUN.contains(t)) return true;
        if (FUNC.contains(t) || Py.strip(t).isEmpty()) return false;
        if (HIRA_ONLY.matcher(t).matches()) return false;
        return !(HIRA_END.matcher(t).find() && VERBISH.matcher(t).find());
    }

    static List<String> chunkBefore(List<String> ws, int j) {
        int k = j;
        while (k >= 0 && nounish(ws.get(k))) k--;
        return new ArrayList<String>(ws.subList(k + 1, j + 1));
    }

    /** 聞いているもの（`p33_scope.asked`）。kind = money・exist・yn・noun・where。取り出せなければ null */
    static final class Asked {
        final String kind;
        final List<String> c;

        Asked(String kind, List<String> c) {
            this.kind = kind;
            this.c = c;
        }
    }

    static Asked asked(String tool, String u) {
        List<String> ws = Words.words(u);
        if (ws.contains("いくら")) return new Asked("money", Arrays.asList("いくら"));
        for (int v = 0; v < ws.size(); v++) {
            String t = ws.get(v);
            if (!startsWithAny(t, Lex.P33_REQ) || v == 0) continue;
            int p = v - 1;
            if ((ws.get(p).equals("お") || ws.get(p).equals("ご")) && p > 0) p--;
            if (ws.get(p).equals("か")) {
                if (!"search_poi".equals(tool)) return null;
                int q = p - 1;
                while (q >= 0 && in(ws.get(q), new String[]{"の", "ん", "て", "いる", "か"})) q--;
                if (q < 0) return null;
                String pred = ws.get(q);
                if (startsWithAny(pred, Lex.P33_EXIST)) {
                    int ga = -1;
                    for (int i = q - 1; i >= 0; i--) {
                        if (in(ws.get(i), new String[]{"が", "は", "も", "って"})) {
                            ga = i;
                            break;
                        }
                    }
                    int s = ga >= 0 ? ga - 1 : q - 1;
                    return s >= 0 ? new Asked("exist", chunkBefore(ws, s)) : null;
                }
                return new Asked("yn", Arrays.asList(pred));
            }
            if (PART.contains(ws.get(p))) p--;
            if (p >= 0) {
                List<String> c = chunkBefore(ws, p);
                if (!c.isEmpty()) return new Asked("noun", c);
            }
            return null;
        }
        for (int p = ws.size() - 1; p > 0; p--) {
            if (ws.get(p).equals("は") || ws.get(p).equals("って")) {
                List<String> tail = ws.subList(p + 1, ws.size());
                boolean all = true;
                for (String t : tail) all &= Q_TAIL.contains(t);
                if (all) {
                    if (tail.contains("どこ")) return new Asked("where", new ArrayList<String>(tail));
                    List<String> c = chunkBefore(ws, p - 1);
                    return c.isEmpty() ? null : new Asked("noun", c);
                }
                break;
            }
        }
        return null;
    }

    /** 語の一覧の語が、聞いているものの単語の切れ目から始まるか（1 文字の語は単語の終わりにも合う）。`_match` の 1 つ目 */
    static boolean matchTerms(String[] terms, List<String> c) {
        StringBuilder sb = new StringBuilder();
        Set<Integer> starts = new HashSet<Integer>();
        for (String t : c) {
            starts.add(sb.length());
            sb.append(t);
        }
        String s = sb.toString();
        for (String w : terms) {
            int i = s.indexOf(w);
            while (i >= 0) {
                if (starts.contains(i) && (w.length() >= 2 || starts.contains(i + 1) || i + 1 == s.length())) return true;
                i = s.indexOf(w, i + 1);
            }
        }
        return false;
    }

    static boolean lastIn(String t, String[] suffix) {
        return t.length() >= 1 && in(t.substring(t.length() - 1), suffix);
    }

    /** `p33_scope._match` が空でないか */
    static boolean match33(String[] terms, String[] suffix, List<String> c) {
        if (matchTerms(terms, c)) return true;
        for (String t : c) {
            for (String sfx : suffix) {
                if (t.endsWith(sfx) && lastIn(t, suffix)) return true;
            }
        }
        return false;
    }

    static String[] concat(String[] a, String[] b) {
        String[] x = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, x, a.length, b.length);
        return x;
    }

    static final String[] RET33_POI = concat(Lex.P31_POI, Lex.P33_POI_RET), RET33_MANUAL = concat(Lex.P31_MANUAL, Lex.P33_MANUAL_RET);

    /** `p33_scope.explains` の可否（取り出せなければ true） */
    public static boolean explains33(String tool, String u) {
        String[] terms, suffix;
        if ("search_poi".equals(tool)) {
            terms = RET33_POI;
            suffix = Lex.P33_POI_SUFFIX;
        } else if ("search_vehicle_manual".equals(tool)) {
            terms = RET33_MANUAL;
            suffix = Lex.P33_MANUAL_SUFFIX;
        } else {
            return true;
        }
        Asked a = asked(tool, u);
        if (a == null) return true;
        if (a.kind.equals("money")) return false;
        if (a.kind.equals("where")) return true;
        if (a.kind.equals("yn")) return startsWithAny(a.c.get(0), Lex.P33_HOURS);
        if (a.kind.equals("exist")) return match33(Lex.P31_POI, Lex.P33_POI_SUFFIX, a.c);
        return match33(terms, suffix, a.c);
    }

    // ---------------------------------------------------------------- P35（返すものを 4 つの機能へ・施設の末尾 1 文字）

    static final String[] RET35_POI = concat(Lex.P31_POI, Lex.P33_POI_RET), RET35_MANUAL = concat(Lex.P31_MANUAL, Lex.P33_MANUAL_RET),
            RET35_STATUS = concat(Lex.P31_STATUS, Lex.P35_STATUS_RET), RET35_MINFO = concat(Lex.P31_MEDIA_INFO, Lex.P35_MINFO_RET);
    static final String[] NONE = new String[0];

    /** `p35_scope._match` が空でないか */
    static boolean match35(String[] terms, String[] suffix, List<String> c) {
        if (matchTerms(terms, c)) return true;
        for (String t : c) {
            if (t.length() >= 2 && lastIn(t, suffix)) return true;
        }
        boolean strong = suffix == Lex.P35_STRONG_SUFFIX;
        for (String t : c) {
            if (in(t, suffix) || (in(t, Lex.P35_WEAK_SUFFIX) && strong)) return true;
        }
        return false;
    }

    /** `p35_scope.explains` の可否（取り出せなければ true） */
    public static boolean explains35(String tool, String u) {
        String[] terms, suffix;
        if ("search_poi".equals(tool)) {
            terms = RET35_POI;
            suffix = Lex.P35_STRONG_SUFFIX;
        } else if ("search_vehicle_manual".equals(tool)) {
            terms = RET35_MANUAL;
            suffix = Lex.P33_MANUAL_SUFFIX;
        } else if ("get_vehicle_status".equals(tool)) {
            terms = RET35_STATUS;
            suffix = NONE;
        } else if ("get_media_info".equals(tool)) {
            terms = RET35_MINFO;
            suffix = NONE;
        } else {
            return true;
        }
        Asked a = asked(tool, u);
        if (a == null) return true;
        if (a.kind.equals("money")) return false;
        if (a.kind.equals("where")) return true;
        if (a.kind.equals("yn")) return startsWithAny(a.c.get(0), Lex.P33_HOURS);
        if (a.kind.equals("exist")) return match35(Lex.P31_POI, Lex.P35_STRONG_SUFFIX, a.c);
        return match35(terms, suffix, a.c);
    }
}
