package org.edgeslm.router;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Python の文字列・正規表現の動きを、そのまま写すための小さな道具（P24）。
 *
 * 規則と引数の取り出しは、PC では Python で書いてある。アプリ（Java）は、同じ結果を返さないといけない。
 * Python と Java で動きが違う所（空白の扱い・strip・split・sub）を、ここにまとめて Python 側に合わせる。
 * Android API 26 で使える Java の機能だけを使う（String.strip・List.of などは使わない）。
 */
final class Py {
    private Py() {}

    /** Python の `\s`（文字列）に当たる文字クラスの中身。Java の `\s` は ASCII だけ、Android は Unicode、と違うので、明示する */
    static final String WS = "\\s\\u00A0\\u3000";

    /** any(w in s for w in words) */
    static boolean any(String s, String... words) {
        for (String w : words) {
            if (s.contains(w)) return true;
        }
        return false;
    }

    static boolean find(Pattern p, String s) {
        return p.matcher(s).find();
    }

    /** s.strip(chars) */
    static String strip(String s, String chars) {
        int a = 0, b = s.length();
        while (a < b && chars.indexOf(s.charAt(a)) >= 0) a++;
        while (b > a && chars.indexOf(s.charAt(b - 1)) >= 0) b--;
        return s.substring(a, b);
    }

    /** s.lstrip(chars) */
    static String lstrip(String s, String chars) {
        int a = 0;
        while (a < s.length() && chars.indexOf(s.charAt(a)) >= 0) a++;
        return s.substring(a);
    }

    static boolean isSpace(char c) {
        return Character.isWhitespace(c) || c == ' ' || c == '　' || c == '\u0085';
    }

    /** s.strip()（空白。全角の空白も落とす） */
    static String strip(String s) {
        int a = 0, b = s.length();
        while (a < b && isSpace(s.charAt(a))) a++;
        while (b > a && isSpace(s.charAt(b - 1))) b--;
        return s.substring(a, b);
    }

    /** s.endswith((a, b, ...)) */
    static boolean endsWithAny(String s, String... suffixes) {
        for (String x : suffixes) {
            if (s.endsWith(x)) return true;
        }
        return false;
    }

    /** re.split(pattern, s)（幅 0 の一致でも分ける。Python 3.7 以降と同じ） */
    static List<String> split(Pattern p, String s) {
        List<String> out = new ArrayList<String>();
        Matcher m = p.matcher(s);
        int last = 0;
        while (m.find()) {
            if (m.end() == m.start() && (m.start() == 0 || m.start() == s.length())) continue;   // Python は、端の空の一致では分けた結果に空文字を足す。使う所では空の節を捨てるので、ここでは飛ばす
            out.add(s.substring(last, m.start()));
            last = m.end();
        }
        out.add(s.substring(last));
        return out;
    }

    /** 全角の数字と小数点を半角にする（str.translate(ZEN)） */
    static String zenToHan(String s) {
        StringBuilder b = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= '０' && c <= '９') b.append((char) ('0' + (c - '０')));
            else if (c == '．') b.append('.');
            else b.append(c);
        }
        return b.toString();
    }

    /** Python の repr(float) と同じ見え方（24.0・24.5） */
    static String reprDouble(double v) {
        if (v == Math.rint(v) && Math.abs(v) < 1e15) return String.valueOf((long) v) + ".0";
        return Double.toString(v);
    }
}
