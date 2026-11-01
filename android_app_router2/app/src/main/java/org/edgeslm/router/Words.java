package org.edgeslm.router;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 発話を単語に分け、語が単語の切れ目に合うかを見る（P30 の `p30_words`。P36 でアプリに入れた）。
 *
 * 単語の分け方は、端末の BERT の入力作りと同じ {@link BertJa#words}（NFKC ＋ MeCab ＋ unidic-lite）。PC の Python は、同じ `BertJa` で分けた表
 * （`data/p30_words/words.tsv`）を引く。アプリは {@link #tok} に、BERT の入力作りで読んだ {@link BertJa} を入れる（画面を開くときに 1 回）。
 * PC の確かめ（`tools/p36_java`）は、{@link #tok} か {@link #table} のどちらかを入れる。
 */
public final class Words {
    private Words() {}

    /** 単語に分ける道具（アプリ・PC の確かめで入れる） */
    public static volatile BertJa tok;
    /** 発話 → 単語の表（PC の確かめで、Python と同じ表を使うときだけ入れる。null ならいつも {@link #tok}） */
    public static volatile Map<String, List<String>> table;

    private static final Map<String, List<String>> CACHE = Collections.synchronizedMap(new LinkedHashMap<String, List<String>>(64, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, List<String>> e) {
            return size() > 256;
        }
    });

    public static String nfkc(String s) {
        return Normalizer.normalize(s, Normalizer.Form.NFKC);
    }

    /** NFKC 後の単語 */
    public static List<String> words(String u) {
        Map<String, List<String>> t = table;
        if (t != null && t.containsKey(u)) return t.get(u);
        List<String> w = CACHE.get(u);
        if (w != null) return w;
        BertJa b = tok;
        if (b == null) throw new IllegalStateException("Words.tok が無い（単語に分けられない）: " + u);
        w = Collections.unmodifiableList(new ArrayList<String>(b.words(u)));
        CACHE.put(u, w);
        return w;
    }

    /** NFKC 後の文字列と、単語の始まり・終わりの位置（`p30_words.bounds`） */
    static final class Bounds {
        final String norm;
        final boolean[] start, end;

        Bounds(String norm) {
            this.norm = norm;
            start = new boolean[norm.length() + 1];
            end = new boolean[norm.length() + 1];
        }
    }

    private static final Map<String, Bounds> BOUNDS = Collections.synchronizedMap(new HashMap<String, Bounds>());

    static Bounds bounds(String u) {
        Bounds b = BOUNDS.get(u);
        if (b != null) return b;
        b = new Bounds(nfkc(u));
        int cur = 0;
        for (String w : words(u)) {
            int i = b.norm.indexOf(w, cur);
            if (i < 0) throw new IllegalStateException("単語の位置が合わない: " + u + " / " + w);
            b.start[i] = true;
            b.end[i + w.length()] = true;
            cur = i + w.length();
        }
        if (BOUNDS.size() > 512) BOUNDS.clear();
        BOUNDS.put(u, b);
        return b;
    }

    /** term が、単語の切れ目から始まり、単語の切れ目で終わる位置にあるか（stem = true なら始まりだけ合えばよい） */
    public static boolean aligned(String term, String u, boolean stem) {
        Bounds b = bounds(u);
        String t = nfkc(term);
        int i = b.norm.indexOf(t);
        while (i >= 0) {
            if (b.start[i] && (stem || b.end[i + t.length()])) return true;
            i = b.norm.indexOf(t, i + 1);
        }
        return false;
    }
}
