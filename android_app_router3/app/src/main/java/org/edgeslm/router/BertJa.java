package org.edgeslm.router;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * 日本語 BERT（`cl-tohoku/bert-base-japanese-v3`）の入力を作る（P24）。PC の `BertJapaneseTokenizer` と同じ ID 列を返す。
 *
 * PC は、1) NFKC でそろえる 2) MeCab（辞書は unidic-lite）で単語に分ける 3) 単語を WordPiece で細かく分ける、の順。
 * 端末には MeCab が無いので、MeCab の単語分け（辞書引き・未知語・最小コストの道）を、ここに Java で書いた。
 * 辞書は、unidic-lite の中身をそのまま使う（`sys_lex.bin` は `sys.dic` の先頭＝見出しと単語の表だけを切り出したもの。品詞などの文字列は使わない）。
 *
 * 同じ ID 列になるかは、PC で `BertJapaneseTokenizer` が出した列と突き合わせて確かめる（`TokTest`・設計書 P24 の 5 節）。
 *
 * 辞書は、ファイルをメモリに対応づけて読む（Java のヒープを使わない。`matrix.bin` が 71MB あるため）。
 */
public final class BertJa {
    /** MeCab の既定（`max-grouping-size`。辞書の dicrc は指定していない） */
    static final int MAX_GROUPING = 24;
    static final int MAX_CHARS_PER_WORD = 100;      // WordPiece の `max_input_chars_per_word`

    /** 辞書（sys・unk 共通の形）: 見出しの表（Darts の double array）と、単語の表（1 語 16 バイト） */
    static final class Dic {
        final ByteBuffer darts, tokens;
        final int lsize, rsize;

        Dic(ByteBuffer file) {
            file.order(ByteOrder.LITTLE_ENDIAN);
            lsize = file.getInt(16);
            rsize = file.getInt(20);
            int dsize = file.getInt(24), tsize = file.getInt(28);
            darts = slice(file, 72, dsize);
            tokens = slice(file, 72 + dsize, tsize);
        }

        static ByteBuffer slice(ByteBuffer b, int from, int n) {
            ByteBuffer d = b.duplicate();
            d.position(from);
            d.limit(from + n);
            return d.slice().order(ByteOrder.LITTLE_ENDIAN);
        }

        int base(int i) {
            return darts.getInt(i * 8);
        }

        int check(int i) {
            return darts.getInt(i * 8 + 4);
        }

        /** Darts の commonPrefixSearch。out に（値・長さ）を入れ、個数を返す */
        int prefix(byte[] key, int from, int end, int[] outValue, int[] outLen) {
            int b = base(0), num = 0, units = darts.capacity() / 8;
            for (int i = from; i < end; i++) {
                int n = base(b);
                if (b == check(b) && n < 0) {
                    if (num < outValue.length) {
                        outValue[num] = -n - 1;
                        outLen[num] = i - from;
                    }
                    num++;
                }
                int p = b + (key[i] & 0xff) + 1;
                if (p < 0 || p >= units) return Math.min(num, outValue.length);
                if (b == check(p)) b = base(p);
                else return Math.min(num, outValue.length);
            }
            int n = base(b);
            if (b == check(b) && n < 0) {
                if (num < outValue.length) {
                    outValue[num] = -n - 1;
                    outLen[num] = end - from;
                }
                num++;
            }
            return Math.min(num, outValue.length);
        }

        /** Darts の exactMatchSearch。無ければ -1 */
        int exact(byte[] key) {
            int b = base(0), units = darts.capacity() / 8;
            for (byte k : key) {
                int p = b + (k & 0xff) + 1;
                if (p < 0 || p >= units || b != check(p)) return -1;
                b = base(p);
            }
            int n = base(b);
            return (b == check(b) && n < 0) ? -n - 1 : -1;
        }

        int lc(int t) {
            return tokens.getShort(t * 16) & 0xffff;
        }

        int rc(int t) {
            return tokens.getShort(t * 16 + 2) & 0xffff;
        }

        int wcost(int t) {
            return tokens.getShort(t * 16 + 6);
        }
    }

    final Dic sys, unk;
    final ByteBuffer matrix;        // 先頭 4 バイト（lsize・rsize）の後ろに、short が lsize × rsize
    final int lsize;
    final int[] charMap = new int[0xffff];      // 文字 → 種類の情報（MeCab の CharInfo。32 ビット）
    final int[] unkFirst, unkCount;             // 文字の種類 → 未知語の単語（unk.dic の中の位置と数）
    final int spaceInfo;
    final HashMap<String, Integer> vocab = new HashMap<String, Integer>();
    final int clsId, sepId, unkId;

    static ByteBuffer map(File f) throws IOException {
        RandomAccessFile r = new RandomAccessFile(f, "r");
        try {
            return r.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, r.length()).order(ByteOrder.LITTLE_ENDIAN);
        } finally {
            r.close();
        }
    }

    BertJa(File dir) throws IOException {
        sys = new Dic(map(new File(dir, "sys_lex.bin")));
        unk = new Dic(map(new File(dir, "unk.dic")));
        matrix = map(new File(dir, "matrix.bin"));
        lsize = matrix.getShort(0) & 0xffff;
        int rsize = matrix.getShort(2) & 0xffff;
        if (lsize != sys.lsize || rsize != sys.rsize) throw new IOException("matrix.bin と辞書の大きさが合わない: " + lsize + "x" + rsize + " / " + sys.lsize + "x" + sys.rsize);
        if (matrix.capacity() != 4 + 2L * lsize * rsize) throw new IOException("matrix.bin の大きさが違う: " + matrix.capacity());
        ByteBuffer c = map(new File(dir, "char.bin"));
        int n = c.getInt(0);
        if (c.capacity() != 4 + 32 * n + 4 * 0xffff) throw new IOException("char.bin の大きさが違う: " + c.capacity());
        unkFirst = new int[n];
        unkCount = new int[n];
        for (int i = 0; i < n; i++) {
            int len = 0;
            while (len < 32 && c.get(4 + 32 * i + len) != 0) len++;
            byte[] name = new byte[len];
            for (int k = 0; k < len; k++) name[k] = c.get(4 + 32 * i + k);
            int v = unk.exact(name);
            if (v < 0) throw new IOException("unk.dic に文字の種類が無い: " + new String(name, "UTF-8"));
            unkFirst[i] = v >>> 8;
            unkCount[i] = v & 0xff;
        }
        for (int i = 0; i < 0xffff; i++) charMap[i] = c.getInt(4 + 32 * n + 4 * i);
        spaceInfo = charMap[0x20];
        BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(new File(dir, "vocab.txt")), "UTF-8"));
        try {
            int id = 0;
            for (String line; (line = r.readLine()) != null; id++) vocab.put(line, id);
        } finally {
            r.close();
        }
        clsId = need("[CLS]");
        sepId = need("[SEP]");
        unkId = need("[UNK]");
    }

    int need(String token) throws IOException {
        Integer v = vocab.get(token);
        if (v == null) throw new IOException("vocab.txt に " + token + " が無い");
        return v;
    }

    /** `dir` に sys_lex.bin・unk.dic・matrix.bin・char.bin・vocab.txt */
    public static BertJa load(File dir) throws IOException {
        return new BertJa(dir);
    }

    // ---------------------------------------------------------------- 文字の種類（MeCab の CharProperty）

    static int type(int info) {
        return info & 0x3ffff;
    }

    static int defaultType(int info) {
        return (info >>> 18) & 0xff;
    }

    static int length(int info) {
        return (info >>> 26) & 0xf;
    }

    static boolean group(int info) {
        return ((info >>> 30) & 1) != 0;
    }

    static boolean invoke(int info) {
        return (info >>> 31) != 0;
    }

    static boolean kindOf(int a, int b) {
        return (type(a) & type(b)) != 0;
    }

    /** 1 回の単語分けの作業場（MeCab の Lattice に当たる） */
    final class Lattice {
        final byte[] s;
        final int end;
        int mblen, clen, info;      // 直前に見た文字のバイト数・数えた文字数・種類
        // ノード（配列の同じ位置が 1 つのノード）
        int[] lc = new int[256], rc = new int[256], wcost = new int[256], cost = new int[256], prev = new int[256], bnext = new int[256], enext = new int[256],
                begin = new int[256], len = new int[256], rlen = new int[256];
        int count = 0;
        final int[] endList;
        final int[] daValue = new int[512], daLen = new int[512];

        Lattice(byte[] s) {
            this.s = s;
            this.end = s.length;
            endList = new int[s.length + 2];
            java.util.Arrays.fill(endList, -1);
        }

        int node(int l, int r, int w) {
            if (count == lc.length) {
                int n = count * 2;
                lc = java.util.Arrays.copyOf(lc, n);
                rc = java.util.Arrays.copyOf(rc, n);
                wcost = java.util.Arrays.copyOf(wcost, n);
                cost = java.util.Arrays.copyOf(cost, n);
                prev = java.util.Arrays.copyOf(prev, n);
                bnext = java.util.Arrays.copyOf(bnext, n);
                enext = java.util.Arrays.copyOf(enext, n);
                begin = java.util.Arrays.copyOf(begin, n);
                len = java.util.Arrays.copyOf(len, n);
                rlen = java.util.Arrays.copyOf(rlen, n);
            }
            int i = count++;
            lc[i] = l;
            rc[i] = r;
            wcost[i] = w;
            cost[i] = 0;
            prev[i] = -1;
            bnext[i] = -1;
            enext[i] = -1;
            return i;
        }

        /** MeCab の utf8_to_ucs2 と getCharInfo。mblen に、その文字のバイト数を入れる */
        int charInfo(int p) {
            int left = end - p;
            int b0 = p < end ? (s[p] & 0xff) : 0;       // 末尾の先は 0（C の文字列の終わり）
            int code;
            if (b0 < 0x80) {
                mblen = 1;
                code = b0;
            } else if (left >= 2 && (b0 & 0xe0) == 0xc0) {
                mblen = 2;
                code = ((b0 & 0x1f) << 6) | (s[p + 1] & 0x3f);
            } else if (left >= 3 && (b0 & 0xf0) == 0xe0) {
                mblen = 3;
                code = ((b0 & 0x0f) << 12) | ((s[p + 1] & 0x3f) << 6) | (s[p + 2] & 0x3f);
            } else if (left >= 4 && (b0 & 0xf8) == 0xf0) {
                mblen = 4;
                code = 0;
            } else if (left >= 5 && (b0 & 0xfc) == 0xf8) {
                mblen = 5;
                code = 0;
            } else if (left >= 6 && (b0 & 0xfe) == 0xfc) {
                mblen = 6;
                code = 0;
            } else {
                mblen = 1;
                code = 0;
            }
            return code < 0xffff ? charMap[code] : charMap[0];
        }

        /** MeCab の seekToOtherType: c と同じ種類が続く間、進む。info に最後に見た文字の種類、clen に進んだ文字数 */
        int seek(int p, int c) {
            clen = 0;
            while (p != end) {
                info = charInfo(p);
                if (!kindOf(c, info)) break;
                p += mblen;
                clen++;
                c = info;
            }
            return p;
        }

        int addUnknown(int result, int type, int from, int begin2, int begin3) {
            int first = unkFirst[type], n = unkCount[type];
            for (int k = 0; k < n; k++) {
                int t = first + k;
                int i = node(unk.lc(t), unk.rc(t), unk.wcost(t));
                begin[i] = begin2;
                len[i] = begin3 - begin2;
                rlen[i] = begin3 - from;
                bnext[i] = result;
                result = i;
            }
            return result;
        }

        /** MeCab の Tokenizer::lookup: from から始まる単語の候補（辞書の語と未知語）。bnext でつないだ先頭を返す */
        int lookup(int from) {
            int begin2 = seek(from, spaceInfo);
            if (begin2 == end) return -1;
            final int cinfo = info;
            int firstLen = mblen;
            int result = -1;
            int n = sys.prefix(s, begin2, end, daValue, daLen);
            for (int i = 0; i < n; i++) {
                int size = daValue[i] & 0xff, first = daValue[i] >>> 8;
                for (int j = 0; j < size; j++) {
                    int t = first + j;
                    int k = node(sys.lc(t), sys.rc(t), sys.wcost(t));
                    begin[k] = begin2;
                    len[k] = daLen[i];
                    rlen[k] = begin2 - from + daLen[i];
                    bnext[k] = result;
                    result = k;
                }
            }
            if (result != -1 && !invoke(cinfo)) return result;

            int type = defaultType(cinfo);
            int begin3 = begin2 + firstLen;
            int groupBegin3 = -1;
            if (begin3 > end) {
                result = addUnknown(result, type, from, begin2, begin3);
                if (result != -1) return result;
            }
            if (group(cinfo)) {
                int tmp = begin3;
                begin3 = seek(begin3, cinfo);
                if (clen <= MAX_GROUPING) result = addUnknown(result, type, from, begin2, begin3);
                groupBegin3 = begin3;
                begin3 = tmp;
            }
            for (int i = 1; i <= length(cinfo); i++) {
                if (begin3 > end) break;
                if (begin3 == groupBegin3) continue;
                result = addUnknown(result, type, from, begin2, begin3);
                if (!kindOf(cinfo, charInfo(begin3))) break;
                begin3 += mblen;
            }
            if (result == -1) result = addUnknown(result, type, from, begin2, begin3);
            return result;
        }

        /** MeCab の connect: pos で終わる候補のうち、いちばん安いものにつなぐ */
        void connect(int pos, int rnode) {
            for (; rnode != -1; rnode = bnext[rnode]) {
                long best = 2147483647L;
                int bestNode = -1;
                int base = 4 + 2 * lsize * lc[rnode];
                for (int l = endList[pos]; l != -1; l = enext[l]) {
                    long c = (long) cost[l] + matrix.getShort(base + 2 * rc[l]) + wcost[rnode];
                    if (c < best) {
                        bestNode = l;
                        best = c;
                    }
                }
                if (bestNode == -1) throw new IllegalStateException("つなぐ先が無い");
                prev[rnode] = bestNode;
                cost[rnode] = (int) best;
                int x = rlen[rnode] + pos;
                enext[rnode] = endList[x];
                endList[x] = rnode;
            }
        }

        /** いちばん安い道の単語（先頭から）。空白は単語に入らない */
        List<String> best() {
            int bos = node(0, 0, 0);
            endList[0] = bos;
            for (int pos = 0; pos < end; pos++) {
                if (endList[pos] != -1) {
                    int right = lookup(pos);
                    connect(pos, right);
                }
            }
            int eos = node(0, 0, 0);
            rlen[eos] = 0;
            for (int pos = end; pos >= 0; pos--) {
                if (endList[pos] != -1) {
                    int save = endList[pos];
                    connect(pos, eos);
                    endList[pos] = save;      // EOS は終わりの表に残さない（結果には関係しない）
                    break;
                }
            }
            ArrayList<String> out = new ArrayList<String>();
            for (int i = prev[eos]; i != -1 && i != bos; i = prev[i]) {
                out.add(0, new String(s, begin[i], len[i], UTF8));
            }
            return out;
        }
    }

    static final java.nio.charset.Charset UTF8 = java.nio.charset.Charset.forName("UTF-8");

    /** MeCab で分けた単語（NFKC でそろえてから） */
    public List<String> words(String text) {
        String t = Normalizer.normalize(text, Normalizer.Form.NFKC);
        int nul = t.indexOf('\u0000');
        if (nul >= 0) t = t.substring(0, nul);      // MeCab は C の文字列として読む
        byte[] s = t.getBytes(UTF8);
        if (s.length == 0) return new ArrayList<String>();
        return new Lattice(s).best();
    }

    /** WordPiece（`WordpieceTokenizer.tokenize`）。語彙に無ければ [UNK] */
    void wordpiece(String word, List<String> out) {
        int[] cps = codePoints(word);
        if (cps.length > MAX_CHARS_PER_WORD) {
            out.add("[UNK]");
            return;
        }
        ArrayList<String> sub = new ArrayList<String>();
        int start = 0;
        while (start < cps.length) {
            int e = cps.length;
            String cur = null;
            while (start < e) {
                String piece = new String(cps, start, e - start);
                if (start > 0) piece = "##" + piece;
                if (vocab.containsKey(piece)) {
                    cur = piece;
                    break;
                }
                e--;
            }
            if (cur == null) {
                out.add("[UNK]");
                return;
            }
            sub.add(cur);
            start = e;
        }
        out.addAll(sub);
    }

    static int[] codePoints(String s) {
        int n = s.codePointCount(0, s.length());
        int[] out = new int[n];
        for (int i = 0, k = 0; i < s.length(); k++) {
            out[k] = s.codePointAt(i);
            i += Character.charCount(out[k]);
        }
        return out;
    }

    /** 単語 → WordPiece（[CLS]・[SEP] は付けない） */
    public List<String> pieces(String text) {
        ArrayList<String> out = new ArrayList<String>();
        for (String w : words(text)) {
            // 単語の中に空白があれば、そこで分ける（`whitespace_tokenize`）。MeCab の単語には、ふつう空白は入らない
            int a = 0;
            for (int i = 0; i <= w.length(); i++) {
                if (i == w.length() || Py.isSpace(w.charAt(i))) {
                    if (i > a) wordpiece(w.substring(a, i), out);
                    a = i + 1;
                }
            }
        }
        return out;
    }

    /** BERT に渡す ID 列: [CLS] ＋ 本体 ＋ [SEP]。maxLength を超えたら、本体の後ろを切る（PC の truncation=True と同じ） */
    public int[] encode(String text, int maxLength) {
        List<String> p = pieces(text);
        int n = Math.min(p.size(), maxLength - 2);
        int[] ids = new int[n + 2];
        ids[0] = clsId;
        for (int i = 0; i < n; i++) {
            Integer v = vocab.get(p.get(i));
            ids[i + 1] = v == null ? unkId : v;
        }
        ids[n + 1] = sepId;
        return ids;
    }
}
