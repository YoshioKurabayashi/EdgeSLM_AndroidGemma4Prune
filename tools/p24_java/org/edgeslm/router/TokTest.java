package org.edgeslm.router;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;

/**
 * 端末用の BERT の入力作り（{@link BertJa}）が、PC の `BertJapaneseTokenizer` と同じ ID 列を返すかを確かめる（P24）。
 *
 *     java -cp out org.edgeslm.router.TokTest <辞書のフォルダ> <正解の表 tok_truth.tsv> [出す行数]
 *     java -cp out org.edgeslm.router.TokTest <辞書のフォルダ> - 文1 文2 …        # 分け方を見るだけ
 *
 * 正解の表は 1 行に「発話 TAB ID 列（空白区切り。[CLS]・[SEP] 込み・64 で切ったもの）TAB 単語（空白区切り）」。PC の `app/p24_assets.py truth` が作る。
 * 食い違いが 1 行でもあれば、終了コード 1。
 */
public final class TokTest {
    public static void main(String[] a) throws Exception {
        BertJa tok = BertJa.load(new File(a[0]));
        if (a[1].equals("-")) {
            for (int i = 2; i < a.length; i++) {
                StringBuilder ids = new StringBuilder();
                for (int v : tok.encode(a[i], 64)) ids.append(v).append(' ');
                System.out.println(a[i] + "\n  単語: " + tok.words(a[i]) + "\n  WordPiece: " + tok.pieces(a[i]) + "\n  ID: " + ids);
            }
            return;
        }
        int show = a.length > 2 ? Integer.parseInt(a[2]) : 20;
        BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(a[1]), "UTF-8"));
        int n = 0, ng = 0, ngWords = 0;
        long t0 = System.nanoTime();
        for (String line; (line = r.readLine()) != null; ) {
            String[] c = line.split("\t", -1);
            if (c.length < 2) continue;
            StringBuilder ids = new StringBuilder();
            for (int v : tok.encode(c[0], 64)) {
                if (ids.length() > 0) ids.append(' ');
                ids.append(v);
            }
            n++;
            boolean bad = !ids.toString().equals(c[1]);
            String words = join(tok.words(c[0]));
            boolean badWords = c.length > 2 && !words.equals(c[2]);
            if (badWords) ngWords++;
            if (bad) {
                ng++;
                if (show-- > 0) System.out.println("  [ID 列が違う] " + c[0] + "\n      PC  : " + c[1] + (c.length > 2 ? " / " + c[2] : "") + "\n      Java: " + ids + " / " + words);
            } else if (badWords && show-- > 0) {
                System.out.println("  [単語の分け方だけ違う] " + c[0] + "\n      PC  : " + c[2] + "\n      Java: " + words);
            }
        }
        r.close();
        System.out.println("BERT の入力: " + n + " 行・ID 列の食い違い " + ng + "・単語の分け方の食い違い " + ngWords
                + String.format("（1 行 %.2f ms）", (System.nanoTime() - t0) / 1e6 / Math.max(n, 1)));
        System.exit(ng == 0 && ngWords == 0 && n > 0 ? 0 : 1);
    }

    static String join(List<String> ws) {
        StringBuilder b = new StringBuilder();
        for (String w : ws) {
            if (b.length() > 0) b.append(' ');
            b.append(w);
        }
        return b.toString();
    }
}
