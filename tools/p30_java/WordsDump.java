package org.edgeslm.router;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;

/** 発話（1 行 1 文）を、端末の BertJa（MeCab ＋ unidic-lite・NFKC）で単語に分け、「発話 TAB 単語（空白区切り）」で出す（P30）。 */
public final class WordsDump {
    public static void main(String[] a) throws Exception {
        BertJa tok = BertJa.load(new File(a[0]));
        BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(a[1]), StandardCharsets.UTF_8));
        PrintWriter w = new PrintWriter(new OutputStreamWriter(new FileOutputStream(a[2]), StandardCharsets.UTF_8));
        int n = 0;
        for (String line; (line = r.readLine()) != null; ) {
            if (line.isEmpty()) continue;
            List<String> ws = tok.words(line);
            w.print(line); w.print('\t'); w.print(String.join(" ", ws)); w.print('\n');
            n++;
        }
        w.close(); r.close();
        System.out.println("単語に分けた行: " + n);
    }
}
