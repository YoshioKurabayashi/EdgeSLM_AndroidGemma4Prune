package org.edgeslm.router;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * 端末用のヘッドの計算（{@link Head}）と判定（{@link Flow}）を、評価セット v6 の端末の h からつなげて、PC と同じ結果になるかを確かめる（P24）。
 *
 *     java -cp out org.edgeslm.router.HeadTest <fixtures のフォルダ> <router のフォルダ（L17・L26 がある所）>
 *
 * 入力: `h_L<n>.bin`（600 行 × [最後 1536 ＋ 平均 1536]・float32）と `flow_L<n>.tsv`（PC の確率・BERT の確率・結果）。
 * 見るもの: ヘッドの確率・棄却の差の最大、閾値、そして「h → ヘッド → 判定」の結果が PC と同じか（BERT の確率は PC の値を使う）。
 */
public final class HeadTest {
    public static void main(String[] a) throws Exception {
        int ng = 0;
        for (int n : new int[]{17, 26}) {
            Head head = Head.load(new File(a[1], "L" + n));
            RandomAccessFile f = new RandomAccessFile(new File(a[0], "h_L" + n + ".bin"), "r");
            byte[] b = new byte[(int) f.length()];
            f.readFully(b);
            f.close();
            FloatBuffer h = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(new File(a[0], "flow_L" + n + ".tsv")), "UTF-8"));
            String[] cls = r.readLine().split("\t", -1);
            boolean sameClasses = cls.length - 1 == head.classes.length;
            for (int i = 0; sameClasses && i < head.classes.length; i++) sameClasses = cls[i + 1].equals(head.classes[i]);
            double bertHi = 0, dth = 0;
            for (String kv : r.readLine().split("\t", -1)) {
                String[] p = kv.split("=", 2);
                if (p.length < 2) continue;
                double v = Double.parseDouble(p[1]);
                if (p[0].equals("BERT_HI")) bertHi = v;
                else if (p[0].equals("CHAT")) dth = Math.max(dth, Math.abs(v - head.thresholds.chat));
                else if (p[0].equals("TOOL_read")) dth = Math.max(dth, Math.abs(v - head.thresholds.toolRead));
                else if (p[0].equals("TOOL_write")) dth = Math.max(dth, Math.abs(v - head.thresholds.toolWrite));
                else if (p[0].equals("REJECT")) dth = Math.max(dth, Math.abs(v - head.thresholds.reject));
            }
            head.thresholds.bertHi = bertHi;
            int rows = 0, bad = 0;
            double dp = 0, dr = 0, ms = 0;
            float[] last = new float[head.hidden], mean = new float[head.hidden];
            for (String line; (line = r.readLine()) != null; rows++) {
                String[] c = line.split("\t", -1);
                h.get(last);
                h.get(mean);
                Head.Out o = head.predict(last, mean);
                ms += o.ms;
                String[] ps = c[1].split(" ");
                for (int i = 0; i < ps.length; i++) dp = Math.max(dp, Math.abs(o.action[i] - Double.parseDouble(ps[i])));
                dr = Math.max(dr, Math.abs(o.reject - Double.parseDouble(c[2])));
                Flow.Decision d = Flow.decide(c[0], o.action, head.classes, o.reject, head.thresholds, Double.parseDouble(c[3]), new Args.State());
                String got = d.outcome + "\t" + (d.stage == null ? "-" : d.stage) + "\t" + (d.tool == null ? "-" : d.tool) + "\t" + Args.canon(d.args) + "\t" + (d.rule == null ? "-" : d.rule);
                if (!got.equals(c[4] + "\t" + c[5] + "\t" + c[6] + "\t" + c[7] + "\t" + c[8])) {
                    bad++;
                    if (bad <= 5) System.out.println("  [判定が違う] " + c[0] + " PC: " + c[4] + "|" + c[5] + "|" + c[6] + "|" + c[7] + " Java: " + got.replace('\t', '|'));
                }
            }
            r.close();
            boolean ok = sameClasses && bad == 0 && dp < 1e-9 && dr < 1e-9 && dth == 0 && rows > 0 && !h.hasRemaining();
            System.out.println(String.format("ヘッド（%d 層）: %d 行・確率の差の最大 %.1e・棄却の差の最大 %.1e・閾値の差 %.1e・h からの判定の食い違い %d（1 行 %.2f ms）%s",
                    n, rows, dp, dr, dth, bad, ms / Math.max(rows, 1), ok ? "" : " ★合わない"));
            if (!ok) ng++;
        }
        System.exit(ng == 0 ? 0 : 1);
    }
}
