package org.edgeslm.router;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * アプリ v2 の判定（Java）が、PC（Python の P35 の構成）と同じ結果を返すかを、行ごとに確かめる（P36。P24 の ParityTest を写して、部品を替えたもの）。
 *
 *     java -cp out org.edgeslm.router.ParityTest2 <fixtures のフォルダ> [表示する行数] [BERT の辞書のフォルダ]
 *
 * 単語の分け方は、fixtures の words.tsv（Python と同じ表）を使う。辞書のフォルダを渡すと、あわせて
 * 1) {@link BertJa#words} が表と同じか 2) 表を使わず {@link BertJa} で分けたときも同じ結果か、も確かめる（アプリはこちら）。
 *
 * 食い違いが 1 行でもあれば、終了コード 1。先頭の数行を出す。
 */
public final class ParityTest2 {
    static BufferedReader open(String path) throws Exception {
        return new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
    }

    static int show = 12;

    static void bad(String what, String line, String got) {
        if (show-- > 0) System.out.println("  [" + what + "] " + line.replace('\t', '|') + "\n      Java: " + got);
    }

    static int rules(String dir) throws Exception {
        int n = 0, ng = 0;
        BufferedReader r = open(dir + "/rules.tsv");
        for (String line; (line = r.readLine()) != null; ) {
            String[] c = line.split("\t", -1);
            Rules.Decision d = Rules2.apply(c[0], c[1]);
            String got = d.action + "\t" + (d.demote ? "1" : "0") + "\t" + (d.rule == null ? "-" : d.rule);
            n++;
            if (!got.equals(c[2] + "\t" + c[3] + "\t" + c[4])) {
                ng++;
                bad("規則", line, got.replace('\t', '|'));
            }
        }
        r.close();
        System.out.println("規則: " + n + " 行・食い違い " + ng);
        return ng;
    }

    static int args(String dir) throws Exception {
        int n = 0, ng = 0;
        BufferedReader r = open(dir + "/args.tsv");
        Args.State stopped = new Args.State();
        Args.State moving = new Args.State();
        moving.motion = "moving";
        for (String line; (line = r.readLine()) != null; ) {
            String[] c = line.split("\t", -1);
            Args.Arguments a = Args2.resolve(c[1], c[0], stopped);
            String v1 = a.complete() ? Policy.inspect(c[1], a.values, stopped).code() : "-";
            String v2 = a.complete() ? Policy.inspect(c[1], a.values, moving).code() : "-";
            String got = (a.complete() ? "1" : "0") + "\t" + Args.canon(a.values) + "\t" + v1 + "\t" + v2;
            n++;
            if (!got.equals(c[2] + "\t" + c[3] + "\t" + c[5] + "\t" + c[6])) {
                ng++;
                bad("引数", line, got.replace('\t', '|') + " / メモ: " + a.note);
            }
        }
        r.close();
        System.out.println("引数と検査: " + n + " 行・食い違い " + ng);
        return ng;
    }

    static int flow(String dir, int layers) throws Exception {
        int n = 0, ng = 0;
        BufferedReader r = open(dir + "/flow_L" + layers + ".tsv");
        String[] head = r.readLine().split("\t", -1);
        String[] classes = new String[head.length - 1];
        System.arraycopy(head, 1, classes, 0, classes.length);
        Flow.Thresholds th = new Flow.Thresholds();
        for (String kv : r.readLine().split("\t", -1)) {
            String[] p = kv.split("=", 2);
            if (p.length < 2) continue;
            double v = Double.parseDouble(p[1]);
            if (p[0].equals("CHAT")) th.chat = v;
            else if (p[0].equals("TOOL_read")) th.toolRead = v;
            else if (p[0].equals("TOOL_write")) th.toolWrite = v;
            else if (p[0].equals("REJECT")) th.reject = v;
            else if (p[0].equals("BERT_HI")) th.bertHi = v;
        }
        List<String> tally = new ArrayList<String>();
        int[] count = new int[4];
        for (String line; (line = r.readLine()) != null; ) {
            String[] c = line.split("\t", -1);
            String[] ps = c[1].split(" ");
            double[] probs = new double[ps.length];
            for (int i = 0; i < ps.length; i++) probs[i] = Double.parseDouble(ps[i]);
            Flow.Decision d = Flow.decide(c[0], probs, classes, Double.parseDouble(c[2]), th, Double.parseDouble(c[3]), new Args.State());
            String got = d.outcome + "\t" + (d.stage == null ? "-" : d.stage) + "\t" + (d.tool == null ? "-" : d.tool) + "\t" + Args.canon(d.args) + "\t" + (d.rule == null ? "-" : d.rule);
            n++;
            if (d.outcome.equals(Flow.EXECUTE)) count[0]++;
            else if (d.outcome.equals(Flow.CHAT)) count[1]++;
            else if (d.outcome.equals(Flow.ASK)) count[2]++;
            else count[3]++;
            if (!got.equals(c[4] + "\t" + c[5] + "\t" + c[6] + "\t" + c[7] + "\t" + c[8])) {
                ng++;
                bad("流れ L" + layers, line.substring(0, Math.min(line.length(), 40)) + " … " + c[4] + "|" + c[5] + "|" + c[6] + "|" + c[7] + "|" + c[8], got.replace('\t', '|'));
            }
        }
        r.close();
        System.out.println("判定の流れ（" + layers + " 層）: " + n + " 行・食い違い " + ng + "（実行 " + count[0] + "・雑談 " + count[1] + "・聞き返し " + count[2] + "・拒否 " + count[3] + "）");
        return ng;
    }

    static java.util.Map<String, List<String>> table(String dir) throws Exception {
        java.util.Map<String, List<String>> t = new java.util.HashMap<String, List<String>>();
        BufferedReader r = open(dir + "/words.tsv");
        for (String line; (line = r.readLine()) != null; ) {
            String[] c = line.split("\t", -1);
            t.put(c[0], c[1].isEmpty() ? new ArrayList<String>() : java.util.Arrays.asList(c[1].split(" ")));
        }
        r.close();
        return t;
    }

    static int words(java.util.Map<String, List<String>> t) {
        int n = 0, ng = 0;
        for (java.util.Map.Entry<String, List<String>> e : t.entrySet()) {
            List<String> got = Words.tok.words(e.getKey());
            n++;
            if (!got.equals(e.getValue())) {
                ng++;
                bad("単語", e.getKey() + "\t" + String.join(" ", e.getValue()), String.join(" ", got));
            }
        }
        System.out.println("単語の分け方（BertJa ↔ 表）: " + n + " 行・食い違い " + ng);
        return ng;
    }

    public static void main(String[] a) throws Exception {
        String dir = a.length > 0 ? a[0] : "fixtures";
        if (a.length > 1) show = Integer.parseInt(a[1]);
        java.util.Map<String, List<String>> t = table(dir);
        Words.table = t;
        System.out.println("--- 単語は表（Python と同じ）---");
        int ng = rules(dir) + args(dir) + flow(dir, 17) + flow(dir, 26);
        if (a.length > 2) {
            Words.tok = BertJa.load(new java.io.File(a[2]));
            ng += words(t);
            Words.table = null;
            System.out.println("--- 単語は BertJa（アプリと同じ）---");
            ng += args(dir) + flow(dir, 17) + flow(dir, 26);
        }
        System.out.println(ng == 0 ? "すべて一致" : "食い違いの合計: " + ng);
        System.exit(ng == 0 ? 0 : 1);
    }
}
