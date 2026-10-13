package org.edgeslm.router;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;

/**
 * 分類ヘッド（mlp・幅 256）を端末の CPU で計算する（P24）。PC の `g3_headlib.numpy_head` と同じ式。
 *
 * <pre>
 * 行き先: z  = (h の最後 − μ) / σ                → Linear → GELU → Linear → softmax(·/温度)
 * 棄却:   zr = ([h の最後, h の平均] − μr) / σr  → Linear → GELU → Linear → softmax の 2 つ目
 * </pre>
 *
 * 読むのは `head.tsv`（形・位置・クラス名・温度・閾値）と `head.bin`（float32・リトルエンディアン）。PC の `app/p24_assets.py heads` が作る。
 * 既存の画面の `HeadModel.kt` と同じ計算だが、積和を double で持ち、最終判定はしない（判定は {@link Flow}）。
 * PC と同じ確率になるかは、`HeadTest` が評価セット v6 の端末の h で確かめる。
 */
public final class Head {
    public final int hidden, rejectHidden;
    public final double temperature;
    public final String[] classes;
    public final Flow.Thresholds thresholds = new Flow.Thresholds();
    private final float[] w;
    private final HashMap<String, int[]> tensors = new HashMap<String, int[]>();      // 名前 → {位置, 形…}

    /** 1 回の計算の結果 */
    public static final class Out {
        public double[] action;     // classes の順
        public double reject;       // 意図が不明・機能にない依頼の確率
        public double ms;
    }

    private Head(File dir) throws IOException {
        int hid = 0, rej = 0, total = -1;
        double temp = 1.0;
        String[] cls = null;
        BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(new File(dir, "head.tsv")), "UTF-8"));
        try {
            for (String line; (line = r.readLine()) != null; ) {
                String[] c = line.split("\t", -1);
                if (c[0].equals("hidden")) hid = Integer.parseInt(c[1]);
                else if (c[0].equals("reject_hidden")) rej = Integer.parseInt(c[1]);
                else if (c[0].equals("temperature")) temp = Double.parseDouble(c[1]);
                else if (c[0].equals("total_floats")) total = Integer.parseInt(c[1]);
                else if (c[0].equals("classes")) {
                    cls = new String[c.length - 1];
                    System.arraycopy(c, 1, cls, 0, cls.length);
                } else if (c[0].equals("threshold")) {
                    double v = Double.parseDouble(c[2]);
                    if (c[1].equals("CHAT")) thresholds.chat = v;
                    else if (c[1].equals("TOOL_read")) thresholds.toolRead = v;
                    else if (c[1].equals("TOOL_write")) thresholds.toolWrite = v;
                    else if (c[1].equals("REJECT")) thresholds.reject = v;
                } else if (c[0].equals("tensor")) {
                    int[] t = new int[c.length - 2];
                    for (int i = 0; i < t.length; i++) t[i] = Integer.parseInt(c[i + 2]);
                    tensors.put(c[1], t);
                }
            }
        } finally {
            r.close();
        }
        if (cls == null || hid <= 0 || rej != 2 * hid || total <= 0) throw new IOException("head.tsv が読めない（" + dir + "）");
        hidden = hid;
        rejectHidden = rej;
        temperature = temp;
        classes = cls;
        RandomAccessFile f = new RandomAccessFile(new File(dir, "head.bin"), "r");
        try {
            if (f.length() != 4L * total) throw new IOException("head.bin の大きさが違う: " + f.length() + "（期待 " + 4L * total + "）");
            byte[] b = new byte[(int) f.length()];
            f.readFully(b);
            w = new float[total];
            ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(w);
        } finally {
            f.close();
        }
    }

    /** `dir` に head.tsv と head.bin */
    public static Head load(File dir) throws IOException {
        return new Head(dir);
    }

    private int[] tensor(String name) {
        int[] t = tensors.get(name);
        if (t == null) throw new IllegalStateException("head.tsv に " + name + " が無い");
        return t;
    }

    private double[] standardize(double[] x, String prefix) {
        int om = tensor(prefix + ".mean")[0], os = tensor(prefix + ".std")[0];
        double[] z = new double[x.length];
        for (int i = 0; i < x.length; i++) z[i] = (x[i] - w[om + i]) / w[os + i];
        return z;
    }

    private double[] linear(double[] x, String prefix) {
        int[] tw = tensor(prefix + ".weight");
        int ob = tensor(prefix + ".bias")[0];
        int out = tw[1], in = tw[2];
        if (in != x.length) throw new IllegalStateException(prefix + ": 入力 " + x.length + " と重み " + in + " が合わない");
        double[] y = new double[out];
        for (int o = 0; o < out; o++) {
            double s = 0.0;
            int base = tw[0] + o * in;
            for (int i = 0; i < in; i++) s += w[base + i] * x[i];
            y[o] = s + w[ob + o];
        }
        return y;
    }

    /** erf の近似（Abramowitz–Stegun 7.1.26）。PC の `g3_headlib.erf_as`・既存の `HeadModel.kt` と同じ */
    static double erf(double x) {
        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return x >= 0 ? y : -y;
    }

    private double[] mlp(double[] z, String prefix) {
        double[] h = linear(z, prefix + ".0");
        for (int i = 0; i < h.length; i++) h[i] = 0.5 * h[i] * (1.0 + erf(h[i] / Math.sqrt(2.0)));
        return linear(h, prefix + ".2");
    }

    static double[] softmax(double[] v, double temp) {
        double m = Double.NEGATIVE_INFINITY;
        for (double a : v) m = Math.max(m, a / temp);
        double[] e = new double[v.length];
        double s = 0;
        for (int i = 0; i < v.length; i++) {
            e[i] = Math.exp(v[i] / temp - m);
            s += e[i];
        }
        for (int i = 0; i < v.length; i++) e[i] /= s;
        return e;
    }

    public Out predict(float[] hLast, float[] hMean) {
        if (hLast.length != hidden || hMean.length != hidden) throw new IllegalArgumentException("h の長さが違う: " + hLast.length + "/" + hMean.length + "（期待 " + hidden + "）");
        long t0 = System.nanoTime();
        double[] x = new double[hidden], xr = new double[rejectHidden];
        for (int i = 0; i < hidden; i++) {
            x[i] = hLast[i];
            xr[i] = hLast[i];
            xr[hidden + i] = hMean[i];
        }
        Out o = new Out();
        o.action = softmax(mlp(standardize(x, "standardizer"), "heads.action"), temperature);
        o.reject = softmax(mlp(standardize(xr, "reject_standardizer"), "heads.reject"), 1.0)[1];
        o.ms = (System.nanoTime() - t0) / 1e6;
        return o;
    }
}
