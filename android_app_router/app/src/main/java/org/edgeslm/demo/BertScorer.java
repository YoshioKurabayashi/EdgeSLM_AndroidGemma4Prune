package org.edgeslm.demo;

import java.io.File;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * 日本語 BERT（案 F・seed 0 を ONNX にしたもの）を、端末の CPU で動かす（P24）。「機能にない依頼」である確率を返す。
 *
 * ONNX Runtime の Java の口（アプリに入っている `onnxruntime-android-qnn-1.27.0.aar` の `ai.onnxruntime`）を使う。NPU は使わない
 * （NPU は Gemma 4 のグラフが使う。BERT は 1 文 1 回・数十 ms なので CPU で足りる）。
 * 入力は {@link org.edgeslm.router.BertJa#encode} の ID 列。1 文ずつなので、詰め物（padding）は無い。
 */
final class BertScorer {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final boolean hasMask, hasType;
    final long loadMs;

    BertScorer(File model, int threads) throws OrtException {
        long t0 = System.currentTimeMillis();
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(threads);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        session = env.createSession(model.getAbsolutePath(), o);
        Set<String> names = session.getInputNames();
        hasMask = names.contains("attention_mask");
        hasType = names.contains("token_type_ids");
        prob(new int[]{2, 3});      // 1 回目は遅いので、先に 1 回流しておく
        loadMs = System.currentTimeMillis() - t0;
    }

    /** softmax の 2 つ目（機能にない依頼である確率） */
    double prob(int[] ids) throws OrtException {
        int n = ids.length;
        long[] a = new long[n], ones = new long[n], zeros = new long[n];
        for (int i = 0; i < n; i++) {
            a[i] = ids[i];
            ones[i] = 1L;
        }
        long[] shape = {1, n};
        HashMap<String, OnnxTensor> in = new HashMap<String, OnnxTensor>();
        try {
            in.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(a), shape));
            if (hasMask) in.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(ones), shape));
            if (hasType) in.put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(zeros), shape));
            OrtSession.Result r = session.run(in);
            try {
                float[][] logits = (float[][]) r.get(0).getValue();
                double z0 = logits[0][0], z1 = logits[0][1], m = Math.max(z0, z1);
                double e0 = Math.exp(z0 - m), e1 = Math.exp(z1 - m);
                return e1 / (e0 + e1);
            } finally {
                r.close();
            }
        } finally {
            for (OnnxTensor t : in.values()) t.close();
        }
    }

    void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
        }
    }
}
