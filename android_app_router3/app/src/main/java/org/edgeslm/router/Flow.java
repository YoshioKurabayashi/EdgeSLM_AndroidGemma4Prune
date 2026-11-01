package org.edgeslm.router;

import java.util.Map;

/**
 * 1 つの発話についての判定（P24。P36 で規則を {@link Rules2}（P31）、引数を {@link Args2}（P35）に替えた）。PC の `app/p36_decide.py` の `decide` と同じ順番・同じ中身。
 *
 * 順番: BERT が断る → 規則 → 棄却ヘッド → 確信度 →（雑談なら BERT のフラグ）→ 引数が足りない → 検査 → BERT のフラグ → 実行。
 * 入力（ヘッドの確率・棄却・BERT の確率）は、呼ぶ側が計算して渡す。ここには、学習したものは入っていない。
 */
public final class Flow {
    private Flow() {}

    public static final String EXECUTE = "実行", CHAT = "雑談", ASK = "聞き返し", REFUSE = "拒否";
    /** これ以上で、BERT のフラグ（`p8_gate.BERT_LO` と同じ） */
    public static final double BERT_LO = 0.5;

    /** 閾値。層の数ごとに違う（`head.json` に入れてある） */
    public static final class Thresholds {
        public double chat, toolRead, toolWrite, reject, bertHi;
    }

    /** 判定の結果と、途中の段の値 */
    public static final class Decision {
        public String outcome;          // 実行・雑談・聞き返し・拒否
        public String stage;            // 止めた段。止めていなければ null
        public String headAction;       // ヘッドの 1 位（規則の前）
        public String action;           // 規則の後
        public String tool;
        public Map<String, Object> args;
        public String argNote;
        public java.util.List<String> missing;   // 足りない引数の名前（「引数が足りない」で止めたとき）
        public String rule;
        public String response;         // 検査が落ちたときの、利用者向けの文
        public double confidence = Double.NaN, tau = Double.NaN, reject, bert;

        Decision end(String outcome, String stage) {
            this.outcome = outcome;
            this.stage = stage;
            return this;
        }
    }

    public static Decision decide(String utterance, double[] actionProbs, String[] classes, double reject, Thresholds th, double bertProb, Args.State state) {
        Decision out = new Decision();
        out.reject = reject;
        out.bert = bertProb;
        if (bertProb >= th.bertHi) return out.end(REFUSE, "BERT が断る");
        int k = 0;
        for (int i = 1; i < classes.length; i++) {
            if (actionProbs[i] > actionProbs[k]) k = i;      // Python の max(range, key) と同じく、同点は最初
        }
        out.headAction = classes[k];
        out.confidence = actionProbs[k];
        Rules.Decision d = Rules2.apply(utterance, classes[k]);          // P36: P31 の規則（R1）
        String act = d.action;
        out.action = act;
        out.rule = d.rule;
        if (d.demote) return out.end(ASK, "規則");
        if (reject >= th.reject) return out.end(ASK, "棄却ヘッド");
        boolean chat = "CHAT".equals(act);
        out.tau = chat ? th.chat : (Policy.isWrite(act.substring(5)) ? th.toolWrite : th.toolRead);
        if (out.confidence < out.tau) return out.end(ASK, "確信度");
        if (chat) {
            if (bertProb >= BERT_LO) return out.end(ASK, "BERT のフラグ");
            return out.end(CHAT, null);
        }
        String tool = act.substring(5);
        out.tool = tool;
        Args.Arguments ar = Args2.resolve(tool, utterance, state);       // P36: P35 の引数（歯止め・範囲外は断る）
        out.argNote = ar.note;
        if (!ar.complete()) {
            out.missing = ar.missing;
            return out.end(ASK, "引数が足りない");
        }
        out.args = ar.values;
        Policy.Verdict v = Policy.inspect(tool, ar.values, state);
        if (!v.ok) {
            out.response = v.response;
            return out.end(v.missing ? ASK : REFUSE, "検査");
        }
        if (bertProb >= BERT_LO) return out.end(ASK, "BERT のフラグ");
        return out.end(EXECUTE, null);
    }
}
