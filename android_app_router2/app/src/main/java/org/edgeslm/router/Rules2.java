package org.edgeslm.router;

import java.util.regex.Pattern;

/**
 * いまの規則（P36 でアプリに入れた）。{@link Rules}（P21 まで）の上に、Python の `p31_rules` の R1 を重ねる:
 * 規則 S5（判定した機能の語が文に無い）でナビを止めたとき、「場所 ＋ に・へ ＋ 戻って・戻る」があれば止めない（鍵を取りに自宅へ戻って）。
 */
public final class Rules2 {
    private Rules2() {}

    static final Pattern BACK_TO = Pattern.compile("[一-龥ァ-ヴーA-Za-z]+(?:に|へ)戻(?:って|る|り|ろ)");

    public static Rules.Decision apply(String utterance, String predictedAction) {
        Rules.Decision d = Rules.apply(utterance, predictedAction);
        if (d.demote && "S5_NO_TARGET_TERM".equals(d.rule) && "TOOL_set_navigation".equals(predictedAction) && Py.find(BACK_TO, utterance)) {
            return new Rules.Decision(predictedAction, false, null);
        }
        return d;
    }
}
