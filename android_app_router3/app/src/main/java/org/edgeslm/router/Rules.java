package org.edgeslm.router;

import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 決定的な規則（P24。Python から写したもの）。分類の結果を実行へ渡す前にかける。
 *
 * 写した元（上から順に重なっている。履歴の無い 1 回目の発話だけを扱う）:
 * - EdgeSLM_MultiHead_Multi `dispatch/rules.py`（自己矛盾・指示語だけ・対象の語だけ）
 * - `p13_rules`（S1 言い直し・S2 取り消し・S3 言いよどみ）
 * - `p14_rules`（S4 対象の語の食い違い）
 * - `p17_rules`（指示語でない言い回しの例外）
 * - `p19_rules`（S5 対象の語が無い）
 * - `p21_rules`（位置の「前の」・動詞に付いた「直し」「忘れて」／「待って」「なんだけど」／タイヤの空気）
 *
 * Python と同じ結果になるかは、`tools/p24_java` の `ParityTest` が `work/p24/fixtures/rules.tsv`（発話 × 判定 9 通り）で確かめる。
 */
public final class Rules {
    private Rules() {}

    public static final class Decision {
        public final String action;
        public final boolean demote;
        public final String rule;      // 当たった規則。無ければ null

        Decision(String action, boolean demote, String rule) {
            this.action = action;
            this.demote = demote;
            this.rule = rule;
        }
    }

    // ---------------------------------------------------------------- EdgeSLM_MultiHead_Multi rules.py
    static final String[] ANAPHORA_MARKERS = {"あれ", "それ", "これ", "あっち", "そっち", "こっち", "さっき", "例の", "いつもの", "前の", "元に", "この前",
            "もう少し", "もうちょい", "あとちょっと", "ちょっとだけ", "少しだけ", "変えて", "調整して", "直して", "戻して", "半分"};
    static final String[] T_HVAC = {"エアコン", "冷房", "暖房", "温度", "風", "送風", "ヒーター", "クーラー", "度", "足元", "デフロスタ", "内気", "外気"};
    static final String[] T_MEDIA = {"音楽", "音量", "音", "曲", "ボリューム", "ラジオ", "再生", "ジャズ", "ミュート"};
    static final String[] T_NAV = {"ナビ", "案内", "目的地", "行き先", "ルート", "場所", "設定", "駅", "空港", "自宅", "実家"};
    static final String[] T_CAM = {"カメラ", "後ろ", "後方", "前方", "周囲", "映像", "映して", "見せて"};
    static final String[] T_STATUS = {"燃費", "ガソリン", "燃料", "空気圧", "バッテリー", "走行距離", "残量"};
    static final String[] T_POI = {"近く", "周辺", "コンビニ", "ガソリンスタンド", "駐車場", "レストラン", "店"};
    static final String[] T_MINFO = {"曲名", "アーティスト", "この曲", "歌手", "アルバム"};
    static final String[] T_MANUAL = {"警告灯", "マニュアル", "取扱説明", "意味", "やり方", "使い方"};
    static final String[] UNDO_MARKERS = {"取り消", "取消", "キャンセル", "今のなし", "今のは", "なかったことに", "やっぱいい", "やっぱりいい", "やっぱなし"};
    static final String[] EXPLICIT_OPERATION_MARKERS = {"上げ", "下げ", "強く", "弱く", "大きく", "小さく", "静かに", "涼しく", "暖か", "つけ", "消し", "止め", "開け", "閉め",
            "送って", "映して", "見せて", "設定", "別の", "他の", "次", "最大", "最小", "最強", "最弱", "最高", "最低", "全開", "全閉", "マックス", "一番"};
    static final String[] CASE_PARTICLES_BY_LEN = {"から", "まで", "は", "が", "を", "に", "で", "と", "の", "へ", "も"};      // sorted(key=len, reverse=True) と同じ並び
    static final String[] CONTRADICTION_MARKERS = {"いや", "やっぱ", "じゃなくて", "ではなく", "でなくて", "違う", "間違え", "訂正"};
    static final String[][] OPPOSITE_PAIRS = {{"上げ", "下げ"}, {"強く", "弱く"}, {"大きく", "小さく"}, {"つけ", "消し"}, {"開け", "閉め"}, {"暑く", "涼しく"}, {"暖か", "涼し"},
            {"止め", "続け"}, {"再生", "停止"}, {"最大", "最小"}, {"速く", "遅く"}, {"増や", "減ら"}, {"進め", "戻し"}, {"入れ", "切っ"}};
    static final String[] CANCEL_PHRASES = {"今はいい", "いいや", "やめとく", "やめておく", "やっぱいい", "やっぱりいい", "なしで", "結構です", "大丈夫です"};
    static final Pattern DIGITS = Pattern.compile("[0-9０-９]");
    static final Set<String> ALL_TARGET_TERMS = new HashSet<String>();

    static {
        for (String[] terms : new String[][]{T_HVAC, T_MEDIA, T_NAV, T_CAM, T_STATUS, T_POI, T_MINFO, T_MANUAL}) {
            for (String t : terms) ALL_TARGET_TERMS.add(t);
        }
    }

    static String[] targetTerms(String tool) {
        if ("control_hvac".equals(tool)) return T_HVAC;
        if ("control_media".equals(tool)) return T_MEDIA;
        if ("set_navigation".equals(tool)) return T_NAV;
        if ("capture_camera".equals(tool)) return T_CAM;
        if ("get_vehicle_status".equals(tool)) return T_STATUS;
        if ("search_poi".equals(tool)) return T_POI;
        if ("get_media_info".equals(tool)) return T_MINFO;
        if ("search_vehicle_manual".equals(tool)) return T_MANUAL;
        return new String[0];
    }

    static String toolOf(String action) {
        return action.startsWith("TOOL_") ? action.substring(5) : null;
    }

    static boolean hasTarget(String u, String tool) {
        return Py.any(u, targetTerms(tool));
    }

    static boolean isUndo(String u) {
        return Py.any(u, UNDO_MARKERS);
    }

    static boolean carriesExplicitOperation(String u) {
        return Py.find(DIGITS, u) || Py.any(u, EXPLICIT_OPERATION_MARKERS);
    }

    static boolean isBareTarget(String utterance) {
        String text = Py.strip(utterance);
        for (String particle : CASE_PARTICLES_BY_LEN) {
            if (text.endsWith(particle)) {
                return ALL_TARGET_TERMS.contains(text.substring(0, text.length() - particle.length()));
            }
        }
        return false;
    }

    static boolean isSelfContradictory(String u) {
        if (!Py.any(u, CONTRADICTION_MARKERS)) return false;
        for (String[] pair : OPPOSITE_PAIRS) {
            if (u.contains(pair[0]) && u.contains(pair[1])) return true;
        }
        return Py.any(u, CANCEL_PHRASES) && carriesExplicitOperation(u);
    }

    /** 元の 4 つの規則（履歴なし・直前のツールなし） */
    static Decision base(String u, String predicted) {
        String tool = toolOf(predicted);
        if (tool == null) return new Decision(predicted, false, null);
        if (isSelfContradictory(u)) return new Decision(predicted, true, "SELF_CONTRADICTION");
        if (isUndo(u) && !hasTarget(u, tool)) return new Decision(predicted, true, "ANAPHORA_WITHOUT_HISTORY");
        if (Py.any(u, ANAPHORA_MARKERS) && !hasTarget(u, tool) && !carriesExplicitOperation(u)) {
            return new Decision(predicted, true, "ANAPHORA_WITHOUT_HISTORY");
        }
        if (isBareTarget(u)) return new Decision(predicted, true, "BARE_TARGET_NO_PREDICATE");
        return new Decision(predicted, false, null);       // 直前のツールが無いので、再実行の抑止（SUPPRESS_REEXECUTION）は当たらない
    }

    // ---------------------------------------------------------------- p13_rules（S1〜S3）
    static final String FILLER = "(?:あのさ|あのー|あの|えっと|えーっと|えーと|えー|んーと|んー|うーんと|うーん|あー|ねえ|ごめん|ちょ|あ)";
    static final Pattern REDO = Pattern.compile("[、。 　](?:あ、|あ |待って、)?(?:いいや|いや|やっぱ)");
    static final Pattern LEAD = Pattern.compile("^(?:" + FILLER + "[、。 　ー]*)+");
    static final Pattern TRAIL = Pattern.compile("[、 　]" + FILLER + "[ー…。、]*$|[、 　](?:その|そのー)[ー…。、]*$");
    static final String[] UNDO_MORE = {"忘れて", "今言った", "いま言った", "さっき言った", "取りやめ", "無かったことに", "無しで", "無しにして", "なしにして",
            "やらなくていい", "何もしないで", "言い間違え", "間違えた"};
    static final String[] PARTICLES = {"は", "が", "を", "に", "で", "と", "の", "へ", "も"};
    static final String EDGE = "、。 　";

    static String dropLead(String s) {
        return LEAD.matcher(s).replaceFirst("");
    }

    static boolean redoWithContent(String u) {
        Matcher m = REDO.matcher(u);
        if (!m.find()) return false;
        return Py.strip(dropLead(u.substring(0, m.start())), EDGE).length() >= 1;
    }

    static boolean trailingHesitation(String u) {
        Matcher m = TRAIL.matcher(u);
        if (!m.find()) return false;
        String head = Py.strip(u.substring(0, m.start()), EDGE);
        return head.length() > 0 && Py.endsWithAny(head, PARTICLES);
    }

    static Decision p13(String u, String predicted) {
        Decision d = base(u, predicted);
        if (d.demote || d.rule != null || !predicted.startsWith("TOOL_")) return d;
        String tool = predicted.substring(5);
        if (redoWithContent(u)) return new Decision(predicted, true, "S1_REDO");
        if (Py.any(u, UNDO_MORE) && !hasTarget(u, tool)) return new Decision(predicted, true, "S2_UNDO");
        if (trailingHesitation(u)) return new Decision(predicted, true, "S3_HESITATION");
        return d;
    }

    // ---------------------------------------------------------------- p14_rules（S4）
    static final String[] W_TOOLS = {"control_hvac", "control_media", "set_navigation"};
    static final String[] TERMS_HVAC = {"エアコン", "冷房", "暖房", "空調", "温度", "風", "ヒーター", "クーラー", "デフロスタ", "曇り", "暖か", "暖め", "涼し", "冷や", "暑", "寒"};
    static final String[] TERMS_MEDIA = {"音楽", "音量", "音", "曲", "ボリューム", "ラジオ", "再生", "ミュート", "消音", "歌"};
    static final String[] TERMS_NAV = {"ナビ", "案内", "目的地", "行き先", "ルート", "経由", "向かっ", "寄"};
    static final Pattern DEGREES = Pattern.compile("[0-9０-９]+度");

    static String[] terms(String tool) {
        if ("control_hvac".equals(tool)) return TERMS_HVAC;
        if ("control_media".equals(tool)) return TERMS_MEDIA;
        if ("set_navigation".equals(tool)) return TERMS_NAV;
        return null;
    }

    static boolean hasTerm(String u, String tool) {
        if ("control_hvac".equals(tool) && Py.find(DEGREES, u)) return true;
        return Py.any(u, terms(tool));
    }

    static String mismatch(String u, String tool) {
        if (terms(tool) == null || hasTerm(u, tool)) return null;
        for (String other : W_TOOLS) {
            if (!other.equals(tool) && hasTerm(u, other)) return other;
        }
        return null;
    }

    static Decision p14(String u, String predicted) {
        Decision d = p13(u, predicted);
        if (d.demote || d.rule != null || !predicted.startsWith("TOOL_")) return d;
        if (mismatch(u, predicted.substring(5)) != null) return new Decision(predicted, true, "S4_TARGET_MISMATCH");
        return d;
    }

    // ---------------------------------------------------------------- p17_rules（指示語でない言い回し）
    static final String[] SONG_WORDS = {"曲", "歌", "誰", "何て", "タイトル", "アーティスト", "バンド"};
    static final String[] NOT_DEICTIC = {"これまで", "これから", "それぞれ", "それほど", "それから", "それなら", "直してくれる", "直してもらえる", "直せる"};

    static Decision p17(String u, String predicted) {
        Decision d = p14(u, predicted);
        if (!"ANAPHORA_WITHOUT_HISTORY".equals(d.rule)) return d;
        String v = u;
        for (String w : NOT_DEICTIC) v = v.replace(w, "");
        if ("TOOL_get_media_info".equals(predicted) && Py.any(u, SONG_WORDS)) v = v.replace("これ", "この曲");
        if (v.equals(u)) return d;
        return p14(v, predicted);
    }

    // ---------------------------------------------------------------- p19_rules（S5）
    static final String[] S5_HVAC = concat(TERMS_HVAC, new String[]{"冷え", "冷た", "換気", "汗", "じめじめ", "蒸", "息苦し", "足元", "曇", "ガラスが白", "乾かし", "吹き出し", "送風", "ひんやり", "AC"});
    static final String[] S5_MEDIA = concat(TERMS_MEDIA, new String[]{"流し", "聴", "聞こえ", "聞き取", "うるさ", "スキップ", "飛ばし", "一時停止", "ポーズ", "プレイ", "トラック", "シャッフル", "リピート",
            "ランダム", "Bluetooth", "FM", "ジャズ", "クラシック", "BGM", "ミュージック", "ソング", "ポップス", "局"});
    static final String[] S5_NAV = concat(TERMS_NAV, new String[]{"まで", "行き", "行っ", "帰", "連れて", "向か", "高速", "下道", "道", "駅", "空港", "場所", "途中", "デスティネーション", "有料"});
    static final Pattern AIR = Pattern.compile("空気(?!圧)");

    static String[] concat(String[] a, String[] b) {
        String[] out = new String[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    static String[] s5Terms(String tool) {
        if ("control_hvac".equals(tool)) return S5_HVAC;
        if ("control_media".equals(tool)) return S5_MEDIA;
        if ("set_navigation".equals(tool)) return S5_NAV;
        return null;
    }

    static boolean hasS5Term(String u, String tool) {
        if ("control_hvac".equals(tool) && (Py.find(DEGREES, u) || Py.find(AIR, u))) return true;
        return Py.any(u, s5Terms(tool));
    }

    static Decision p19(String u, String predicted) {
        Decision d = p17(u, predicted);
        if (d.demote || d.rule != null || !predicted.startsWith("TOOL_")) return d;
        String tool = predicted.substring(5);
        if (s5Terms(tool) != null && !hasS5Term(u, tool)) return new Decision(predicted, true, "S5_NO_TARGET_TERM");
        return d;
    }

    // ---------------------------------------------------------------- p21_rules
    static final Pattern FRONT = Pattern.compile("((?:右|左|斜め|目の|手|すぐ|真ん)?)前の(?=ガラス|カメラ|席|座席|シート|映像|タイヤ|バンパー|ほう|方|車|様子|窓|ライト|ワイパー|ドア|角|景色|信号|道)");
    static final Pattern FRONT_SIDE = Pattern.compile("(右|左|斜め|目の|手|すぐ)前の");
    static final Pattern REDO_VERB = Pattern.compile("(入れ|かけ|つけ|設定し|案内し|引き|探し|調べ|映し|流し|セットし|計算し|付け|掛け)直し");
    static final String[] NOUNS_NAV = {"ナビ", "案内", "目的地", "行き先", "ルート", "経由", "道"};
    static final String[] PRONOUNS = {"あれ", "それ", "これ", "あっち", "そっち", "こっち", "さっき", "例の", "いつもの", "元に", "この前"};
    static final Pattern FORGET_VERB = Pattern.compile("(?<=[ぁ-ん一-龥ァ-ヶ])(?<![はをもがの])忘れて");
    static final Pattern FORGET_STATE = Pattern.compile("忘れて(?=ない|なかっ|た|る|き|しま|いな)");
    static final Pattern WITHOUT = Pattern.compile("(?<=[一-龥ァ-ヶー])(?:無し|なし)(?=で|に)");
    static final Pattern WAIT = Pattern.compile("[、。 　](?:あ、|あ |いや、?|やっぱ、?)?(?:ちょっと|少し)?待っ(?:て|た)");
    static final Pattern NOUN_BUT = Pattern.compile("^([^、。]{1,14}?)(?:のこと|の件)?なんだけど(?:[、 　]?" + FILLER + ")?[ー…。、]*$");
    static final String[] CASE = {"を", "に", "で", "へ"};

    /** group(1) ＋ 後ろに付ける文字 に置き換える（re.sub(lambda m: m.group(1) + tail)） */
    static String subGroup1(Pattern p, String s, String tail) {
        Matcher m = p.matcher(s);
        StringBuffer b = new StringBuffer();
        while (m.find()) {
            String g = m.group(1) == null ? "" : m.group(1);
            m.appendReplacement(b, Matcher.quoteReplacement(g + tail));
        }
        m.appendTail(b);
        return b.toString();
    }

    static String softenAnaphora(String u) {
        String v = subGroup1(FRONT, u, "まえの");
        v = subGroup1(FRONT_SIDE, v, "まえの");
        return subGroup1(REDO_VERB, v, "なおし");
    }

    static String softenUndo(String u) {
        String v = FORGET_VERB.matcher(u).replaceAll("わすれて");
        v = FORGET_STATE.matcher(v).replaceAll("わすれて");
        return WITHOUT.matcher(v).replaceAll("ぬき");
    }

    static boolean hasNoun(String u, String tool) {
        if ("control_hvac".equals(tool) && Py.find(DEGREES, u)) return true;
        if ("control_hvac".equals(tool)) return Py.any(u, TERMS_HVAC);
        if ("control_media".equals(tool)) return Py.any(u, TERMS_MEDIA);
        return Py.any(u, NOUNS_NAV);
    }

    static boolean tireAirOnly(String u) {
        return u.contains("タイヤ") && Py.find(AIR, u) && !Py.find(DEGREES, u) && !Py.any(u, S5_HVAC);
    }

    static boolean waitWithContent(String u) {
        Matcher m = WAIT.matcher(u);
        if (!m.find()) return false;
        return Py.strip(dropLead(u.substring(0, m.start())), EDGE).length() >= 1;
    }

    static boolean nounBut(String u) {
        Matcher m = NOUN_BUT.matcher(u);
        if (!m.find()) return false;
        return !Py.any(m.group(1), CASE);
    }

    /** いまの規則（P21 まで）。履歴の無い 1 回目の発話に使う */
    public static Decision apply(String u, String predicted) {
        Decision d = p19(u, predicted);
        if (!predicted.startsWith("TOOL_")) return d;
        String tool = predicted.substring(5);
        if ("ANAPHORA_WITHOUT_HISTORY".equals(d.rule)) {
            String v = softenAnaphora(u);
            if (!v.equals(u)) return apply(v, predicted);
            if (terms(tool) != null && hasNoun(u, tool) && !Py.any(u, PRONOUNS) && !isUndo(u)) d = new Decision(predicted, false, null);
        }
        if ("S2_UNDO".equals(d.rule)) {
            String v = softenUndo(u);
            if (!v.equals(u)) return apply(v, predicted);
        }
        if (d.demote || d.rule != null) return d;
        if ("control_hvac".equals(tool) && tireAirOnly(u)) return new Decision(predicted, true, "S5_NO_TARGET_TERM");
        if (waitWithContent(u)) return new Decision(predicted, true, "S1_REDO_WAIT");
        if (nounBut(u)) return new Decision(predicted, true, "S3_NOUN_BUT");
        return d;
    }
}
