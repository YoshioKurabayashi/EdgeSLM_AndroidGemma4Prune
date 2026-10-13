package org.edgeslm.router;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 引数の取り出し（P24。Python から写したもの）。LLM は使わない。キーワードと正規表現の決定的な処理。
 *
 * 写した元（下から順に重なっている）:
 * - EdgeSLM_MultiHead_Multi `dispatch/arguments.py`（元の処理）
 * - `p9_arguments`・`p17_arguments`・`p21_arguments`（元が「足りない」と返したときの補い）
 * - `p22_arguments`（元の処理より先に、主節を見て決める）
 * - `p23_arguments`（発話を整えてから渡す）
 *
 * 入口は {@link #resolve}（いまの処理 = P23）。履歴の無い 1 回目の発話だけを扱う（直前の周辺検索の結果は使わない）。
 * Python と同じ結果になるかは、`tools/p24_java` の `ParityTest` が `work/p24/fixtures/args.tsv`（発話 × 機能 8 つ）で確かめる。
 */
public final class Args {
    private Args() {}

    /** 決めた引数。missing が空でなければ「引数が足りない」（聞き返す） */
    public static final class Arguments {
        public final LinkedHashMap<String, Object> values = new LinkedHashMap<String, Object>();
        public final List<String> missing = new ArrayList<String>();
        public String note = null;       // どう決めたか（P21 の「目的地は道の条件」だけは、後ろの処理が見る）

        public boolean complete() {
            return missing.isEmpty();
        }

        Arguments put(String k, Object v) {
            values.put(k, v);
            return this;
        }

        Arguments miss(String k) {
            missing.add(k);
            return this;
        }

        Arguments note(String s) {
            note = s;
            return this;
        }
    }

    /** 車両の状態（相対の変更に使う）。EdgeSLM_MultiHead_Multi の `vehicle_state.json` の見本と同じ値 */
    public static final class State {
        public boolean hvacPower = true;
        public Double temperature = 25.0;
        public Integer fanLevel = 3;
        public boolean mediaPlaying = false;
        public Integer volume = 12;
        public String motion = "stopped";
        public String permission = "allowed";
    }

    static final String HVAC = "control_hvac", MEDIA = "control_media", NAV = "set_navigation";
    static final String EDGE = "、。 　";

    static double step(String slot) {
        return "volume".equals(slot) ? 2 : 1;
    }

    static double lo(String slot) {
        return "temperature".equals(slot) ? 16 : 0;
    }

    static double hi(String slot) {
        return "temperature".equals(slot) ? 30 : "fan_level".equals(slot) ? 7 : 40;
    }

    static double clamp(String slot, double v) {
        return Math.max(lo(slot), Math.min(hi(slot), v));
    }

    static Double current(State st, String slot) {
        if ("temperature".equals(slot)) return st.temperature;
        if ("fan_level".equals(slot)) return st.fanLevel == null ? null : Double.valueOf(st.fanLevel);
        return st.volume == null ? null : Double.valueOf(st.volume);
    }

    /** 温度は小数、風量と音量は整数（Python の値の型と同じ） */
    static Object typed(String slot, double v) {
        if ("temperature".equals(slot)) return Double.valueOf(v);
        return Integer.valueOf((int) v);
    }

    // ================================================================ EdgeSLM_MultiHead_Multi arguments.py
    static final Pattern NUM = Pattern.compile("[0-9０-９]+(?:[.．][0-9０-９]+)?");
    static final Pattern B_TARGET_VALUE = Pattern.compile("[0-9０-９]+(?:[.．][0-9０-９]+)?[" + Py.WS + "]*(?:度|段階|段|つ|目盛り?)?[" + Py.WS + "]*(?:に|まで|へ)");
    static final String[] B_FAN_WORDS = {"風量", "送風", "ファン", "風"};
    static final String[] B_TEMP_WORDS = {"温度", "度", "設定温度", "暖か", "涼し", "冷や", "暖め"};
    static final String[] B_VOL_WORDS = {"音量", "ボリューム", "音を", "音が", "音は", "ボリュームを"};
    static final String[] UP_WORDS = {"上げ", "強く", "大きく", "暖か", "暖め", "温め", "熱く", "寒い", "寒く", "冷える", "冷え", "聞こえにく", "聞き取れな", "小さいな"};
    static final String[] DOWN_WORDS = {"下げ", "落とし", "落として", "弱く", "小さく", "涼し", "冷や", "静か", "暑い", "暑く", "うるさい", "蒸し", "汗ば", "じめじめ"};
    static final String[] MAX_WORDS = {"最大", "全開", "一番", "マックス", "最強", "最高", "フルパワー", "フル"};
    static final String[] MIN_WORDS = {"最小", "最低", "一番弱", "最弱", "全閉"};
    static final String[] TOGGLE_WORDS = {"切り替え", "切替", "つけて", "入れて", "オン", "オフ", "消して", "切って", "ミュート", "無音", "停止", "止めて", "鳴らして"};

    static List<Double> numbers(String u) {
        List<Double> out = new ArrayList<Double>();
        Matcher m = NUM.matcher(u);
        while (m.find()) out.add(Double.valueOf(Py.zenToHan(m.group())));
        return out;
    }

    static int direction(String u) {
        if (Py.any(u, UP_WORDS)) return 1;
        if (Py.any(u, DOWN_WORDS)) return -1;
        return 0;
    }

    static boolean isRelativeAmount(String u) {
        return direction(u) != 0 && !Py.find(B_TARGET_VALUE, u);
    }

    static String inferArgmode(String u) {
        if (Py.any(u, MAX_WORDS) || Py.any(u, MIN_WORDS)) return "MAX_MIN";
        boolean num = Py.find(NUM, u);
        if (num && isRelativeAmount(u)) return direction(u) > 0 ? "INCREASE" : "DECREASE";
        if (num) return "ABSOLUTE";
        if (Py.any(u, UP_WORDS)) return "INCREASE";
        if (Py.any(u, DOWN_WORDS)) return "DECREASE";
        if (Py.any(u, TOGGLE_WORDS)) return "TOGGLE";
        return "UNSPECIFIED";
    }

    static Arguments baseHvac(String u, State st) {
        String mode = inferArgmode(u);
        String slot = Py.any(u, B_FAN_WORDS) ? "fan_level" : Py.any(u, B_TEMP_WORDS) ? "temperature" : null;
        Arguments out = new Arguments();
        if (Py.any(u, "切って", "消して", "止めて", "オフ")) return out.put("mode", "off");
        if (Py.any(u, "つけて", "入れて", "オン", "起動")) return out.put("mode", "auto");
        if (u.contains("デフロスタ") || u.contains("曇り")) return out.put("mode", "defrost");
        if ("ABSOLUTE".equals(mode)) {
            List<Double> nums = numbers(u);
            if (nums.isEmpty()) return out.miss(slot != null ? slot : "temperature");
            if (slot == null) slot = (nums.get(0) <= 7 && !u.contains("度")) ? "fan_level" : "temperature";
            return out.put(slot, typed(slot, nums.get(0)));
        }
        if ("INCREASE".equals(mode) || "DECREASE".equals(mode)) {
            if (slot == null) slot = "temperature";
            int sign = direction(u);
            if (sign == 0) sign = "INCREASE".equals(mode) ? 1 : -1;
            Double base = current(st, slot);
            if (base == null) return out.miss(slot);
            List<Double> nums = numbers(u);
            double stp = nums.isEmpty() ? step(slot) : nums.get(0);
            return out.put(slot, typed(slot, clamp(slot, base + sign * stp)));
        }
        if ("MAX_MIN".equals(mode)) {
            if (slot == null) slot = "fan_level";
            boolean toMax = Py.any(u, MAX_WORDS) || !Py.any(u, MIN_WORDS);
            return out.put(slot, typed(slot, toMax ? hi(slot) : lo(slot)));
        }
        if ("TOGGLE".equals(mode)) return out.put("mode", st.hvacPower ? "off" : "auto");
        if (slot != null) return out.miss(slot);
        return out.put("mode", "auto");
    }

    static Arguments baseMedia(String u, State st) {
        String mode = inferArgmode(u);
        Arguments out = new Arguments();
        if (Py.any(u, "ミュート", "消音")) return out.put("action", "mute");
        if (Py.any(u, "次", "スキップ")) return out.put("action", "next");
        if (u.contains("戻") && Py.any(u, "音", "ボリューム") && !u.contains("曲")) {
            int restored = (st.volume == null || st.volume == 0) ? 12 : st.volume;
            return out.put("action", "volume_set").put("volume", Integer.valueOf(restored > 0 ? restored : 12));
        }
        if (Py.any(u, "前の曲", "戻して", "一つ前")) return out.put("action", "previous");
        if (Py.any(u, "止め", "停止", "ポーズ", "消して")) return out.put("action", "pause");
        boolean slotVolume = Py.any(u, B_VOL_WORDS);
        if ("ABSOLUTE".equals(mode)) {
            List<Double> nums = numbers(u);
            if (nums.isEmpty()) return out.put("action", "play");
            return out.put("action", "volume_set").put("volume", Integer.valueOf((int) (double) nums.get(0)));
        }
        if ("INCREASE".equals(mode) || "DECREASE".equals(mode)) {
            int sign = direction(u);
            if (sign == 0) sign = "INCREASE".equals(mode) ? 1 : -1;
            if (st.volume == null) return out.miss("volume");
            List<Double> nums = numbers(u);
            double stp = nums.isEmpty() ? step("volume") : nums.get(0);
            return out.put("action", "volume_set").put("volume", Integer.valueOf((int) clamp("volume", st.volume + sign * stp)));
        }
        if ("MAX_MIN".equals(mode)) {
            boolean toMax = Py.any(u, MAX_WORDS) || !Py.any(u, MIN_WORDS);
            return out.put("action", "volume_set").put("volume", Integer.valueOf(toMax ? 40 : 0));
        }
        if ("TOGGLE".equals(mode)) return out.put("action", st.mediaPlaying ? "pause" : "play");
        if (slotVolume) return out.miss("volume");
        return out.put("action", "play");
    }

    static final Pattern DESTINATION_HINT = Pattern.compile("([^" + Py.WS + "、。]{2,12}?)(?:(?:まで|へ|に)(?:案内|ナビ|ルート|行|向か|停め|連れ|変え|寄)|への(?:ルート|案内|道))");
    static final String[] ANAPHORIC_DESTINATION = {"そこ", "あそこ", "ここ", "その", "あの", "この", "さっきの", "例の", "一番近い", "最寄り", "近いところ", "そちら", "あちら"};
    static final String[] DESTINATION_PREFIXES = {"やっぱり", "やっぱ", "じゃあ", "それなら", "それじゃ", "なら", "行き先を", "行き先は", "目的地を", "目的地は", "あと"};
    static final String[][] CATEGORY_HINTS = {
            {"ガソリンスタンド", "スタンド", "給油", "ガソリン"}, {"駐車場", "駐車", "パーキング", "停め"}, {"レストラン", "店", "レストラン", "食事", "ラーメン", "カフェ"},
            {"コンビニ", "コンビニ"}, {"充電スタンド", "充電"}};

    static String cleanDestination(String text) {
        boolean changed = true;
        while (changed) {
            changed = false;
            for (String prefix : DESTINATION_PREFIXES) {
                if (text.startsWith(prefix) && text.length() > prefix.length()) {
                    text = Py.lstrip(text.substring(prefix.length()), "、 ");
                    changed = true;
                }
            }
        }
        return text;
    }

    static boolean isAnaphoricDestination(String text) {
        return Py.any(text, ANAPHORIC_DESTINATION);
    }

    static Arguments baseNavigation(String u) {
        Arguments out = new Arguments();
        if (Py.any(u, "やめ", "中止", "キャンセル", "取り消")) return out.put("action", "cancel");
        if (Py.any(u, "再開", "もう一度")) return out.put("action", "reroute");
        Matcher m = DESTINATION_HINT.matcher(u);
        boolean found = m.find();
        if (isAnaphoricDestination(found ? m.group(1) : u)) return out.put("action", "set").miss("destination");
        if (found) return out.put("action", "set").put("destination", cleanDestination(m.group(1)));
        return out.put("action", "set").miss("destination");
    }

    static final String[][] CAMERA_VIEWS = {{"rear", "後ろ", "後方", "バック", "リア"}, {"front", "前", "前方", "フロント"}, {"left", "左"}, {"right", "右"},
            {"surround", "周囲", "俯瞰", "全体", "まわり"}, {"cabin", "車内", "後席", "室内"}};
    static final String[][] STATUS_ITEMS = {{"odometer", "走行距離", "オドメーター", "総走行"}, {"tire_pressure", "空気圧", "タイヤ"}, {"battery", "バッテリー", "電圧"},
            {"fuel", "ガソリン", "燃料", "残量", "給油"}, {"range", "航続", "あと何キロ", "走れる", "あとどれくらい走"}, {"door", "ドア", "開いて", "閉まって"}};
    static final String[][] MEDIA_INFO_ITEMS = {{"artist", "誰", "アーティスト", "歌手"}, {"album", "アルバム"}, {"queue", "次", "この後"}, {"source", "ソース", "どこから", "入力"}};

    static String firstOf(String u, String[][] table, String otherwise) {
        for (String[] row : table) {
            for (int i = 1; i < row.length; i++) {
                if (u.contains(row[i])) return row[0];
            }
        }
        return otherwise;
    }

    static Arguments base(String tool, String u, State st) {
        if (HVAC.equals(tool)) return baseHvac(u, st);
        if (MEDIA.equals(tool)) return baseMedia(u, st);
        if (NAV.equals(tool)) return baseNavigation(u);
        if ("capture_camera".equals(tool)) return new Arguments().put("view", firstOf(u, CAMERA_VIEWS, "rear"));
        if ("get_vehicle_status".equals(tool)) return new Arguments().put("item", firstOf(u, STATUS_ITEMS, "all"));
        if ("get_media_info".equals(tool)) return new Arguments().put("item", firstOf(u, MEDIA_INFO_ITEMS, "now_playing"));
        if ("search_poi".equals(tool) || "search_vehicle_manual".equals(tool)) return new Arguments().put("query", Py.strip(u));
        return new Arguments().miss("unknown_tool");
    }

    // ================================================================ p9_arguments
    static final String[] NAV_WORDS = {"ナビ", "案内", "ルート", "目的地", "行き先", "行先"};
    static final String[] P9_CANCEL = {"消して", "終了", "いらない", "止めて", "ストップ", "やめ", "中止", "キャンセル", "取り消"};
    static final String[] P9_REROUTE = {"別のルート", "他のルート", "他の道", "別の道", "引き直", "高速", "下道", "一般道", "有料道路", "有料を"};
    static final String[] P9_WAYPOINT = {"経由", "寄り道", "途中"};
    static final String HOME = "家|自宅|うち|実家|会社|職場";
    static final String DEST = "目的地|行き先|行先|デスティネーション|ゴール";
    static final Pattern P9_CLAUSE = Pattern.compile("、|から|ので|けど|して|たら|なら");
    static final String[] P9_ADVERBS = {"いったん", "一度", "とりあえず", "まず", "ひとまず", "今から", "これから"};
    static final Pattern[] P9_PATTERNS = {
            Pattern.compile("(?:" + DEST + ")[をは](.+?)に(?:して|設定|変更|セット|変え)"),
            Pattern.compile("(.+?)を(?:" + DEST + ")に"),
            Pattern.compile("(.+?)(?:まで|へ)(?:お願い|の道|のルート|の案内|の道案内|連れ|ナビ|案内)"),
            Pattern.compile("(" + HOME + ")(?:まで|へ|に)"),
            Pattern.compile("(.+?)に(?:帰|戻)")};
    static final String[][] ZONES = {{"rear", "後ろ", "後席", "リア"}, {"passenger", "助手席"}, {"driver", "運転席"}};
    static final String[] P9_UP = {"強め", "上げ", "強く"};
    static final String[] P9_DOWN = {"弱め", "下げ", "弱く", "絞"};
    static final String[] LOUD = {"大きい", "大きすぎ", "うるさ", "でかい"};
    static final String[] QUIET = {"小さい", "小さすぎ", "聞こえな"};

    /** 前置きを落として、最後のかたまりを目的地の候補にする（_CLAUSE.split(text)[-1]） */
    static String lastChunk(String text) {
        Matcher m = P9_CLAUSE.matcher(text);
        int last = 0;
        while (m.find()) last = m.end();
        String t = Py.strip(text.substring(last));
        for (String a : P9_ADVERBS) {
            if (t.startsWith(a)) t = t.substring(a.length());
        }
        return Py.strip(t, "、 をはに");
    }

    static Arguments p9Navigation(String u, Arguments base) {
        if (Py.any(u, NAV_WORDS) && Py.any(u, P9_CANCEL)) return new Arguments().put("action", "cancel");
        if (Py.any(u, P9_REROUTE)) return new Arguments().put("action", "reroute");
        if (Py.any(u, P9_WAYPOINT)) {
            for (String[] row : CATEGORY_HINTS) {
                for (int i = 1; i < row.length; i++) {
                    if (u.contains(row[i])) return new Arguments().put("action", "add_waypoint").put("destination", row[0]);
                }
            }
            return base;
        }
        for (Pattern pat : P9_PATTERNS) {
            Matcher m = pat.matcher(u);
            if (!m.find()) continue;
            String dest = cleanDestination(lastChunk(m.group(1)));
            if (dest.isEmpty() || dest.length() > 15 || isAnaphoricDestination(dest) || Py.any(dest, NAV_WORDS)) continue;
            return new Arguments().put("action", "set").put("destination", dest);
        }
        return base;
    }

    static Arguments p9Hvac(String u, State st, Arguments base) {
        String slot = base.missing.isEmpty() ? null : base.missing.get(0);
        if (("fan_level".equals(slot) || "temperature".equals(slot)) && current(st, slot) != null) {
            int sign = Py.any(u, P9_UP) ? 1 : Py.any(u, P9_DOWN) ? -1 : 0;
            if (sign != 0) return new Arguments().put(slot, typed(slot, clamp(slot, current(st, slot) + sign * step(slot))));
        }
        String zone = firstOf(u, ZONES, null);
        if (zone != null) return new Arguments().put("zone", zone);
        return base;
    }

    static Arguments volumeSet(double v) {
        return new Arguments().put("action", "volume_set").put("volume", Integer.valueOf((int) v));
    }

    static Arguments p9Media(String u, State st, Arguments base) {
        if (st.volume == null) return base;
        if (u.contains("半分")) return volumeSet(Math.floor(st.volume / 2.0));
        int sign = Py.any(u, LOUD) ? -1 : Py.any(u, QUIET) ? 1 : 0;
        if (sign != 0) return volumeSet(clamp("volume", st.volume + sign * step("volume")));
        return base;
    }

    static Arguments p9(String tool, String u, State st) {
        Arguments base = base(tool, u, st);
        if (base.complete()) return base;
        if (NAV.equals(tool)) return p9Navigation(u, base);
        if (HVAC.equals(tool)) return p9Hvac(u, st, base);
        if (MEDIA.equals(tool)) return p9Media(u, st, base);
        return base;
    }

    // ================================================================ p17_arguments
    static final String[] REROUTE_MORE = {"別ルート", "最短", "案内し直", "ルートを探", "近道", "早い道", "速い道"};
    static final Pattern PURPOSE = Pattern.compile("^.*を[ぁ-ん一-龥]{1,6}に(?=[^にをは]+$)");
    static final Pattern[] DEST_MORE = {
            Pattern.compile("(?:" + DEST + ")は(.+?)(?:で|に|へ|まで)?(?:お願い|頼む|して)?$"),
            Pattern.compile("(.+?)(?:まで|へ)(?:道案内|ナビゲート|ガイド)"),
            Pattern.compile("(.+?)を(?:" + DEST + ")に")};
    static final Pattern[] P17_WAYPOINT = {
            Pattern.compile("経由地[にへ](.+?)を(?:追加|入れ|設定|セット|足)"),
            Pattern.compile("(.+?)を経由地[にへ]"),
            Pattern.compile("(?:途中で|途中に)(.+?)[にへ]寄")};
    static final String[] SOFTER = {"小さめ", "控えめ", "静かめ", "低め"};
    static final String[] LOUDER = {"大きめ"};

    static boolean okDest(String dest) {
        return dest.length() > 0 && dest.length() <= 15 && !isAnaphoricDestination(dest) && !Py.any(dest, NAV_WORDS);
    }

    static String afterLastComma(String s) {
        int i = s.lastIndexOf('、');
        return i < 0 ? s : s.substring(i + 1);
    }

    static String dropPurpose(String s) {
        return PURPOSE.matcher(s).replaceFirst("");
    }

    static Arguments p17Navigation(String u, Arguments base) {
        if (Py.any(u, REROUTE_MORE)) return new Arguments().put("action", "reroute");
        for (Pattern pat : P17_WAYPOINT) {
            Matcher m = pat.matcher(u);
            if (m.find()) {
                String dest = cleanDestination(Py.strip(afterLastComma(m.group(1))));
                if (okDest(dest)) return new Arguments().put("action", "add_waypoint").put("destination", dest);
            }
        }
        String text = Py.strip(afterLastComma(u));
        String[] cands = {text, dropPurpose(text)};
        for (String cand : cands) {
            for (Pattern pat : DEST_MORE) {
                Matcher m = pat.matcher(cand);
                if (!m.find()) continue;
                String dest = cleanDestination(Py.strip(dropPurpose(m.group(1)), "、 をはに"));
                if (okDest(dest)) return new Arguments().put("action", "set").put("destination", dest);
            }
        }
        return base;
    }

    static Arguments p17(String tool, String u, State st) {
        Arguments base = p9(tool, u, st);
        if (base.complete()) return base;
        if (NAV.equals(tool)) return p17Navigation(u, base);
        if (MEDIA.equals(tool) && st.volume != null) {
            int sign = Py.any(u, SOFTER) ? -1 : Py.any(u, LOUDER) ? 1 : 0;
            if (sign != 0) return volumeSet(clamp("volume", st.volume + sign * step("volume")));
        }
        return base;
    }

    // ================================================================ p21_arguments
    static final String[] CANCEL_MORE = {"終わり", "おしまい", "切って", "オフ"};
    static final Pattern ROUTE_END = Pattern.compile("[ぁ-ん](?:道|道路|ルート|経路|コース)$");
    static final Pattern ROUTE_COND = Pattern.compile("[ぁ-ん](?:道|道路|ルート|経路|コース)(?:で|に|を|が)?(?:行|案内|変え|向か|お願い|頼|し)");
    static final String[] AGAIN_WORDS = {"もう一度", "再開"};
    static final Pattern PLACE_DEICTIC = Pattern.compile("あそこ|そこ(?:へ|に|まで)|(?:あの|その|さっきの|例の|この前の|この間の)(?:場所|店|ところ|辺|へん|駅|あたり)");
    static final String[] P21_AUTO = {"おまかせ", "お任せ", "自動", "オート"};
    static final String[] DOWN_MEDIA = {"絞", "控え"};
    static final String NOTE_P21_ROUTE = "P21: 目的地";

    static Arguments p21(String tool, String u, State st) {
        Arguments base = p17(tool, u, st);
        if (NAV.equals(tool)) {
            Object action = base.values.get("action");
            Object dest = base.values.get("destination");
            if (base.complete() && "set".equals(action) && Py.find(ROUTE_END, dest == null ? "" : String.valueOf(dest))) {
                return new Arguments().put("action", "reroute").note(NOTE_P21_ROUTE);
            }
            if (base.complete() && "reroute".equals(action) && Py.any(u, AGAIN_WORDS) && Py.find(PLACE_DEICTIC, u)) {
                return new Arguments().put("action", "set").miss("destination");
            }
            if (base.complete()) return base;
            if (Py.any(u, NAV_WORDS) && Py.any(u, CANCEL_MORE)) return new Arguments().put("action", "cancel");
            if (Py.find(ROUTE_COND, u) && !isAnaphoricDestination(u)) return new Arguments().put("action", "reroute");
            return base;
        }
        if (base.complete()) return base;
        if (HVAC.equals(tool) && Py.any(u, P21_AUTO)) return new Arguments().put("mode", "auto");
        if (MEDIA.equals(tool) && Py.any(u, DOWN_MEDIA) && st.volume != null) return volumeSet(clamp("volume", st.volume - step("volume")));
        return base;
    }

    // ================================================================ p22_arguments
    static final Pattern CLAUSE_END = Pattern.compile("、|。|(?<=[いたるだすくうつむぶぬぐん])から|ので|のに|けれど|けど|(?<=[っいえしきちりみにけめれせげべてねんだ])たら|あいだ|間に|間は|前に|うちに|ように");
    static final Pattern TARGET_VALUE = Pattern.compile("[0-9０-９]+(?:[.．][0-9０-９]+)?[" + Py.WS + "]*(?:度|段階|段目|段|つ|目盛り?)?[" + Py.WS + "]*(?:に|まで|へ|で)");
    static final Pattern MAX = Pattern.compile("最大|全開|マックス|最強|最高|フルパワー|めいっぱい|目一杯|目いっぱい|フル(?:に|で)|(?:一番|いちばん)(?:強|大|高|上)");
    static final Pattern MIN = Pattern.compile("最小|最低|最弱|全閉|(?:一番|いちばん)(?:弱|小|低|下)");
    static final Pattern NEG = Pattern.compile("(?:止め|とめ|切ら|消さ|下げ|上げ|変え)ないで");
    static final String[] H_TARGET = {"冷房", "暖房", "エアコン", "空調", "クーラー", "ヒーター", "風", "送風", "ファン"};
    static final String[] FAN_WORD = {"風量", "風", "送風", "ファン"};
    static final String[] TEMP_WORD = {"温度", "設定温度"};
    static final String[] STRONG = {"強く", "強め", "強化", "足し", "パワーを上げ", "パワーをアップ", "パワーアップ"};
    static final String[] WEAK = {"弱く", "弱め", "控えめ", "抑え", "落とし", "落として", "ゆるめ", "ゆるく", "絞"};
    static final String[] H_UP = {"上げ", "高く", "高め"};
    static final String[] H_DOWN = {"下げ", "低く", "低め"};
    static final String[] WARMER = {"暖かく", "暖かめ", "暖め", "温め", "あたたかく", "あったかく", "ぬくく"};
    static final String[] COOLER = {"涼しく", "涼しめ", "冷やし", "冷やす", "冷たく", "すずしく"};
    static final String[] H_OFF = {"切っ", "消し", "止め", "とめて", "オフ", "停止", "いらない"};
    static final Pattern DEFROST = Pattern.compile("デフロスタ|曇り(?:取|止|ど)|くもり(?:取|止)|曇りを|霜取|結露");
    static final String[] DEFROST_OFF = {"オフ", "切っ", "止めて", "消し", "解除", "やめ", "いらない"};
    static final String[] AUTO = {"オート", "自動", "おまかせ", "お任せ"};
    static final String[] HANGING = {"を", "が", "の", "は", "に", "で", "と", "も", "へ", "って", "けど", "えっと", "あの", "その", "うーん"};

    /** （前置き, 主節）。前置きが無ければ（"", 発話） */
    static String[] splitMain(String u) {
        int end = 0;
        Matcher m = CLAUSE_END.matcher(u);
        while (m.find()) {
            if (Py.strip(u.substring(m.end()), EDGE).length() >= 2) end = m.end();
        }
        return new String[]{u.substring(0, end), Py.strip(u.substring(end), EDGE)};
    }

    static Double firstNumber(String m) {
        Matcher x = NUM.matcher(m);
        return x.find() ? Double.valueOf(Py.zenToHan(x.group())) : null;
    }

    static Arguments fan(State st, int sign) {
        return new Arguments().put("fan_level", typed("fan_level", clamp("fan_level", st.fanLevel + sign * step("fan_level"))));
    }

    /** 決まらなければ null（これまでの処理に任せる） */
    static Arguments p22Hvac(String u, State st) {
        String[] pm = splitMain(u);
        String pre = pm[0], m = pm[1];
        if (Py.find(NEG, m)) return null;
        if (Py.find(DEFROST, m) || (u.contains("曇") && Py.any(m, "取って", "取り", "取る", "とって"))) {
            if (Py.any(m, DEFROST_OFF) && !Py.any(m, "曇り止め", "くもり止め")) return null;
            return new Arguments().put("mode", "defrost");
        }
        if (Py.any(m, H_OFF)) return new Arguments().put("mode", "off");
        Double n = firstNumber(m);
        if (n != null) {
            double v = n;
            boolean isFan = Py.any(m, FAN_WORD) && !m.contains("度");
            String slot = isFan ? "fan_level" : (m.contains("度") || Py.any(m, TEMP_WORD)) ? "temperature" : null;
            if (slot == null) return null;
            int sign = (Py.any(m, H_UP) || Py.any(m, WARMER) || Py.any(m, STRONG)) ? 1 : (Py.any(m, H_DOWN) || Py.any(m, COOLER) || Py.any(m, WEAK)) ? -1 : 0;
            if (sign != 0 && !Py.find(TARGET_VALUE, m) && current(st, slot) != null) v = clamp(slot, current(st, slot) + sign * v);
            return new Arguments().put(slot, typed(slot, v));
        }
        boolean isMin = Py.find(MIN, m);
        if (isMin || Py.find(MAX, m)) {
            String slot = Py.any(m, TEMP_WORD) ? "temperature" : "fan_level";
            return new Arguments().put(slot, typed(slot, isMin ? lo(slot) : hi(slot)));
        }
        boolean target = Py.any(m, H_TARGET) || Py.any(pre, H_TARGET);
        boolean up = Py.any(m, H_UP), down = Py.any(m, H_DOWN);
        boolean updown = Py.any(m, "冷房", "暖房", "クーラー", "ヒーター") && (up || down);
        boolean strong = Py.any(m, STRONG), weak = Py.any(m, WEAK);
        if ((strong || weak || updown) && target && !Py.any(m, TEMP_WORD) && st.fanLevel != null) {
            return fan(st, (strong || (updown && up)) ? 1 : -1);
        }
        if (Py.any(m, FAN_WORD) && !Py.any(m, TEMP_WORD) && (up || down) && st.fanLevel != null) return fan(st, up ? 1 : -1);
        if (st.temperature != null) {
            int sign = 0;
            if (Py.any(m, TEMP_WORD) && (up || down)) sign = up ? 1 : -1;
            else if (Py.any(m, WARMER) || Py.any(m, COOLER)) sign = Py.any(m, WARMER) ? 1 : -1;
            if (sign != 0) return new Arguments().put("temperature", typed("temperature", clamp("temperature", st.temperature + sign * step("temperature"))));
        }
        if (Py.any(m, AUTO)) return new Arguments().put("mode", "auto");
        if (Py.endsWithAny(m, HANGING)) return null;
        if (m.contains("暖房") || m.contains("ヒーター")) return new Arguments().put("mode", "heat");
        if (m.contains("冷房") || m.contains("クーラー")) return new Arguments().put("mode", "cool");
        return null;
    }

    static final String[] MUTE = {"ミュート", "消音", "無音"};
    static final String[] UNMUTE = {"解除", "やめ", "取りやめ", "オフ", "外し", "戻"};
    static final String[] RESUME = {"続き", "再開", "続けて"};
    static final Pattern NEXT = Pattern.compile("次の曲|次へ|次に|スキップ|飛ばし|曲先|先の曲|曲送り|次を|送って");
    static final Pattern PREV = Pattern.compile("前の曲|(?:一|ひと|1|１)つ前|(?:一|1|１)曲前|曲.{0,4}戻|戻って|巻き戻");
    static final Pattern AGAIN = Pattern.compile("(?:最初|頭|はじめ)から|もう(?:一回|一度|いっかい|1回)");
    static final String[] STOP = {"止め", "とめて", "停止", "ポーズ", "ストップ"};
    static final Pattern V_SLOT = Pattern.compile("音量|ボリューム|音(?!楽|声|源)");
    static final String[] V_UP = {"上げ", "大きく", "大きめ", "足し", "増や"};
    static final String[] V_DOWN = {"下げ", "小さく", "小さめ", "絞", "控え", "落とし", "落として", "減ら", "静かに"};
    static final Pattern V_UPDOWN = Pattern.compile("(?:音量|ボリューム|音)(?:を|の)?(アップ|ダウン)");
    static final Pattern NOT_VOLUME_UNIT = Pattern.compile("[0-9０-９]+(?:[.．][0-9０-９]+)?[" + Py.WS + "]*(?:曲|分|秒|番|回|時|倍|人|件|日|年|個目|番目)");
    static final String[] PLAY = {"かけ", "流し", "流す", "再生", "つけ", "聴", "聞きたい", "切り替え", "替え", "お願い"};

    static Arguments p22Media(String u, State st) {
        Integer vol = st.volume;
        String[] pm = splitMain(u);
        String pre = pm[0], m = pm[1];
        if (Py.find(NEG, m)) return null;
        if (Py.any(m, MUTE)) {
            if (Py.any(m, UNMUTE) && vol != null) return volumeSet(vol > 0 ? vol : 12);
            return new Arguments().put("action", "mute");
        }
        boolean slot = Py.find(V_SLOT, m);
        if (slot && Py.any(m, "消し", "消す")) return new Arguments().put("action", "mute");
        if (Py.any(m, RESUME)) return new Arguments().put("action", "play");
        if (Py.find(NEXT, m)) return new Arguments().put("action", "next");
        if (Py.find(PREV, m) && !slot) return new Arguments().put("action", "previous");
        if (Py.find(AGAIN, m)) return new Arguments().put("action", "previous");
        if (Py.any(m, STOP) || (m.contains("消し") && Py.any(m, "音楽", "曲", "ラジオ", "再生"))) return new Arguments().put("action", "pause");
        int sign = Py.any(m, V_UP) ? 1 : Py.any(m, V_DOWN) ? -1 : 0;
        if (sign == 0) {
            Matcher x = V_UPDOWN.matcher(m);
            if (x.find()) sign = "アップ".equals(x.group(1)) ? 1 : -1;
        }
        Double n = firstNumber(m);
        if (n != null && !Py.find(NOT_VOLUME_UNIT, m) && (slot || sign != 0) && vol != null) {
            double v = n;
            if (sign != 0 && !Py.find(TARGET_VALUE, m)) v = clamp("volume", vol + sign * v);
            return volumeSet(v);
        }
        boolean isMin = Py.find(MIN, m);
        if ((isMin || Py.find(MAX, m)) && (slot || sign != 0)) return volumeSet(isMin ? 0 : 40);
        if (sign != 0 && vol != null && (slot || Py.find(V_SLOT, pre) || Py.any(m, "音楽", "曲", "ラジオ") || !Py.any(m, PLAY))) {
            return volumeSet(clamp("volume", vol + sign * step("volume")));
        }
        if (Py.any(m, PLAY)) return new Arguments().put("action", "play");
        return null;
    }

    static final String[] ADVERBS = {"いったん", "一度", "とりあえず", "まずは", "まず", "ひとまず", "今から", "これから", "いまから", "途中で", "途中に", "途中", "ついでに", "急いで", "いそいで"};
    static final Pattern VAGUE_PLACE = Pattern.compile("さっき|いつも|こないだ|この前|この間|前回|昨日|先日|例の|(?:ところ|とこ|場所|あたり|辺|へん)$");
    static final Pattern SELF_FIX = Pattern.compile("、(?:いや|やっぱ|違う|ちがう|じゃなくて)");
    static final Pattern BOUNDARY = Pattern.compile("、|。|から|ので|けど|たら|なら|まで|を|前に|うちに|あいだに|間に|ついでに|ではなくて|じゃなくて|ではなく|じゃなく"
            + "|(?<=[一-龥ァ-ヶー々])で|(?<=[いきしちにひみりぎじびえけせてねへめれげぜべ])に|(?<=を[一-龥])に");
    static final Pattern WEAK_BOUNDARY = Pattern.compile("[がは]");
    static final Pattern WAY = Pattern.compile("経由|寄り道|途中|寄(?:る|れる)(?:ルート|道|よう)|寄ってから|寄りたい");
    static final Pattern[] WAY_PAT = {
            Pattern.compile("経由地?[にへ](.+?)を(?:追加|入れ|設定|セット|足|加え)"), Pattern.compile("(.+?)を経由"), Pattern.compile("(.+?)[にへ](?:寄|立ち寄)"), Pattern.compile("(.+?)経由で")};
    static final Pattern[] DEST_PAT = {
            Pattern.compile("(.+?)を(?:" + DEST + ")に"),
            Pattern.compile("(?:" + DEST + ")[をは](.+?)(?:に(?:して|設定|変更|セット|変え|お願い)|で(?:お願い|頼)|へ変)"),
            Pattern.compile("(.+?)(?<![るくつうすむぶぬぐ])まで(?=案内|ナビ|お願い|連れ|行|向か|頼|出発|走|進|の(?:ルート|案内|道|ナビ|行き方)|道案内|ルート|急|戻|帰|迎え|取り|届け|運|乗せ|、|$)"),
            Pattern.compile("(.+?)へ(?=案内|ナビ|お願い|連れ|行|向か|頼|出発|走|進|の(?:ルート|案内|道|ナビ|行き方)|道案内|ルート|急|戻|帰|$)"),
            Pattern.compile("(.+?)に(?=向か|行っ|行き|行く|案内|ナビ|連れ|戻|帰|停め)")};
    static final Pattern CLEAR = Pattern.compile("(?:" + DEST + "|案内|ルート|ナビ).{0,6}(?:なし|無し|クリア|削除|リセット|消し|消去|解除)");
    static final Pattern ROUTE_MORE = Pattern.compile("道順|(?:渋滞|混雑|工事|事故|通行止め).{0,4}(?:避け|回避|よけ|迂回)|迂回|(?:違う|ちがう|ほかの|他の|別の)(?:道|ルート|行き方)|(?:ルート|経路|道順|行き方)[にをで](?:して|変|切り替|選)");
    static final Pattern ROUTE_IN = Pattern.compile("[ぁ-ん](?:道|道路|ルート|経路|コース|道順)$|^(?:ルート|経路|道順|別ルート|最短|近道)$");

    /** 助詞の後ろだけ（最後の名詞のかたまり）を取る。「の」は残す */
    static String chunk(String text) {
        String t = Py.strip(text, EDGE);
        int cut = 0;
        Pattern[] pats = {BOUNDARY, WEAK_BOUNDARY};
        int[] least = {1, 2};
        for (int k = 0; k < 2; k++) {
            Matcher m = pats[k].matcher(t);
            while (m.find()) {
                if (m.start() > 0 && Py.strip(t.substring(m.end()), EDGE).length() >= least[k]) cut = Math.max(cut, m.end());
            }
        }
        t = Py.strip(t.substring(cut), EDGE);
        boolean changed = true;
        while (changed) {
            changed = false;
            for (String a : ADVERBS) {
                if (t.startsWith(a) && t.length() > a.length()) {
                    t = Py.lstrip(t.substring(a.length()), "、 のを");
                    changed = true;
                }
            }
        }
        return cleanDestination(t);
    }

    static String placeOf(String u, Pattern[] patterns) {
        for (Pattern pat : patterns) {
            Matcher m = pat.matcher(u);
            if (m.find()) {
                String d = chunk(m.group(1));
                if (!d.isEmpty()) return d;
            }
        }
        return null;
    }

    static Arguments askDestination() {
        return new Arguments().put("action", "set").miss("destination");
    }

    static Arguments p22Navigation(String u, Arguments base) {
        Object action = base.values.get("action");
        if ((base.complete() && "cancel".equals(action)) || Py.find(SELF_FIX, u)) return base;
        if (Py.find(CLEAR, u)) return new Arguments().put("action", "cancel");
        if (Py.find(WAY, u)) {
            String d = placeOf(u, WAY_PAT);
            if (d != null && !isAnaphoricDestination(d) && !Py.find(VAGUE_PLACE, d) && !Py.find(ROUTE_IN, d)) {
                return new Arguments().put("action", "add_waypoint").put("destination", d);
            }
            return ((base.complete() && "add_waypoint".equals(action)) || !base.complete()) ? base : askDestination();
        }
        if (base.complete() && "reroute".equals(action) && !(base.note != null && base.note.contains(NOTE_P21_ROUTE))) return base;
        String d = placeOf(u, DEST_PAT);
        if (d == null && Py.find(ROUTE_MORE, u) && !Py.find(PLACE_DEICTIC, u)) return new Arguments().put("action", "reroute");
        if (d == null && base.complete() && "set".equals(action)) {
            Object bd = base.values.get("destination");
            String c = chunk(bd == null ? "" : String.valueOf(bd));
            d = c.isEmpty() ? null : c;
        }
        boolean routeLike = d != null && (Py.find(ROUTE_IN, d) || Py.find(ROUTE_END, d));
        if (d != null && !routeLike) {
            if (isAnaphoricDestination(d) || Py.find(PLACE_DEICTIC, d) || Py.find(VAGUE_PLACE, d)) return !base.complete() ? base : askDestination();
            return new Arguments().put("action", "set").put("destination", d);
        }
        if (routeLike || Py.find(ROUTE_MORE, u)) {
            return (!base.complete() && Py.find(PLACE_DEICTIC, u)) ? base : new Arguments().put("action", "reroute");
        }
        return base;
    }

    static Arguments p22(String tool, String u, State st) {
        if (HVAC.equals(tool)) {
            Arguments x = p22Hvac(u, st);
            return x != null ? x : p21(tool, u, st);
        }
        if (MEDIA.equals(tool)) {
            Arguments x = p22Media(u, st);
            return x != null ? x : p21(tool, u, st);
        }
        Arguments base = p21(tool, u, st);
        if (NAV.equals(tool)) return p22Navigation(u, base);
        return base;
    }

    // ================================================================ p23_arguments
    static final Pattern CONNECT = Pattern.compile("((?<=[たいだるく])んで|(?<=[たるいの])ため(?:に)?|(?<=[いただ])し(?=[一-龥ァ-ヶ])|ければ|(?<=[きえけせてねへめれげべ])れば|(?<=[んい])だら|ように|てから|おかげで|せいで|もんで)(?![、。])");
    static final Pattern REPLACE = Pattern.compile("^.*?(?:じゃなくて|ではなくて|じゃなく|ではなく|はやめて|をやめて|(?<=[ぁ-ん])ないで|(?<=[ぁ-ん])ずに)[、 　]*(?!ください|くれ|ほしい|欲しい|もらえ|いただ|おいて|ね|よ)(?=.{3,}$)");
    static final Pattern REQUEST_END = Pattern.compile("(?:(?<!く)て|で|って|てよ|てね|てー|くれる|くれない|ほしい|ほしいな|ください|お願い|お願いします|頼む|たのむ|ますか|ませんか|よろしく|に|へ|まで)$");
    static final Pattern KANJI_NUM = Pattern.compile("(?<!もう)(?<!今)(?<!いま)([一二三四五六七八九十]+)(?=度|段階|段目|段|つ(?!ぎ)|目盛)"
            + "|(?:(?<=音量を)|(?<=音量は)|(?<=ュームを)|(?<=風量を)|(?<=風量は))([一二三四五六七八九十]+)(?=に|まで|へ|で)");
    static final String KANJI_DIGITS = "一二三四五六七八九";
    static final String[][] KANA = {{"つよめ", "強め"}, {"よわめ", "弱め"}, {"つよく", "強く"}, {"よわく", "弱く"}, {"あげて", "上げて"}, {"さげて", "下げて"}};
    static final Pattern COMMA = Pattern.compile("[、。]");
    static final Pattern ENDS_HIRAGANA = Pattern.compile("[ぁ-ん]$");

    static int digit(String c, int otherwise) {
        int i = KANJI_DIGITS.indexOf(c);
        return (c.length() == 1 && i >= 0) ? i + 1 : otherwise;
    }

    /** 十までの位だけ（二十二 → 22・十八 → 18・三 → 3） */
    static int kanjiToInt(String s) {
        int j = s.indexOf('十');
        if (j >= 0) {
            String a = s.substring(0, j), b = s.substring(j + 1);
            return (a.isEmpty() ? 1 : digit(a, 1)) * 10 + (b.isEmpty() ? 0 : digit(b, 0));
        }
        if (s.length() == 1) return digit(s, 0);
        StringBuilder x = new StringBuilder();
        for (int i = 0; i < s.length(); i++) x.append(digit(s.substring(i, i + 1), 0));
        return Integer.parseInt(x.toString());
    }

    static String kanjiNumbers(String u) {
        Matcher m = KANJI_NUM.matcher(u);
        StringBuffer b = new StringBuffer();
        while (m.find()) {
            String s = m.group(1) != null ? m.group(1) : m.group(2);
            boolean keep = "一".equals(s) && m.group(1) != null && u.startsWith("段", m.end());
            m.appendReplacement(b, Matcher.quoteReplacement(keep ? s : String.valueOf(kanjiToInt(s))));
        }
        m.appendTail(b);
        return b.toString();
    }

    static boolean isRequestEnd(String part) {
        return Py.find(REQUEST_END, Py.strip(part));
    }

    static boolean isComment(String part) {
        String t = Py.strip(part);
        return Py.find(ENDS_HIRAGANA, t) && !Py.find(REQUEST_END, t);
    }

    /** 発話を整える（接続の語の後ろで区切る・打ち消しの前を落とす・後ろに足した理由を落とす・漢数字をそろえる） */
    static String normalize(String u, boolean replace) {
        String v = kanjiNumbers(u);
        for (String[] ab : KANA) v = v.replace(ab[0], ab[1]);
        List<String> parts = new ArrayList<String>();
        for (String p : Py.split(COMMA, v)) {
            if (!Py.strip(p).isEmpty()) parts.add(p);
        }
        if (parts.size() >= 2 && isComment(parts.get(parts.size() - 1))) {
            boolean anyRequest = false;
            for (int i = 0; i < parts.size() - 1; i++) anyRequest |= isRequestEnd(parts.get(i));
            if (anyRequest) {
                while (parts.size() >= 2 && isComment(parts.get(parts.size() - 1))) parts.remove(parts.size() - 1);
                StringBuilder b = new StringBuilder();
                for (int i = 0; i < parts.size(); i++) {
                    if (i > 0) b.append('、');
                    b.append(parts.get(i));
                }
                v = b.toString();
            }
        }
        if (replace) v = REPLACE.matcher(v).replaceFirst("");
        Matcher m = CONNECT.matcher(v);
        StringBuffer b = new StringBuffer();
        while (m.find()) m.appendReplacement(b, Matcher.quoteReplacement(m.group(1) + "、"));
        m.appendTail(b);
        return b.toString();
    }

    static final Pattern TWO = Pattern.compile("(?<=[てで])(?=(?:設定)?温度|風量|風|冷房|暖房|窓|曇|デフロスタ|[0-9０-９]+度)");
    static final String NOUN = "[一-龥ァ-ヶー々0-9０-９A-Za-z]{2,}";
    static final String PLACE_END = "(?:駅|空港|港|店|屋|館|園|場|所|院|局|庁|城|寺|神社|橋|町|市|区|村|県|湖|山|浜|海岸|タワー|センター|ホテル|モール|ストア|ショップ|スタンド|パーキング|ランド|ドーム|ビル|ホール|カフェ"
            + "|コンビニ|スーパー|デパート|会社|自宅|家|実家|学校|大学|公園|銀行|インター|サービスエリア|ジャンクション)";
    static final String PLACE = "[一-龥ァ-ヶー々0-9０-９A-Za-z]*" + PLACE_END;
    static final Pattern[] CASUAL = {
            Pattern.compile("^(?:行き先|目的地|次)[、はを]?[" + Py.WS + "]*(" + PLACE + ")(?:な|ね|で|に)?$"),
            Pattern.compile("^(" + PLACE + ")(?:を|まで|へ|に)?(?:お願い|よろしく|頼む|たのむ)(?:します|ね|な)?$"),
            Pattern.compile("^(" + NOUN + ")(?:を|まで|へ|に)?(?:いって|行って|ナビして|ナビ)(?:ね|な)?$"),
            Pattern.compile("^([一-龥ァ-ヶー々0-9０-９A-Za-z]+)(?:まで|へ)(?:お願い|よろしく|頼む|たのむ)(?:します|ね|な)?$"),
            Pattern.compile("^(家|自宅|うち|実家|会社|職場)(?:に|へ|まで)?(?:帰る|帰りたい|戻る|帰ろう)$"),
            Pattern.compile("^(" + NOUN + ")(?:に)?寄って$")};
    static final Pattern ROUTE_ONLY = Pattern.compile("(?:ルート|経路|道順|行き方)(?:だけ|は|も)?(?:を|に)?(?:変|引き直|切り替|探)");
    static final Pattern WAY_REMOVE = Pattern.compile("経由地?.{0,4}(?:消|削除|外し|取り消|やめ|キャンセル|なし)");
    static final Pattern VOLUME_WORD = Pattern.compile("音量|ボリューム|音を");
    static final Pattern ANY_DIGIT = Pattern.compile("[0-9０-９]");

    /** 温度と風量を 1 文で言う。分けて取り出し、まとめる（どれかが決まらなければ null） */
    static Arguments hvacTwo(String u, State st) {
        List<String> segs = new ArrayList<String>();
        for (String s : Py.split(TWO, splitMain(u)[1])) {
            if (Py.strip(s, EDGE).length() >= 3) segs.add(s);
        }
        if (segs.size() < 2) return null;
        Arguments out = new Arguments();
        for (String s : segs) {
            Arguments x = p22Hvac(s, st);
            if (x == null || !x.complete()) return null;
            for (Map.Entry<String, Object> e : x.values.entrySet()) out.values.put(e.getKey(), e.getValue());
        }
        return out;
    }

    static Arguments p23One(String tool, String u, State st) {
        if (HVAC.equals(tool)) {
            Arguments x = hvacTwo(u, st);
            return x != null ? x : p22(tool, u, st);
        }
        if (MEDIA.equals(tool)) {
            if (u.contains("倍") && Py.find(VOLUME_WORD, u) && st.volume != null && !Py.find(ANY_DIGIT, u)) return volumeSet(clamp("volume", st.volume * 2));
            return p22(tool, u, st);
        }
        if (Py.find(WAY_REMOVE, u)) return askDestination();
        Arguments out = p22(tool, u, st);
        if (out.complete()) return out;
        String t = Py.strip(u, EDGE);
        for (Pattern pat : CASUAL) {
            Matcher m = pat.matcher(t);
            if (m.find()) {
                String g = m.group(1);
                if (!Py.find(PLACE_DEICTIC, g) && !Py.find(VAGUE_PLACE, g) && !isAnaphoricDestination(g)) return new Arguments().put("action", "set").put("destination", g);
            }
        }
        if (Py.find(ROUTE_ONLY, u) && !Py.find(PLACE_DEICTIC, u)) return new Arguments().put("action", "reroute");
        return out;
    }

    /** いまの引数の処理（P23）。履歴の無い 1 回目の発話に使う */
    public static Arguments resolve(String tool, String utterance, State st) {
        if (!HVAC.equals(tool) && !MEDIA.equals(tool) && !NAV.equals(tool)) return p22(tool, utterance, st);
        String u = normalize(utterance, true), plain = normalize(utterance, false);
        Arguments out = p23One(tool, u, st);
        if (!out.complete() && !plain.equals(u)) out = p23One(tool, plain, st);
        return out;
    }

    /** Python の json.dumps(values, ensure_ascii=False, sort_keys=True) に近い形（突き合わせと表示に使う） */
    public static String canon(Map<String, Object> values) {
        if (values == null) return "-";
        List<String> keys = new ArrayList<String>(values.keySet());
        java.util.Collections.sort(keys);
        StringBuilder b = new StringBuilder("{");
        for (int i = 0; i < keys.size(); i++) {
            Object v = values.get(keys.get(i));
            if (i > 0) b.append(", ");
            b.append('"').append(keys.get(i)).append("\": ");
            if (v instanceof Double) b.append(Py.reprDouble((Double) v));
            else if (v instanceof Integer) b.append(v);
            else b.append('"').append(v).append('"');
        }
        return b.append('}').toString();
    }
}
