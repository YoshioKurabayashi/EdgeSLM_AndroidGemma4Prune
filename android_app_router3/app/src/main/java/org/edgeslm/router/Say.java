package org.edgeslm.router;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 判定の結果を、画面に出す日本語にする（P24）。判定そのものには関わらない（表示だけ）。
 */
public final class Say {
    private Say() {}

    static String pick(String key, String... pairs) {
        for (int i = 0; i + 1 < pairs.length; i += 2) {
            if (pairs[i].equals(key)) return pairs[i + 1];
        }
        return key;
    }

    /** 機能の名前 */
    public static String tool(String tool) {
        if (tool == null) return "";
        return pick(tool, "control_hvac", "エアコン操作", "control_media", "音楽・メディア操作", "set_navigation", "ナビの設定", "get_vehicle_status", "車両の状態を調べる",
                "search_vehicle_manual", "取扱説明書を調べる", "capture_camera", "カメラで撮影・表示", "search_poi", "周辺の施設を探す", "get_media_info", "再生中の曲を調べる");
    }

    /** 行き先（CHAT・TOOL_…）の名前 */
    public static String action(String action) {
        if (action == null) return "";
        if (action.equals("CHAT")) return "雑談";
        return action.startsWith("TOOL_") ? tool(action.substring(5)) : action;
    }

    static String num(Object v) {
        if (v instanceof Double) {
            double d = (Double) v;
            return d == Math.rint(d) ? String.valueOf((long) d) : String.valueOf(d);
        }
        return String.valueOf(v);
    }

    /** 実行する中身（例: 「温度 25 度」「目的地: 水族館」） */
    public static String args(String tool, Map<String, Object> v) {
        if (v == null || tool == null) return "";
        List<String> parts = new ArrayList<String>();
        if (tool.equals("control_hvac")) {
            if (v.containsKey("mode")) parts.add(pick(String.valueOf(v.get("mode")), "auto", "オートで入れる", "cool", "冷房を入れる", "heat", "暖房を入れる", "defrost", "曇り取りを入れる", "off", "止める"));
            if (v.containsKey("temperature")) parts.add("温度 " + num(v.get("temperature")) + " 度");
            if (v.containsKey("fan_level")) parts.add("風量 " + num(v.get("fan_level")) + "（0〜7）");
            if (v.containsKey("zone")) parts.add(pick(String.valueOf(v.get("zone")), "driver", "運転席", "passenger", "助手席", "rear", "後席", "all", "全席"));
        } else if (tool.equals("control_media")) {
            String a = String.valueOf(v.get("action"));
            if (a.equals("volume_set")) parts.add("音量 " + num(v.get("volume")) + "（0〜40）");
            else parts.add(pick(a, "play", "再生", "pause", "一時停止", "next", "次の曲へ", "previous", "前の曲へ", "mute", "消音"));
            if (v.containsKey("target")) parts.add(String.valueOf(v.get("target")));
        } else if (tool.equals("set_navigation")) {
            String a = String.valueOf(v.get("action"));
            if (a.equals("set")) parts.add("目的地: " + v.get("destination"));
            else if (a.equals("add_waypoint")) parts.add("経由地に追加: " + v.get("destination"));
            else parts.add(pick(a, "cancel", "案内を中止", "reroute", "ルートを引き直す"));
        } else if (tool.equals("get_vehicle_status")) {
            parts.add(pick(String.valueOf(v.get("item")), "fuel", "燃料", "range", "航続距離", "odometer", "走行距離", "tire_pressure", "タイヤの空気圧", "door", "ドアの状態", "battery", "バッテリー", "all", "全体"));
        } else if (tool.equals("capture_camera")) {
            parts.add(pick(String.valueOf(v.get("view")), "front", "前", "rear", "後ろ", "left", "左", "right", "右", "surround", "周囲", "cabin", "車内"));
        } else if (tool.equals("get_media_info")) {
            parts.add(pick(String.valueOf(v.get("item")), "now_playing", "再生中の曲", "artist", "アーティスト", "album", "アルバム", "source", "音源", "queue", "次の曲"));
        } else if (v.containsKey("query")) {
            parts.add("検索語: " + v.get("query"));
        }
        StringBuilder b = new StringBuilder();
        for (String p : parts) {
            if (b.length() > 0) b.append("・");
            b.append(p);
        }
        return b.toString();
    }

    /** 規則の名前 → 何を見て止めたか */
    public static String rule(String rule) {
        if (rule == null) return "";
        return pick(rule, "ANAPHORA_WITHOUT_HISTORY", "「あれ」「それ」などが何を指すか分からない", "BARE_TARGET_NO_PREDICATE", "対象の語だけで、どうするかが無い",
                "SELF_CONTRADICTION", "文の中に食い違いがある", "S1_REDO", "途中で言い直している", "S1_REDO_WAIT", "「待って」で言い直している",
                "S2_UNDO", "取り消しの依頼（直前の操作が無い）", "S3_HESITATION", "言いよどんで終わっている", "S3_NOUN_BUT", "「〜だけど」で終わっている",
                "S4_TARGET_MISMATCH", "判定した機能と、文中の機能の語が食い違う", "S5_NO_TARGET_TERM", "判定した機能の語が、文中に無い");
    }

    /** 足りない引数の名前 */
    public static String missing(List<String> keys) {
        if (keys == null || keys.isEmpty()) return "";
        StringBuilder b = new StringBuilder();
        for (String k : keys) {
            if (b.length() > 0) b.append("・");
            b.append(pick(k, "destination", "目的地", "volume", "音量", "temperature", "温度", "fan_level", "風量", "mode", "モード", "action", "操作", "unknown_tool", "機能"));
        }
        return b.toString();
    }
}
