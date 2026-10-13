# -*- coding: utf-8 -*-
"""P22: 引数の**値**を直す層。`p21_arguments` の上に重ねる。EdgeSLM_MultiHead_Multi の `arguments.py` は変えない。

設計: DoC/設計書_P22_引数の値を測って直す.md の 4 節。

これまでの補いの層（P9・P17・P21）は、元の処理が「引数が足りない」と返したときにだけ動いた。
元の処理が**違う値を「完成」として返す**行は、通り抜けていた（評価セット v6 の操作系の対象 124 行のうち 40 行）。
この層は、元の処理より**先に**、依頼そのものの部分（主節）を見て決める。決まらないときだけ、これまでの処理に任せる。

考え方は 3 つ。

|考え方|中身|直る例|
|---|---|---|
|1 主節を先に見る|「〜だから」「〜ので」「〜たら」「〜あいだ」「〜前に」「〜ように」の後ろだけを、操作の語を探す範囲にする。前置きの語（全開・静か・消して）を拾わない|窓を**全開**にしたら音が負けるのでボリュームを足して → 音量を 1 段上げる|
|2 語を見る順番を決め直す|空調: 曇り取り → 止める → 数字 → 最大最小 → 強さ（風量）→ 温度の向き → 暖房・冷房を入れる。メディア: 消音の解除 → 消音 → 続きから → 次 → 前 → 止める → 数字 → 最大最小 → 音量の向き → 再生|デフロスターを入れて → 曇り取り（元は「入れて」を先に見て auto）|
|3 目的地は、最後の名詞のかたまりだけ|「まで」「へ」「に向かって」の前から、助詞（を・が・は・から・ので・〜しに・〜で）の後ろだけを取る|ライトが点かないからカー用品店に向かって → カー用品店|

利用者が決めたこと（2026-10-05）: 冷房・暖房を「強く」は風量／暖房を「入れて」は heat（冷房は cool）／席の指定は見ない。

    make_resolve(A) -> resolve(tool, argmode, utterance, state, **kw)      # `p8_gate.run(make_resolve=...)` に渡す
"""
from __future__ import annotations

import re

from .p9_arguments import DEST
from .p21_arguments import PLACE_DEICTIC, ROUTE_END, make_resolve as make_p21

# ---------------------------------------------------------------- 1. 主節
# 理由・条件・時の前置きの終わり。「から」は、述語の後ろ（寒いから・切るから・心配だから）だけ。名詞の後ろ（窓から・続きから・最初から）は切らない
CLAUSE_END = re.compile(r"、|。|(?<=[いたるだすくうつむぶぬぐん])から|ので|のに|けれど|けど|(?<=[っいえしきちりみにけめれせげべてねんだ])たら|あいだ|間に|間は|前に|うちに|ように")


def split_main(u: str) -> tuple[str, str]:
    """（前置き, 主節）。前置きが無ければ（"", 発話）"""
    end = 0
    for m in CLAUSE_END.finditer(u):
        if len(u[m.end():].strip("、。 　")) >= 2:
            end = m.end()
    return u[:end], u[end:].strip("、。 　")


NUM = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?")
ZEN = str.maketrans("０１２３４５６７８９．", "0123456789.")
TARGET_VALUE = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?\s*(?:度|段階|段目|段|つ|目盛り?)?\s*(?:に|まで|へ|で)")      # 数字のすぐ後ろが「に・まで・へ・で」なら、行き先の値
MAX = re.compile(r"最大|全開|マックス|最強|最高|フルパワー|めいっぱい|目一杯|目いっぱい|フル(?:に|で)|(?:一番|いちばん)(?:強|大|高|上)")
MIN = re.compile(r"最小|最低|最弱|全閉|(?:一番|いちばん)(?:弱|小|低|下)")
NEG = re.compile(r"(?:止め|とめ|切ら|消さ|下げ|上げ|変え)ないで")


def num_of(m) -> float:
    return float(m.group(0).translate(ZEN))


# ---------------------------------------------------------------- 2. 空調
H_TARGET = ("冷房", "暖房", "エアコン", "空調", "クーラー", "ヒーター", "風", "送風", "ファン")
FAN_WORD = ("風量", "風", "送風", "ファン")
TEMP_WORD = ("温度", "設定温度")
STRONG = ("強く", "強め", "強化", "足し", "パワーを上げ", "パワーをアップ", "パワーアップ")
WEAK = ("弱く", "弱め", "控えめ", "抑え", "落とし", "落として", "ゆるめ", "ゆるく", "絞")
H_UP = ("上げ", "高く", "高め")
H_DOWN = ("下げ", "低く", "低め")
WARMER = ("暖かく", "暖かめ", "暖め", "温め", "あたたかく", "あったかく", "ぬくく")
COOLER = ("涼しく", "涼しめ", "冷やし", "冷やす", "冷たく", "すずしく")
H_OFF = ("切っ", "消し", "止め", "とめて", "オフ", "停止", "いらない")
DEFROST = re.compile(r"デフロスタ|曇り(?:取|止|ど)|くもり(?:取|止)|曇りを|霜取|結露")
DEFROST_OFF = ("オフ", "切っ", "止めて", "消し", "解除", "やめ", "いらない")
AUTO = ("オート", "自動", "おまかせ", "お任せ")
HANGING = ("を", "が", "の", "は", "に", "で", "と", "も", "へ", "って", "けど", "えっと", "あの", "その", "うーん")


def _hvac(u: str, state: dict, A):
    cur = state.get("hvac", {})
    pre, m = split_main(u)
    has = lambda words, text=None: any(w in (m if text is None else text) for w in words)
    done = lambda values, note: A.Arguments(values=values, notes=[f"P22: {note}"])
    if NEG.search(m):
        return None
    # 曇り取り（「入れて」「止めて」より先に見る。曇り止め・デフロスターを入れて）
    if DEFROST.search(m) or ("曇" in u and has(("取って", "取り", "取る", "とって"))):
        return None if has(DEFROST_OFF) and not has(("曇り止め", "くもり止め")) else done({"mode": "defrost"}, "曇り取り → mode=defrost")
    if has(H_OFF):
        return done({"mode": "off"}, "止める語（主節）→ mode=off")
    # 数字（主節にあるものだけ）
    n = NUM.search(m)
    if n:
        v = num_of(n)
        is_fan = has(FAN_WORD) and "度" not in m
        slot = "fan_level" if is_fan else "temperature" if ("度" in m or has(TEMP_WORD)) else None
        if slot is None:
            return None
        sign = +1 if has(H_UP + WARMER + STRONG) else -1 if has(H_DOWN + COOLER + WEAK) else 0
        if sign and not TARGET_VALUE.search(m) and cur.get(slot) is not None:
            v = A.clamp(slot, cur[slot] + sign * v)                               # あと 1 度下げて・2 つ上げて
        return done({slot: int(v) if slot == "fan_level" else float(v)}, f"数字（主節）→ {slot}")
    # 最大・最小（「一番小さく」は最小。「一番」だけで最大にしない）
    if MIN.search(m) or MAX.search(m):
        slot = "temperature" if has(TEMP_WORD) else "fan_level"
        lo, hi = A.EXTREMES[slot]
        v = lo if MIN.search(m) else hi
        return done({slot: int(v) if slot == "fan_level" else float(v)}, "最大・最小（主節）")
    target = has(H_TARGET) or has(H_TARGET, pre)
    # 強さ = 風量（利用者の決め）。冷房を強く・暖房を弱めて・風を強めに
    updown = has(("冷房", "暖房", "クーラー", "ヒーター")) and (has(H_UP) or has(H_DOWN))                # 暖房を上げて・冷房を下げて も、強さとして読む
    if (has(STRONG) or has(WEAK) or updown) and target and not has(TEMP_WORD) and cur.get("fan_level") is not None:
        sign = +1 if has(STRONG) or (updown and has(H_UP)) else -1
        return done({"fan_level": int(A.clamp("fan_level", cur["fan_level"] + sign * A.STEPS["fan_level"]))}, f"強さ → 風量を {sign:+d} 段")
    if has(FAN_WORD) and not has(TEMP_WORD) and (has(H_UP) or has(H_DOWN)) and cur.get("fan_level") is not None:
        sign = +1 if has(H_UP) else -1
        return done({"fan_level": int(A.clamp("fan_level", cur["fan_level"] + sign * A.STEPS["fan_level"]))}, f"風量を {sign:+d} 段")
    # 温度の向き
    if cur.get("temperature") is not None:
        sign = 0
        if has(TEMP_WORD) and (has(H_UP) or has(H_DOWN)):
            sign = +1 if has(H_UP) else -1
        elif has(WARMER) or has(COOLER):
            sign = +1 if has(WARMER) else -1
        if sign:
            return done({"temperature": float(A.clamp("temperature", cur["temperature"] + sign * A.STEPS["temperature"]))}, f"温度を {sign:+d} 段")
    # 暖房・冷房を入れる（利用者の決め: 暖房は heat。冷房は cool）
    if has(AUTO):
        return done({"mode": "auto"}, "おまかせ・自動 → mode=auto")
    if m.endswith(HANGING):
        return None                                                  # 「暖房の温度を」「冷房の風が」。言いかけは決めない
    if "暖房" in m or "ヒーター" in m:
        return done({"mode": "heat"}, "暖房を入れる → mode=heat")
    if "冷房" in m or "クーラー" in m:
        return done({"mode": "cool"}, "冷房を入れる → mode=cool")
    return None


# ---------------------------------------------------------------- 2. メディア
MUTE = ("ミュート", "消音", "無音")
UNMUTE = ("解除", "やめ", "取りやめ", "オフ", "外し", "戻")
RESUME = ("続き", "再開", "続けて")
NEXT = re.compile(r"次の曲|次へ|次に|スキップ|飛ばし|曲先|先の曲|曲送り|次を|送って")
PREV = re.compile(r"前の曲|(?:一|ひと|1|１)つ前|(?:一|1|１)曲前|曲.{0,4}戻|戻って|巻き戻")
AGAIN = re.compile(r"(?:最初|頭|はじめ)から|もう(?:一回|一度|いっかい|1回)")
STOP = ("止め", "とめて", "停止", "ポーズ", "ストップ")
V_SLOT = re.compile(r"音量|ボリューム|音(?!楽|声|源)")
V_UP = ("上げ", "大きく", "大きめ", "足し", "増や")
V_DOWN = ("下げ", "小さく", "小さめ", "絞", "控え", "落とし", "落として", "減ら", "静かに")
V_UPDOWN = re.compile(r"(?:音量|ボリューム|音)(?:を|の)?(アップ|ダウン)")          # 「ライトアップ」の「アップ」は当てない
NOT_VOLUME_UNIT = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?\s*(?:曲|分|秒|番|回|時|倍|人|件|日|年|個目|番目)")
PLAY = ("かけ", "流し", "流す", "再生", "つけ", "聴", "聞きたい", "切り替え", "替え", "お願い")


def _media(u: str, state: dict, A):
    vol = state.get("media", {}).get("volume")
    pre, m = split_main(u)
    has = lambda words, text=None: any(w in (m if text is None else text) for w in words)
    done = lambda values, note: A.Arguments(values=values, notes=[f"P22: {note}"])
    if NEG.search(m):
        return None
    if has(MUTE):
        if has(UNMUTE) and vol is not None:
            return done({"action": "volume_set", "volume": int(vol if vol > 0 else 12)}, "消音の解除 → いまの音量に戻す")
        return done({"action": "mute"}, "消音")
    if V_SLOT.search(m) and has(("消し", "消す")):
        return done({"action": "mute"}, "音を消す → 消音")
    if has(RESUME):
        return done({"action": "play"}, "続きから・再開 → 再生")
    if NEXT.search(m):
        return done({"action": "next"}, "次の曲")
    if PREV.search(m) and not V_SLOT.search(m):
        return done({"action": "previous"}, "前の曲")
    if AGAIN.search(m):
        return done({"action": "previous"}, "最初から・もう一回 → 頭出し")
    if has(STOP) or (has(("消し",)) and has(("音楽", "曲", "ラジオ", "再生"))):
        return done({"action": "pause"}, "止める語（主節）→ 一時停止")
    slot = bool(V_SLOT.search(m))
    sign = +1 if has(V_UP) else -1 if has(V_DOWN) else 0
    if not sign and V_UPDOWN.search(m):
        sign = +1 if V_UPDOWN.search(m).group(1) == "アップ" else -1
    n = NUM.search(m)
    if n and not NOT_VOLUME_UNIT.search(m) and (slot or sign) and vol is not None:
        v = num_of(n)
        if sign and not TARGET_VALUE.search(m):
            v = A.clamp("volume", vol + sign * v)                                 # 音を 4 つ上げて
        return done({"action": "volume_set", "volume": int(v)}, "数字（主節）→ 音量")
    if (MIN.search(m) or MAX.search(m)) and (slot or sign):
        lo, hi = A.EXTREMES["volume"]
        return done({"action": "volume_set", "volume": int(lo if MIN.search(m) else hi)}, "最大・最小（主節）")
    if sign and vol is not None and (slot or V_SLOT.search(pre) or has(("音楽", "曲", "ラジオ")) or not has(PLAY)):
        return done({"action": "volume_set", "volume": int(A.clamp("volume", vol + sign * A.STEPS["volume"]))}, f"音量を {sign:+d} 段")
    if has(PLAY):
        return done({"action": "play"}, "再生の語（主節）→ 再生")
    return None


# ---------------------------------------------------------------- 3. ナビ
ADVERBS = ("いったん", "一度", "とりあえず", "まずは", "まず", "ひとまず", "今から", "これから", "いまから", "途中で", "途中に", "途中", "ついでに", "急いで", "いそいで")
# 場所を名前で言っていない目的地（元の処理の指示語の表に足す）。聞き返す
VAGUE_PLACE = re.compile(r"さっき|いつも|こないだ|この前|この間|前回|昨日|先日|例の|(?:ところ|とこ|場所|あたり|辺|へん)$")
SELF_FIX = re.compile(r"、(?:いや|やっぱ|違う|ちがう|じゃなくて)")          # 言い直し。規則（S1）が止める形。ここでは決めない
# 目的地の前で切る所: 助詞（を。が・は は弱い）・から・ので・読点・名詞に付く「で」・動詞に付く「に」（買いに・届けに）。
# 動詞の「て」では切らない（「そこの空いてる枠」「さっき寄ったところ」を途中で切ると、指示語が落ちて、目的地として通ってしまう）
BOUNDARY = re.compile(r"、|。|から|ので|けど|たら|なら|まで|を|前に|うちに|あいだに|間に|ついでに|ではなくて|じゃなくて|ではなく|じゃなく"
                      r"|(?<=[一-龥ァ-ヶー々])で|(?<=[いきしちにひみりぎじびえけせてねへめれげぜべ])に|(?<=を[一-龥])に")
WEAK_BOUNDARY = re.compile(r"[がは]")          # 後ろが 2 文字以上のときだけ切る（「いろは坂」を「坂」にしない）
WAY = re.compile(r"経由|寄り道|途中|寄(?:る|れる)(?:ルート|道|よう)|寄ってから|寄りたい")
WAY_PAT = (re.compile(r"経由地?[にへ](.+?)を(?:追加|入れ|設定|セット|足|加え)"), re.compile(r"(.+?)を経由"), re.compile(r"(.+?)[にへ](?:寄|立ち寄)"), re.compile(r"(.+?)経由で"))
DEST_PAT = (re.compile(rf"(.+?)を(?:{DEST})に"),                                           # X を目的地にして
            re.compile(rf"(?:{DEST})[をは](.+?)(?:に(?:して|設定|変更|セット|変え|お願い)|で(?:お願い|頼)|へ変)"),   # 目的地を X に変更して
            re.compile(r"(.+?)(?<![るくつうすむぶぬぐ])まで(?=案内|ナビ|お願い|連れ|行|向か|頼|出発|走|進|の(?:ルート|案内|道|ナビ|行き方)|道案内|ルート|急|戻|帰|迎え|取り|届け|運|乗せ|、|$)"),   # X まで（終わるまで・着くまで は当てない）
            re.compile(r"(.+?)へ(?=案内|ナビ|お願い|連れ|行|向か|頼|出発|走|進|の(?:ルート|案内|道|ナビ|行き方)|道案内|ルート|急|戻|帰|$)"),
            re.compile(r"(.+?)に(?=向か|行っ|行き|行く|案内|ナビ|連れ|戻|帰|停め)"))
CLEAR = re.compile(rf"(?:{DEST}|案内|ルート|ナビ).{{0,6}}(?:なし|無し|クリア|削除|リセット|消し|消去|解除)")
ROUTE_MORE = re.compile(r"道順|(?:渋滞|混雑|工事|事故|通行止め).{0,4}(?:避け|回避|よけ|迂回)|迂回|(?:違う|ちがう|ほかの|他の|別の)(?:道|ルート|行き方)|(?:ルート|経路|道順|行き方)[にをで](?:して|変|切り替|選)")
ROUTE_IN = re.compile(r"[ぁ-ん](?:道|道路|ルート|経路|コース|道順)$|^(?:ルート|経路|道順|別ルート|最短|近道)$")


def chunk(text: str, A) -> str:
    """助詞の後ろだけ（最後の名詞のかたまり）を取る。「の」は残す（駅前の店）"""
    t = text.strip("、。 　")
    cut = 0
    for pat, least in ((BOUNDARY, 1), (WEAK_BOUNDARY, 2)):
        for m in pat.finditer(t):
            if m.start() > 0 and len(t[m.end():].strip("、。 　")) >= least:
                cut = max(cut, m.end())
    t = t[cut:].strip("、。 　")
    changed = True
    while changed:
        changed = False
        for a in ADVERBS:
            if t.startswith(a) and len(t) > len(a):
                t, changed = t[len(a):].lstrip("、 のを"), True
    return A.clean_destination(t)


def place_of(u: str, A, patterns=DEST_PAT) -> str | None:
    for pat in patterns:
        m = pat.search(u)
        if m:
            d = chunk(m.group(1), A)
            if d:
                return d
    return None


def _navigation(u: str, base, A):
    done = lambda values, note: A.Arguments(values=values, notes=[f"P22: {note}"])
    ask = lambda note: A.Arguments(values={"action": "set"}, missing=["destination"], notes=[f"P22: {note}"])
    if (base.complete and base.values.get("action") == "cancel") or SELF_FIX.search(u):
        return base
    if CLEAR.search(u):
        return done({"action": "cancel"}, "目的地を消す → cancel")
    if WAY.search(u):
        d = place_of(u, A, WAY_PAT)
        if d and not A.is_anaphoric_destination(d) and not VAGUE_PLACE.search(d) and not ROUTE_IN.search(d):
            return done({"action": "add_waypoint", "destination": d}, f"経由地 {d!r}")
        return base if (base.complete and base.values.get("action") == "add_waypoint") or not base.complete else ask("経由地が決まらない")
    if base.complete and base.values.get("action") == "reroute" and not (base.notes and "P21: 目的地" in base.notes[0]):
        return base
    d = place_of(u, A)
    if d is None and ROUTE_MORE.search(u) and not PLACE_DEICTIC.search(u):
        return done({"action": "reroute"}, "道の条件 → reroute")
    if d is None and base.complete and base.values.get("action") == "set":
        d = chunk(str(base.values.get("destination", "")), A) or None
    if d and not ROUTE_IN.search(d) and not ROUTE_END.search(d):
        if A.is_anaphoric_destination(d) or PLACE_DEICTIC.search(d) or VAGUE_PLACE.search(d):
            return base if not base.complete else ask("指示語の目的地")
        return done({"action": "set", "destination": d}, f"目的地 {d!r}")
    if (d and (ROUTE_IN.search(d) or ROUTE_END.search(d))) or ROUTE_MORE.search(u):
        return base if not base.complete and PLACE_DEICTIC.search(u) else done({"action": "reroute"}, "道の条件 → reroute")
    return base


def make_resolve(A):
    p21 = make_p21(A)

    def resolve(tool: str, argmode, utterance: str, state: dict, **kw):
        if tool == "control_hvac":
            return _hvac(utterance, state, A) or p21(tool, argmode, utterance, state, **kw)
        if tool == "control_media":
            return _media(utterance, state, A) or p21(tool, argmode, utterance, state, **kw)
        base = p21(tool, argmode, utterance, state, **kw)
        if tool == "set_navigation":
            if kw.get("recent_poi"):
                return base                                          # 直前の検索結果があるとき（2 回目以降）は、元の処理のまま
            return _navigation(utterance, base, A)
        return base
    return resolve
