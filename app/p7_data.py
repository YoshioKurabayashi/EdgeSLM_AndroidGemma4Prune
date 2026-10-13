# -*- coding: utf-8 -*-
"""P7 手順 1: 新しい評価セット（challenge_v2）と、学習に足す発話（train_add_v2）を作る。

設計: DoC/設計書_P7_機能にない依頼の止め方.md の 3 節。

**評価セットを先に固定する**（challenge_v1 は P5 で種類を学習に合わせてしまい、物差しとして使い切った）。

challenge_v2（200 行。学習には使わない）:
|種類|行|正解|中身|
|---|---|---|---|
|H 機能にない依頼|60|確認へ回す|30 機能 × 2 文。15 機能は学習に同じ機能がある（別の言い方）、15 機能は学習に 1 文も無い|
|K 機能の語を含む、機能にある依頼|60|各機能|「窓」「ドア」「シート」などを含むが、空調・状態・説明書などで対応できる依頼。**止めてはいけない側**|
|N ふつうの依頼|40|各機能（8 機能 × 5）|短い言い換え|
|F 雑談|20|CHAT|半分は「窓」「ライト」などの語を含む|
|G 意図が不明|20|確認へ回す|途中で切れる・自己矛盾・指示語・曖昧|
|U 利用者が足した行（任意）|—|—|`data/challenge_v2/user_rows.tsv` があれば足す（発話<TAB>正解）|

train_add_v2（168 行。933 行に足して 1,101 行）:
|種類|行|ラベル|中身|
|---|---|---|---|
|OS 機能にない依頼|120|UNKNOWN|40 単位 × 3 文。v1 の 20 機能の別の言い方＋新しい 20 機能|
|HN 機能の語を含む、機能にある依頼|48|各機能・雑談|16 単位 × 3 文。機能にない依頼と同じ語を含む対（語だけを覚えさせないため）|

    python -m app.p7_data      # data/challenge_v2/utterances.jsonl・data/train_add_v2/added.jsonl・utterances.jsonl
"""
from __future__ import annotations

import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, ROUTER_DATA
from .p5_train_data import SPLIT_CYCLE, TOOL, near

CH1 = PROJECT_ROOT / "data" / "challenge_v1" / "utterances.jsonl"
CH2 = PROJECT_ROOT / "data" / "challenge_v2"
TR1 = PROJECT_ROOT / "data" / "train_add_v1" / "utterances.jsonl"      # 933 行（714 行＋v1 で足した行）
TR2 = PROJECT_ROOT / "data" / "train_add_v2"
USER_ROWS = CH2 / "user_rows.tsv"
MAX_CHARS, NEAR = 40, 0.60
ACT = {**{k: f"TOOL_{v}" for k, v in TOOL.items()}, "chat": "CHAT", "unk": "UNKNOWN"}
USER_LABELS = {"機能なし": ("H", "unk"), "不明": ("G", "unk"), "雑談": ("F", "chat"), **{k: ("N", k) for k in TOOL}}
NOTES = {"H": "機能にない依頼", "K": "機能の語を含む、機能にある依頼", "N": "ふつうの依頼", "F": "雑談", "G": "意図が不明", "U": "利用者が足した行"}

# ================================================================ 評価セット challenge_v2
# (機能の名前, 学習に同じ機能があるか, 2 文)
H2 = [
    ("window", True, ["窓を半分だけ開けて", "後ろの窓を閉めてほしい"]),
    ("light", True, ["スモールランプをつけて", "ライトを自動にして"]),
    ("phone", True, ["お父さんに電話をかけて", "電話をスピーカーに切り替えて"]),
    ("lock", True, ["車の鍵をかけて", "全部のドアを開錠して"]),
    ("seat", True, ["シートをもう少し後ろに下げて", "座面を高くして"]),
    ("wiper", True, ["ワイパーを間欠にして", "ワイパーを一回だけ動かして"]),
    ("engine", True, ["エンジンをもう一度かけ直して", "エンジンをオフにして"]),
    ("display", True, ["ナビ画面を暗くして", "画面の明るさを最大にして"]),
    ("horn", True, ["クラクションを一瞬だけお願い", "前の車にホーンで知らせて"]),
    ("network", True, ["車のWi-Fiをオンにして", "ネットワークに接続し直して"]),
    ("seatheat", True, ["シートヒーターを入れて", "座席を温めて"]),
    ("steerheat", True, ["ハンドルヒーターをつけて", "ステアリングを温めてほしい"]),
    ("childlock", True, ["チャイルドロックをかけて", "後ろのドアのチャイルドロックを外して"]),
    ("sunroof", True, ["サンルーフを少しチルトして", "屋根を開けて"]),
    ("update", True, ["車のソフトをアップデートして", "最新のバージョンに更新しておいて"]),
    ("massage", False, ["マッサージ機能をつけて", "腰のマッサージを強めにして"]),
    ("turnsignal", False, ["右のウインカーを出して", "ウインカーを消して"]),
    ("gear", False, ["ギアをバックに入れて", "ドライブに入れて"]),
    ("timer", False, ["タイマーを10分でセットして", "ストップウォッチを開始して"]),
    ("calendar", False, ["明日の予定を登録して", "来週の会議をカレンダーに入れて"]),
    ("reservation", False, ["今夜のホテルを予約して", "レストランの席を取っておいて"]),
    ("shopping", False, ["洗剤をネットで注文して", "いつもの水を買っておいて"]),
    ("payment", False, ["駐車料金を払っておいて", "電気代を振り込んで"]),
    ("assist", False, ["車線維持をオンにして", "自動ブレーキを切って"]),
    ("speedlimit", False, ["上限速度を80キロにセットして", "スピードリミッターを解除して"]),
    ("outlet", False, ["コンセントの電源を入れて", "USBの給電を止めて"]),
    ("profile", False, ["運転者を妻に切り替えて", "ゲストモードにして"]),
    ("security", False, ["盗難防止アラームをセットして", "セキュリティを解除して"]),
    ("fourwd", False, ["四駆に切り替えて", "デフロックを入れて"]),
    ("delivery", False, ["荷物の再配達を頼んで", "宅配便の集荷を依頼して"]),
]
K2 = {
    "hvac": ["リアの窓が曇ってるから取って", "シートが熱いから冷房を強めて", "ハンドルが冷たいから暖房を上げて", "後ろの席にも風を送って",
             "運転席側だけ温度を下げて", "サンルーフから日が入って暑いから温度を下げて", "ドアを開けたら冷えたから暖房を強くして",
             "エンジンが温まったみたいだから暖房をつけて", "助手席の足元に風を出して", "窓を閉めたから冷房に切り替えて"],
    "media": ["電話が終わったから音楽を再開して", "ワイパーの音がうるさいから音量を上げて", "エンジン音で聞こえないからボリュームを大きくして",
              "スマホの曲に切り替えて", "窓を開けてるから音を小さくして", "後ろの席の子が寝たから音を止めて", "テレビじゃなくてラジオにして",
              "着信音が聞こえるように音楽を小さくして"],
    "nav": ["シートを買いに行きたいからカー用品店まで案内して", "ライトが切れたからディーラーまで連れてって", "ワイパーを買いたいからホームセンターを目的地にして",
            "エンジンの調子が悪いから整備工場へ向かって", "鍵を忘れたから家に戻るルートにして", "トランクの荷物を届けたいから実家まで案内して",
            "窓の修理に行くからガラス屋までナビして"],
    "status": ["ドアが全部閉まってるか見て", "トランクが開いたままになってないか確認して", "エンジンオイルの量は今どうなってる",
               "ライトの警告が出てないか教えて", "タイヤの空気、減ってないか見て", "後ろのドアがちゃんと閉まったか教えて",
               "エンジンの警告灯は点いてない", "バッテリーが弱ってないか、電圧を見せて", "充電はあと何パーセント残ってる"],
    "manual": ["窓が開かないときの対処を教えて", "シートの倒し方がわからない", "ワイパーの替え方を調べて", "ライトの消し忘れ警告ってどういう意味",
               "トランクの開け方を説明書で調べて", "サンルーフの使い方を教えて", "ミラーの調整はどうやるの", "エンジンのかけ方を教えてほしい",
               "電話のつなぎ方を説明書で確認したい", "シートヒーターのスイッチはどこにあるの", "ドアロックの設定方法を知りたい",
               "クルーズコントロールはどうやって使うの"],
    "poi": ["近くでワイパーを売ってる店を探して", "この辺に鍵屋はある", "シートカバーを買える店を調べて", "近くの携帯ショップを探して",
            "ライトを交換してくれる整備工場はある"],
    "minfo": ["窓を閉めたら聞こえたけど、今の曲は何て名前", "スマホから流れてるのは誰の曲"],
    "cam": ["トランクの中を映して", "ドアの外に人がいないか見せて", "ミラーに映らないところを映して", "後ろの窓の向こうをカメラで見せて",
            "サイドミラーの下のあたりを映して", "シートに座ったまま真後ろを確認したい", "ドアを開ける前に右後ろを映して"],
}
N2 = {
    "hvac": ["車内をもう少し暖めて", "冷房を弱めにして", "風をいちばん弱くして", "温度を20度に設定して", "エアコンを自動にして"],
    "media": ["何か一曲かけて", "次の曲に飛ばして", "音量を半分にして", "曲をいったん止めて", "消音をやめて音を出して"],
    "nav": ["名古屋駅まで案内して", "一番近い病院へ向かって", "目的地を自宅に設定して", "ルート案内を中止して", "有料道路を避けて案内して"],
    "status": ["燃料の残量を教えて", "航続距離はあとどのくらい", "タイヤの空気圧を表示して", "総走行距離を見せて", "バッテリー残量は何パーセント"],
    "manual": ["給油ランプが点いたらあと何キロ走れるの", "タイヤのローテーションは何キロごと", "冷却水の補充方法を教えて",
               "ヒューズボックスはどこにあるの", "この車の最小回転半径を教えて"],
    "poi": ["近くのラーメン屋を探して", "ガソリンを入れられる場所を調べて", "周辺のコインパーキングを調べて", "近くに病院はある", "一番近いコンビニはどこ"],
    "minfo": ["この曲のタイトルを教えて", "今のアーティストは誰", "再生中のアルバム名は", "この曲は何分ある", "今流れてる放送局はどこ"],
    "cam": ["後ろのカメラを表示して", "前方の映像を見せて", "左側をカメラで映して", "車の周囲を確認させて", "右後ろの様子を見たい"],
}
F2 = ["窓を開けて走ると気持ちいいよね", "最近のライトは明るいよね", "シートの座り心地がいい車だね", "電話しながらの運転は危ないよね",
      "ワイパーの音って眠くなるよね", "エンジン音が静かな車だね", "鍵をよくなくすんだよね", "サンルーフのある車に憧れてたんだ",
      "トランクが広いと旅行に便利だね", "クラクションを鳴らす人って苦手だな",
      "今日は仕事がうまくいったよ", "週末は釣りに行こうかな", "お昼に食べたカレーがおいしかった", "最近よく眠れてるんだ",
      "子どもが運動会で一等だったよ", "そろそろ紅葉の季節だね", "映画でも観に行きたいな", "コーヒーを飲むと落ち着くね",
      "明日は早起きしないと", "今年の夏は暑かったね"]
G2 = ["冷房を、あの", "音楽の", "次の交差点で", "後ろを、あの", "ガソリンが、えっと",
      "流して、いややっぱり止めて", "向かって、あ、待って", "見せて、あ、いいや",
      "それをもう少し", "あれに戻して",
      "ちょうどいい感じで頼む", "うまく調整しといて", "なんでもいいからやって",
      "さっきのは取りやめ", "今言ったのは忘れて",
      "前に行ったあそこまで", "いつものところへ",
      "えーと、なんだったかな", "あー、やっぱなんでもない", "うーん、ちょっと考える"]

# ================================================================ 学習に足す train_add_v2
# 機能にない依頼: (単位の名前, 3 文)。評価セット H の「学習に無い」15 機能と、challenge_v1 の H の「学習に無い」15 機能は入れない
OS2 = [
    ("window2", ["窓を少しだけ開けておいて", "全部の窓を閉めて", "助手席側の窓を上げて"]),
    ("horn2", ["クラクションを鳴らしておいて", "ホーンを一回お願い", "後ろの車に警笛で合図して"]),
    ("light2", ["ライトを点灯して", "車内のランプを消して", "フォグを消しておいて"]),
    ("display2", ["モニターの明るさを落として", "メーターを明るくして", "画面を昼のモードに戻して"]),
    ("phone2", ["自宅に電話して", "この電話を保留にして", "さっきの番号にかけ直して"]),
    ("network2", ["モバイル通信をオフにして", "Wi-Fiを切って", "テザリングでつなぎ直して"]),
    ("message2", ["妻にメッセージを書いて", "未読のLINEを読み上げて", "「遅れます」と返事を送って"]),
    ("update2", ["地図を更新しておいて", "システムのアップデートを始めて", "車を再起動して"]),
    ("lock2", ["ドアの鍵を開けておいて", "全ドアを施錠して", "運転席だけロックを外して"]),
    ("home2", ["家のエアコンをつけておいて", "リビングの照明をつけて", "家の玄関の鍵を閉めて"]),
    ("trunk2", ["トランクを少しだけ開けて", "リアゲートを閉めて", "荷室のドアをロックして"]),
    ("memo2", ["買い物リストにメモして", "今から言うことを書き留めて", "ボイスメモを再生して"]),
    ("seat2", ["背もたれをもっと倒して", "運転席を少し上げて", "助手席のシートを前に動かして"]),
    ("sns2", ["今の写真をインスタに上げて", "ツイートして", "友だちに今いる場所を投稿して"]),
    ("wiper2", ["ワイパーをオンにして", "ワイパーをゆっくりにして", "後ろのワイパーを止めて"]),
    ("steering2", ["ハンドルを少し下げて", "ステアリングの高さを戻して", "アクセルペダルを手前にして"]),
    ("engine2", ["エンジンを始動して", "エンジンを停止させて", "アイドリングストップを有効にして"]),
    ("settings2", ["日付を合わせて", "単位をマイルに変えて", "表示を日本語に戻して"]),
    ("sunroof2", ["サンルーフを全開にして", "ルーフのシェードを開けて", "天井の窓を閉めて"]),
    ("app2", ["天気アプリを開いて", "ゲームを終了して", "新しいアプリを入れて"]),
    ("seatheat", ["シートヒーターを強にして", "助手席のシートヒーターを切って", "座席のヒーターを弱めて"]),
    ("steerheat", ["ハンドルのヒーターを入れて", "ステアリングヒーターを切って", "ハンドルを温かくして"]),
    ("childlock", ["チャイルドロックを有効にして", "チャイルドロックを解除して", "後席のドアを内側から開かないようにして"]),
    ("seatvent", ["シートの送風をつけて", "座席のベンチレーションを強くして", "シートクーラーを止めて"]),
    ("washer", ["ウォッシャー液を出して", "フロントガラスを洗って", "後ろのガラスにウォッシャーをかけて"]),
    ("tireinflate", ["タイヤに空気を入れて", "空気圧を調整しておいて", "タイヤの空気を少し抜いて"]),
    ("evcharge", ["充電を開始して", "充電を80パーセントで止めて", "充電の予約を夜11時にして"]),
    ("tripmeter", ["トリップメーターをリセットして", "燃費の記録を消して", "走行距離の表示をゼロに戻して"]),
    ("hud", ["ヘッドアップディスプレイを消して", "フロントガラスの表示を上にずらして", "HUDを明るくして"]),
    ("contacts", ["連絡先に山田さんを登録して", "電話帳を開いて", "この番号を保存して"]),
    ("voiceassist", ["声を男性に変えて", "話す速さをゆっくりにして", "呼びかけの言葉を変えて"]),
    ("locshare", ["今いる場所を家族に共有して", "到着予定を母に送って", "位置情報の共有を止めて"]),
    ("roadservice", ["ロードサービスを呼んで", "レッカーを手配して", "保険会社に連絡して"]),
    ("glovebox", ["グローブボックスを開けて", "小物入れのふたを閉めて", "コンソールボックスをロックして"]),
    ("key", ["スマートキーを探して", "キーの登録を追加して", "リモコンでエンジンをかけられるようにして"]),
    ("ambient", ["アンビエントライトを青にして", "足元の照明を消して", "車内のイルミネーションを暗くして"]),
    ("rearmonitor", ["後ろの席のモニターをつけて", "後席の画面を消して", "リアのモニターに映画を映して"]),
    ("tpmsreset", ["空気圧の警告をリセットして", "タイヤの警告表示を消して", "空気圧センサーを初期化して"]),
    ("maintreset", ["点検のお知らせをリセットして", "オイル交換の表示を消して", "メンテナンスの通知を止めて"]),
    ("fragrance", ["車内の香りを変えて", "芳香剤の機能をオンにして", "香りを強くして"]),
]
# 機能の語を含む、機能にある依頼: (単位の名前, クラス, 3 文)
HN2 = [
    ("hvac_1", "hvac", ["ガラスが曇って見えないから乾かして", "シートが冷たいから暖房を強めて", "ハンドルが熱いから冷房を入れて"]),
    ("hvac_2", "hvac", ["ドアを閉めたから冷房を入れ直して", "後ろの人が暑いらしいから後席の温度を下げて", "エンジンを止める前にエアコンを切って"]),
    ("media_1", "media", ["通話中は音を消しておいて", "窓を全開にしたから音量を上げて", "携帯の音楽に切り替えて"]),
    ("media_2", "media", ["ドアを閉める音で起きちゃうから曲を止めて", "トランクを開けるあいだ音楽を一時停止して", "鍵をかける前に音を消して"]),
    ("nav_1", "nav", ["鍵を取りに家へ向かって", "ライトを買いにカー用品店まで案内して", "窓の修理で板金屋に行きたい、案内して"]),
    ("nav_2", "nav", ["エンジンを見てもらうからディーラーを目的地にして", "シートを取りに行くから倉庫までナビして", "電話で言われた通り、市役所へ向かって"]),
    ("status_1", "status", ["ドアはちゃんと閉まった", "トランク、半開きになってない", "エンジンオイルは足りてる"]),
    ("status_2", "status", ["ライト関係の警告は出てる", "鍵をかける前に燃料の残りを教えて", "タイヤの空気、抜けてないか教えて"]),
    ("manual_1", "manual", ["窓の挟み込み防止ってどういう機能", "シートの位置を記憶させる方法を教えて", "ワイパーゴムの交換時期はいつ"]),
    ("manual_2", "manual", ["ライトのバルブはどの型を使うの", "トランクが開かないときはどうすればいい", "ドアロックが勝手にかかるのはなぜ"]),
    ("manual_3", "manual", ["サンルーフの手入れの仕方を知りたい", "ハンドルの位置の変え方を教えて", "電話の音が出ないときの直し方は"]),
    ("poi_1", "poi", ["近くで鍵を作ってくれる店を探して", "ワイパーを交換してくれる店はある", "窓ガラスを直せる修理工場を調べて"]),
    ("cam_1", "cam", ["トランクを閉める前に後ろを映して", "ドアの横に自転車がいないか見せて", "窓の外が見えないから左を映して"]),
    ("cam_2", "cam", ["ミラーで見えない後ろの下を映して", "シートの後ろじゃなくて車の後ろを見せて", "ライトの当たってる前のほうを映して"]),
    ("minfo_1", "minfo", ["携帯から流れてる曲の名前は", "スマホで再生してるアルバムは何", "エンジンをかけたら流れたこの曲、誰の"]),
    ("chat_1", "chat", ["窓を開けると花粉が入るよね", "ライトが明るい車は夜も安心だね", "シートが柔らかいと眠くなるな"]),
]
TYPE_NOTE = {"OS": "機能にない依頼", "HN": "機能の語を含む、機能にある依頼"}


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_challenge() -> list[dict]:
    rows = []

    def add(cat: str, k: int, utt: str, act: str, extra: dict | None = None) -> None:
        rows.append({"id": f"c2_{cat}_{k:03d}", "category": cat, "category_note": NOTES[cat], "utterance": utt, "history": [],
                     "state": {"motion": "stopped", "permission": "allowed"}, "action": ACT[act], "argmode": None,
                     "trainable": False, "purpose": "challenge_eval", **(extra or {})})

    k = 0
    for name, seen, utts in H2:
        for u in utts:
            add("H", k, u, "unk", {"os_function": name, "os_function_in_train": seen})
            k += 1
    k = 0
    for cls, utts in K2.items():
        for u in utts:
            add("K", k, u, cls)
            k += 1
    k = 0
    for cls, utts in N2.items():
        for u in utts:
            add("N", k, u, cls)
            k += 1
    for k, u in enumerate(F2):
        add("F", k, u, "chat")
    for k, u in enumerate(G2):
        add("G", k, u, "unk")
    if USER_ROWS.exists():           # 任意: 利用者が足した行（発話<TAB>正解）。正解は 機能なし・不明・雑談・hvac・media・nav・status・manual・poi・minfo・cam
        k = 0
        for line in USER_ROWS.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("\t")]
            if len(parts) < 2 or parts[1] not in USER_LABELS:
                raise SystemExit(f"{USER_ROWS.name}: 読めない行「{line}」（発話<TAB>正解。正解は {'・'.join(USER_LABELS)}）")
            base, act = USER_LABELS[parts[1]]
            add("U", k, parts[0], act, {"user_kind": base})
            k += 1
    return rows


def build_train() -> list[dict]:
    tools = {t["name"]: t for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    rows = []

    def add(kind: str, unit: str, split: str, utts: list[str], cls: str | None, extra: dict | None = None) -> None:
        if len(utts) != 3:
            raise SystemExit(f"{unit}: 3 文で 1 単位（{len(utts)} 文）")
        for k, u in enumerate(utts):
            if cls is None:
                lab = {"route": "UNKNOWN", "category": None, "effect": None, "tool": None, "action": "UNKNOWN"}
            elif cls == "chat":
                lab = {"route": "CHAT", "category": None, "effect": None, "tool": None, "action": "CHAT"}
            else:
                t = tools[TOOL[cls]]
                lab = {"route": "TOOLCALL", "category": t["category"], "effect": t["effect"], "tool": t["name"], "action": f"TOOL_{t['name']}"}
            rows.append({"id": f"{unit}:{k}", "group": unit, "split_unit": unit, "split": split, "utterance": u, "history": [],
                         "state": {"motion": "stopped", "permission": "allowed"}, "last_tool": None, **lab, "argmode": None,
                         "difficulty": "hard", "confusable": None, "note": TYPE_NOTE[kind],
                         "trainable": cls is not None, "purpose": "classification" if cls is not None else "rejection_eval",
                         "added": "train_add_v2", "add_type": kind, **(extra or {})})

    for ui, (name, utts) in enumerate(OS2):
        add("OS", f"add2_os_{name}", SPLIT_CYCLE[ui % 5], utts, None, {"os_function": name})
    for ui, (name, cls, utts) in enumerate(HN2):
        add("HN", f"add2_hn_{name}", SPLIT_CYCLE[(ui + 1) % 5], utts, cls)
    return rows


def main() -> None:
    ch2, tr2 = build_challenge(), build_train()
    base = read_jsonl(TR1)                                   # 933 行
    ch1 = read_jsonl(CH1)
    train_all = {r["utterance"] for r in base} | {r["utterance"] for r in tr2}
    # 確かめ 1: 同じ文が無い（セットの中・学習と評価のあいだ）
    problems = []
    for name, rows in (("challenge_v2", ch2), ("train_add_v2", tr2)):
        c = collections.Counter(r["utterance"] for r in rows)
        problems += [f"{name} の中で重なる: {u}" for u, k in c.items() if k > 1]
        problems += [f"{name}: {MAX_CHARS} 文字超 {r['utterance']}" for r in rows if len(r["utterance"]) > MAX_CHARS]
    problems += [f"評価セット v2 が学習（933 行＋v2）と同じ: {r['utterance']}" for r in ch2 if r["utterance"] in train_all]
    old = {r["utterance"] for r in base} | {r["utterance"] for r in ch1}
    problems += [f"足す行が 933 行・評価セット v1 と同じ: {r['utterance']}" for r in tr2 if r["utterance"] in old]
    problems += [f"評価セット v2 が v1 と同じ: {r['utterance']}" for r in ch2 if r["utterance"] in {x['utterance'] for x in ch1}]
    # 確かめ 2: 評価セット（v1・v2）と学習の文が似すぎていない
    tr_new = [r["utterance"] for r in tr2]
    tr_all = sorted(train_all)
    pairs = sorted(((near(a["utterance"], b), a["utterance"], b) for a in ch2 for b in tr_all), reverse=True)
    pairs1 = sorted(((near(a["utterance"], b), a["utterance"], b) for a in ch1 for b in tr_new), reverse=True)
    print("評価セット v2 ↔ 学習（933 行＋v2）で一番似ている組:")
    for s, a, b in pairs[:10]:
        print(f"  {s:.2f}「{a}」↔「{b}」")
    problems += [f"似すぎ（{s:.2f}）: 評価 v2「{a}」↔ 学習「{b}」" for s, a, b in pairs if s >= NEAR]
    problems += [f"似すぎ（{s:.2f}）: 評価 v1「{a}」↔ 足す行「{b}」" for s, a, b in pairs1 if s >= NEAR]
    # 確かめ 3: 評価セット H の「学習に無い」機能の名前が、学習の単位に無い
    train_fn = {r.get("os_function", "").rstrip("2") for r in tr2} | {r["split_unit"].replace("add_os_", "") for r in base if r.get("add_type") == "OS"}
    for name, seen, _ in H2:
        if seen != (name in train_fn):
            problems.append(f"H の {name}: 「学習に同じ機能がある」の印（{seen}）と、学習の単位（{name in train_fn}）が合わない")
    by = collections.Counter((r["split"], r["action"]) for r in tr2)
    if problems:
        raise SystemExit("止める:\n  " + "\n  ".join(problems))

    CH2.mkdir(parents=True, exist_ok=True)
    TR2.mkdir(parents=True, exist_ok=True)
    dump = lambda rs: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    (CH2 / "utterances.jsonl").write_text(dump(ch2), encoding="utf-8")
    (TR2 / "added.jsonl").write_text(dump(tr2), encoding="utf-8")
    (TR2 / "utterances.jsonl").write_text(dump(base + tr2), encoding="utf-8")       # 933 行はそのまま・同じ並びで先頭に置く
    c = collections.Counter(r["category"] for r in ch2)
    print(f"評価セット v2: {len(ch2)} 行（" + "・".join(f"{k} {NOTES[k]} {c[k]}" for k in NOTES if c[k]) + "）")
    hs = collections.Counter(r["os_function_in_train"] for r in ch2 if r["category"] == "H")
    print(f"  機能にない依頼: 学習に同じ機能がある {hs[True]} 行・無い {hs[False]} 行" + ("" if not USER_ROWS.exists() else f" / 利用者が足した行 {c['U']}"))
    t = collections.Counter(r["add_type"] for r in tr2)
    print(f"学習に足す行 v2: {len(tr2)} 行（" + "・".join(f"{TYPE_NOTE[k]} {t[k]}" for k in TYPE_NOTE) + "）/ split: "
          + "・".join(f"{k} {v}" for k, v in collections.Counter(r["split"] for r in tr2).most_common()))
    allr = base + tr2
    os_n = sum(1 for r in allr if r.get("add_type") == "OS")
    print(f"合わせた学習データ: {len(allr)} 行（933 行＋{len(tr2)} 行）= 行き先のある行 {sum(r['trainable'] for r in allr)}・"
          f"UNKNOWN {sum(not r['trainable'] for r in allr)}（うち機能にない依頼 {os_n}）")
    print(f"保存しました: {CH2 / 'utterances.jsonl'}・{TR2}（added.jsonl・utterances.jsonl）")


if __name__ == "__main__":
    main()
