# -*- coding: utf-8 -*-
"""P4 手順 1: データの外の発話での評価セット（challenge_v1）を作る。学習には使わない。

設計: DoC/設計書_P4_データの外の発話での評価.md の 3 節。

EdgeSLM_MultiHead_Multi の 714 行（synthetic_v2）は発話が短く（平均 8.5 文字）、言い回しも学習側と評価側で似ている。
ここでは**同じ行き先・同じラベルの決め方**で、言い方だけを変えた発話と、誤実行が起きやすい発話を用意する。
ラベルの決め方は EdgeSLM_MultiHead_Multi の `docs/TOOLS.md`・`data/synthetic_v2` に合わせた（例: 「暑いな」は空調・下げる、
「今エアコン何度に設定されてる」は車両情報、「近くの〜は」は POI 検索、「〜に行って」はナビ）。

|種類|中身|正解|
|---|---|---|
|A|言い換え|各機能|
|B|長い文・前置きつき|各機能（雑談 2 行）|
|C|言いよどみ・口語|各機能|
|D|音声認識の誤り風（同音の漢字・助詞抜け）|各機能|
|E|敬語・方言・カタカナ語|各機能|
|F|機能の語を含む雑談|CHAT|
|G|意図が不明（自己矛盾・対象なし・途中で切れる・曖昧・言いよどみ）|UNKNOWN（確認へ回す）|
|H|機能にない依頼|UNKNOWN（確認へ回す）|
|I|読むだけか操作かの取り違え|各機能|
|M|複数の依頼（参考。合否に数えない）|MULTI|
|V|走行中（A・F・G の一部を、状態だけ走行中にしたもの）|元の行と同じ|

    python -m app.p4_challenge_data        # data/challenge_v1/utterances.jsonl を書く（714 行との重なりも確かめる）
"""
from __future__ import annotations

import collections
import json

from .common import PROJECT_ROOT, ROUTER_DATA

OUT = PROJECT_ROOT / "data" / "challenge_v1" / "utterances.jsonl"
T = {"hvac": "TOOL_control_hvac", "media": "TOOL_control_media", "nav": "TOOL_set_navigation",
     "status": "TOOL_get_vehicle_status", "manual": "TOOL_search_vehicle_manual", "poi": "TOOL_search_poi",
     "minfo": "TOOL_get_media_info", "cam": "TOOL_capture_camera", "chat": "CHAT", "unk": "UNKNOWN", "multi": "MULTI"}

# (発話, 行き先, ArgMode（空調・メディアで言い切れるものだけ。それ以外は None）)
A = [
    ("車内を涼しくして", "hvac", "DECREASE"), ("冷房を強くして", "hvac", "DECREASE"), ("暖房を入れて", "hvac", None),
    ("もうちょっと暖かくしてほしい", "hvac", "INCREASE"), ("設定温度を26度に変えて", "hvac", "ABSOLUTE"),
    ("風を強めにして", "hvac", "INCREASE"), ("フロントガラスの曇りを取って", "hvac", None), ("空調を切って", "hvac", "TOGGLE"),
    ("何か音楽ほしいな", "media", None), ("BGMをお願い", "media", None), ("この曲は飛ばして", "media", None),
    ("ひとつ前の曲をもう一回", "media", None), ("ボリュームを上げて", "media", "INCREASE"),
    ("音をもう少し小さく", "media", "DECREASE"), ("音を出さないで", "media", "TOGGLE"), ("FMに切り替えて", "media", None),
    ("新宿駅に連れてって", "nav", None), ("横浜までの道を教えて", "nav", None), ("目的地を大阪城にして", "nav", None),
    ("実家に向かいたい", "nav", None), ("最寄りの駅まで案内お願い", "nav", None),
    ("ここから成田空港へのルートを引いて", "nav", None), ("職場へナビを開始して", "nav", None), ("京都駅を行き先に設定", "nav", None),
    ("タイヤの空気、今どうなってる", "status", None), ("燃料はまだ残ってる", "status", None), ("ガソリンの残りを見せて", "status", None),
    ("あとどのくらいの距離走れそう", "status", None), ("累計で何キロ走ったっけ", "status", None),
    ("ドアが開いてないか確認して", "status", None), ("バッテリーはあと何パーセント", "status", None),
    ("警告ランプ点いてない", "status", None),
    ("タイヤの空気圧の基準値を知りたい", "manual", None), ("このランプが点いたらどうすればいいの", "manual", None),
    ("エンジンオイルは何番を使うの", "manual", None), ("定期点検って何キロごと", "manual", None),
    ("ガソリンタンクの容量を教えて", "manual", None), ("チャイルドロックのかけ方", "manual", None),
    ("給油口の開け方がわからない", "manual", None),
    ("この近くにガソスタある", "poi", None), ("近所でご飯食べられるところ", "poi", None), ("近くのトイレを探して", "poi", None),
    ("周辺の駐車場を調べて", "poi", None), ("一番近い薬局はどこ", "poi", None), ("この辺にカフェない", "poi", None),
    ("近くのATMを検索", "poi", None),
    ("いま流れてるの誰の曲", "minfo", None), ("この歌のタイトルは", "minfo", None), ("これ何ていうアルバム", "minfo", None),
    ("今の曲、何年の曲", "minfo", None), ("再生してる曲の情報を出して", "minfo", None), ("歌ってる人の名前を教えて", "minfo", None),
    ("この曲あと何分ある", "minfo", None),
    ("バックモニターを表示して", "cam", None), ("後ろの様子を映して", "cam", None), ("車の左側を映して", "cam", None),
    ("周りに人がいないか見せて", "cam", None), ("前のカメラの映像を見たい", "cam", None), ("後部座席の様子を見せて", "cam", None),
    ("駐車したいから周囲を映して", "cam", None),
]
B = [
    ("さっきから少し暑い気がするんだけど、エアコンの温度を2度くらい下げてもらえるかな", "hvac", "DECREASE"),
    ("後ろの席の子どもが寒いって言ってるから、暖房をもう少し強くしてあげて", "hvac", "INCREASE"),
    ("雨で窓ガラスが曇って前が見えにくいから、曇りを取ってほしい", "hvac", None),
    ("外がだいぶ涼しくなってきたから、エアコンはもう切ってしまっていいよ", "hvac", "TOGGLE"),
    ("長いドライブで眠くなってきたから、何か元気の出る音楽をかけてくれる", "media", None),
    ("電話がかかってきそうだから、音楽の音量をかなり小さくしておいて", "media", "DECREASE"),
    ("この曲はもう何回も聴いて飽きたから、次の曲に進めてほしい", "media", None),
    ("子どもが寝たから、流れている音楽をいったん止めておいて", "media", None),
    ("このあと打ち合わせがあるから、会社までの一番早い道で案内して", "nav", None),
    ("今日は疲れたからもう帰りたい、自宅までのルートを出してくれる", "nav", None),
    ("友だちを迎えに行くことになったので、東京駅を目的地にセットして", "nav", None),
    ("予定が変わったから、目的地を羽田空港に変更してほしい", "nav", None),
    ("次のサービスエリアまで持つか心配だから、ガソリンの残りを教えて", "status", None),
    ("高速に乗る前に確認したいんだけど、タイヤの空気圧は今どうなってる", "status", None),
    ("最近あまり乗っていなかったから、バッテリーの残量を見ておきたい", "status", None),
    ("さっきドアを閉めた音が変だったから、ちゃんと閉まっているか見て", "status", None),
    ("見たことのないオレンジ色のランプが点いたんだけど、これは何の警告なの", "manual", None),
    ("初めて自分で給油するんだけど、給油口はどうやって開ければいいの", "manual", None),
    ("冬に備えてタイヤを替えたいから、この車に合うタイヤサイズを教えて", "manual", None),
    ("子どもを後ろに乗せるから、チャイルドロックの設定方法を知りたい", "manual", None),
    ("お腹が空いてきたから、この近くで食事ができるお店を探してほしい", "poi", None),
    ("そろそろ充電が必要だから、近くにある急速充電器を調べてくれる", "poi", None),
    ("急にトイレに行きたくなったから、近くのコンビニを探して", "poi", None),
    ("目的地に着いたら車を停めたいので、周辺の駐車場を調べておいて", "poi", None),
    ("さっきから流れているこの曲がすごく気になるから、曲名を教えて", "minfo", None),
    ("この歌声に聞き覚えがあるんだけど、歌っているのは誰なの", "minfo", None),
    ("狭い駐車場でバックするから、後ろのカメラの映像を出してほしい", "cam", None),
    ("左に寄せて停めたいから、左側の様子をカメラで見せてくれる", "cam", None),
    ("今日は朝から会議が続いて疲れたよ、帰ったらゆっくり休みたいな", "chat", None),
    ("最近の車は機能が多すぎて、全部を使いこなすのは難しいよね", "chat", None),
]
C = [
    ("えーっと、あの、ナビ、東京駅でお願い", "nav", None), ("あ、そうだ、ガソリンってあとどんくらい", "status", None),
    ("んーと、エアコン、もうちょい下げてくんない", "hvac", "DECREASE"), ("あのさ、音楽かけてよ", "media", None),
    ("えっと、近くにコンビニないかな", "poi", None), ("ねえ、今の曲なんだっけ", "minfo", None),
    ("あ、後ろ、後ろ見して", "cam", None), ("えー、このランプ何だっけ、どういう意味", "manual", None),
    ("ちょ、音でかい、下げて", "media", "DECREASE"), ("あー寒い寒い、暖房つけて", "hvac", None),
    ("んー、家、家に帰るわ", "nav", None), ("あれ、ドア閉まってるっけ", "status", None),
    ("えーと、次、次の曲にして", "media", None), ("ねえねえ、この辺に駐車場ある", "poi", None),
    ("あ、ちょっと、右側見せて", "cam", None), ("えっとね、オイル交換っていつやればいいの", "manual", None),
    ("あのー、25度にしてもらえる", "hvac", "ABSOLUTE"), ("うーんと、誰が歌ってんのこれ", "minfo", None),
    ("あ、やば、ガソスタ探して", "poi", None), ("えー、空港、羽田まで行って", "nav", None),
    ("ねえ、あと何キロ走れんの", "status", None), ("あ、そうそう、ラジオにして", "media", None),
    ("んー、風、弱くしてくれる", "hvac", "DECREASE"), ("えっと、車の周り、映してみて", "cam", None),
    ("あのさ、タイヤの空気圧って普通いくつなの", "manual", None), ("ちょっと、エアコン消してくれる", "hvac", "TOGGLE"),
    ("あ、音楽止めて止めて", "media", None), ("えーっと、会社までナビして", "nav", None),
    ("ねえ、バッテリーってまだ大丈夫", "status", None), ("あー、この曲のアルバムって何", "minfo", None),
]
D = [
    ("エアコンを22度二して", "hvac", "ABSOLUTE"), ("東京駅まで暗内して", "nav", None), ("音楽書けて", "media", None),
    ("次の局", "media", None), ("温度を挙げて", "hvac", "INCREASE"), ("音量を下げ手", "media", "DECREASE"),
    ("エアコン 22度", "hvac", "ABSOLUTE"), ("ナビ 東京駅", "nav", None), ("ガソリン あと どれくらい", "status", None),
    ("近く コンビニ", "poi", None), ("後ろ 見せて", "cam", None), ("家二帰る", "nav", None), ("燃料残量和", "status", None),
    ("この曲何手いうの", "minfo", None), ("空気厚の適正値は", "manual", None), ("警告等の意味を教えて", "manual", None),
    ("自宅まで納日して", "nav", None), ("冷房を強くして下差い", "hvac", "DECREASE"), ("駐車上を探して", "poi", None),
    ("音楽を再生死て", "media", None), ("バックカメラを写して", "cam", None), ("車の周りを各認して", "cam", None),
    ("走行距離を押しえて", "status", None), ("充電スポット或る", "poi", None), ("誰が謡ってるの", "minfo", None),
    ("オイル交換の目安派", "manual", None), ("風量を再大にして", "hvac", "MAX_MIN"), ("ミュート仕手", "media", "TOGGLE"),
    ("空港まで尾根がい", "nav", None), ("タイヤの空気圧は今いくつ出すか", "status", None),
]
E = [
    ("温度を下げていただけますか", "hvac", "DECREASE"), ("東京駅までの案内をお願いできますでしょうか", "nav", None),
    ("音楽を再生していただけますか", "media", None), ("燃料の残量を教えていただけますか", "status", None),
    ("近くのガソリンスタンドを探していただけますか", "poi", None), ("後方の映像を表示してください", "cam", None),
    ("現在再生中の曲名を教えてください", "minfo", None), ("推奨の空気圧をお教えいただけますか", "manual", None),
    ("恐れ入りますが、音量を下げてください", "media", "DECREASE"), ("お手数ですが、自宅までナビをお願いします", "nav", None),
    ("クーラーつけてや", "hvac", None), ("音楽かけてくれへん", "media", None), ("家まで案内してほしいねん", "nav", None),
    ("ガソリンあとなんぼ残っとる", "status", None), ("この辺にコンビニあるんかいな", "poi", None),
    ("うしろ見せてくれんね", "cam", None), ("この曲なんちゅう曲や", "minfo", None), ("ちょっと寒いけん暖房つけて", "hvac", None),
    ("音ば小さくして", "media", "DECREASE"), ("空気圧ってなんぼがええの", "manual", None),
    ("ACオンにして", "hvac", None), ("ミュージックをプレイして", "media", None), ("ネクストソング", "media", None),
    ("ボリュームアップ", "media", "INCREASE"), ("ナビをスタートして、目的地は東京駅", "nav", None), ("リアカメラをオン", "cam", None),
    ("フューエルの残りは", "status", None), ("パーキングをサーチして", "poi", None), ("テンプを22度にセット", "hvac", "ABSOLUTE"),
    ("マニュアルでタイヤプレッシャーを調べて", "manual", None),
]
F = [
    "エアコンって電気食うよね", "ナビがあると知らない道でも安心だね", "この歌、なつかしいな", "昔の車はバックカメラなんてなかったな",
    "最近ガソリン高いよね", "音楽があるとドライブが楽しいね", "カーナビがない時代は地図を見てたんだよ", "冬の朝はエアコンが効くまで寒いよね",
    "タイヤ交換って面倒だよね", "この車のカメラは画質がきれいだね", "充電スタンドも増えてきたね", "コンビニのコーヒーって意外とおいしいよね",
    "ラジオって最近聴かなくなったな", "昔はカセットテープで音楽を聴いてたよ", "エアコンの風って苦手なんだよね", "渋滞のときは音楽に救われるよ",
    "ナビの声って誰の声なんだろうね", "燃費がいい車っていいよね", "車検ってお金かかるよね", "駐車が苦手だからカメラがあって助かってる",
    "東京駅は人が多くて疲れるよね", "空港って行くだけでわくわくするね", "温度の感じ方って人それぞれだよね", "音量の好みって人によって違うよね",
    "オイル交換をさぼると良くないらしいね", "マニュアルって分厚くて読む気がしないよ", "最近の曲はよくわからないな", "ジャズって大人の音楽って感じだね",
    "警告灯が点くとどきっとするよね", "ガソリンスタンドの店員さんは大変そうだね", "暖房が効いてくると眠くなるんだよね", "バックで駐車するのは今でも緊張するよ",
    "次の休みは温泉に行きたいな", "明日は雨らしいね", "子どものころ車酔いがひどかったんだ", "高速道路の景色って単調だよね",
    "この車に乗ってもう三年になるよ", "運転って肩がこるよね", "眠気覚ましにはガムがいいらしいよ", "いつもありがとう、助かってるよ",
]
G = [
    # 自己矛盾
    "つけて、いややっぱ消して", "行って、いや待って", "上げて、やっぱり下げて、いやそのままで", "かけて、あ、やっぱいい", "案内して、いややめとく",
    # 対象なし（履歴なし）
    "もっと上げて", "それ下げて", "さっきのところに行って", "同じのをもう一回", "逆にして", "あっちに変えて", "それじゃなくて、あっち",
    "いつもの設定にして", "前の状態に戻して", "もうちょっと強く",
    # 途中で切れる
    "ナビを", "音量を", "近くの", "温度を、えっと", "あの、後ろの", "目的地は", "えっと、この曲の",
    # 曖昧
    "ちょうどよくして", "ほどほどにやって", "なんか変えて", "うまくやっといて", "どうにかならない", "任せるよ", "とりあえずお願い", "どっちでもいいよ",
    # 言いよどみ・取り消し
    "えーっとね", "うーん、どうしようかな", "あ、なんでもない", "いや、別に", "あ、ちょっとタイム", "やっぱりやめた", "今のは取り消し",
    "んー、まあいいか", "あれ、なんだっけ", "そうじゃなくて",
]
H = [
    "窓を開けて", "シートを倒して", "ワイパー動かして", "母さんに電話して", "ライトをつけて", "ドアのロックを開けて", "トランクを開けて",
    "サンルーフを開けて", "ハザードをつけて", "クルーズコントロールをセットして", "メッセージを送って", "ミラーをたたんで", "エンジンを切って",
    "駐車ブレーキをかけて", "自動で駐車して", "車高を上げて", "給油口を開けて", "ボンネットを開けて", "ルームランプを消して",
    "ヘッドライトをハイビームにして", "スマホを充電して", "料金所の支払いをして", "走行モードをスポーツにして", "後ろの席の窓をロックして",
    "アラームを7時にセットして", "メールを読み上げて", "電話を切って", "リマインダーを設定して", "ピザを注文して", "タクシーを呼んで",
]
I = [
    ("いま何度に設定してある", "status", None), ("設定温度を上げて", "hvac", "INCREASE"),
    ("今の風量はいくつ", "status", None), ("風量を一段上げて", "hvac", "INCREASE"),
    ("この曲を止めて", "media", None), ("この曲は誰の", "minfo", None), ("この曲をもう一度かけて", "media", None),
    ("この曲いつ終わる", "minfo", None), ("このアーティストの曲をかけて", "media", None), ("このアーティストの名前は", "minfo", None),
    ("空気圧って今いくつ", "status", None), ("空気圧の適正値は", "manual", None),
    ("オイルは今どのくらい減ってる", "status", None), ("オイルは何キロで交換するの", "manual", None),
    ("いまのバッテリー電圧を見せて", "status", None), ("バッテリーの寿命は何年くらい", "manual", None),
    ("水温は今何度", "status", None), ("水温が高いときはどうすればいい", "manual", None),
    ("燃料はあと何リットル", "status", None), ("燃料タンクの大きさは", "manual", None),
    ("近くのガソリンスタンドを教えて", "poi", None), ("近くのガソリンスタンドに行って", "nav", None),
    ("駐車場はどこにある", "poi", None), ("一番近い駐車場まで案内して", "nav", None),
    ("コンビニの場所を調べて", "poi", None), ("近くのコンビニに寄って", "nav", None),
    ("充電できるところを一覧で見せて", "poi", None), ("最寄りの充電スポットへ向かって", "nav", None),
    ("後ろのドアは閉まってる", "status", None), ("後ろに車が来てないか見せて", "cam", None),
]
M = [
    "エアコンつけて、音楽もかけて", "家まで案内して、あと音量下げて", "ガソリンの残りを教えて、近くのスタンドも探して",
    "音楽を止めて、東京駅までナビして", "後ろを映して、音量も下げて", "充電スポットを探して、そこに向かって",
    "この曲の名前を教えて、次の曲にして", "空気圧を確認して、適正値も教えて", "エアコンを消して、ラジオをつけて", "駐車場を探して、周りも映して",
]
NOTES = {"A": "言い換え", "B": "長い文・前置きつき", "C": "言いよどみ・口語", "D": "音声認識の誤り風", "E": "敬語・方言・カタカナ語",
         "F": "機能の語を含む雑談", "G": "意図が不明", "H": "機能にない依頼", "I": "読むだけか操作かの取り違え",
         "M": "複数の依頼（参考）", "V": "走行中"}


def main() -> None:
    rows = []

    def add(cat: str, k: int, utt: str, act: str, arg, motion: str = "stopped", src: str | None = None) -> None:
        rows.append({"id": f"ch_{cat}_{k:03d}", "category": cat, "category_note": NOTES[cat], "utterance": utt, "history": [],
                     "state": {"motion": motion, "permission": "allowed"}, "action": T[act], "argmode": arg,
                     "trainable": False, "purpose": "challenge_eval", **({"source_id": src} if src else {})})

    for cat, items in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("I", I)):
        for k, (u, a, m) in enumerate(items):
            add(cat, k, u, a, m)
    for k, u in enumerate(F):
        add("F", k, u, "chat", None)
    for k, u in enumerate(G):
        add("G", k, u, "unk", None)
    for k, u in enumerate(H):
        add("H", k, u, "unk", None)
    for k, u in enumerate(M):
        add("M", k, u, "multi", None)
    # 走行中: A の先頭 15・F の先頭 8・G の先頭 7 を、状態だけ走行中にする
    base = [r for r in rows if r["category"] == "A"][:15] + [r for r in rows if r["category"] == "F"][:8] + \
           [r for r in rows if r["category"] == "G"][:7]
    for k, r in enumerate(base):
        act = next(key for key, v in T.items() if v == r["action"])
        add("V", k, r["utterance"], act, r["argmode"], motion="moving", src=r["id"])

    # 確かめ: 種類の中の重なり、714 行との重なり
    seen = collections.Counter(r["utterance"] for r in rows if r["category"] != "V")
    dup = [u for u, c in seen.items() if c > 1]
    old = {json.loads(l)["utterance"] for l in ROUTER_DATA.read_text(encoding="utf-8").splitlines() if l.strip()}
    hit = sorted({r["utterance"] for r in rows if r["utterance"] in old})
    if dup:
        raise SystemExit(f"セットの中で重なっている発話: {dup}")
    if hit:
        raise SystemExit(f"714 行（synthetic_v2）と同じ発話がある（学習に使った文は入れない）: {hit}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    c = collections.Counter(r["category"] for r in rows)
    print(f"{len(rows)} 行: " + "・".join(f"{k} {c[k]}" for k in sorted(c)))
    print("行き先: " + "・".join(f"{k} {v}" for k, v in collections.Counter(r["action"] for r in rows).most_common()))
    ln = [len(r["utterance"]) for r in rows]
    print(f"発話の長さ: 平均 {sum(ln) / len(ln):.1f} 文字・最長 {max(ln)} 文字（714 行は平均 8.5・最長 17）")
    print(f"保存しました: {OUT}（714 行との重なり 0）")


if __name__ == "__main__":
    main()
