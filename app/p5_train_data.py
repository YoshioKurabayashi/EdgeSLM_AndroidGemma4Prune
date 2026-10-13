# -*- coding: utf-8 -*-
"""P5 手順 1: 学習に足す発話（train_add_v1）を作り、714 行と合わせた学習データを書く。

設計: DoC/設計書_P5_学習データを足してヘッドを学習し直す.md の 3 節。

P4 で弱かった種類（長い文・口語・敬語や方言・機能にない依頼・意図が不明）を、**評価セット（challenge_v1）とは別の文**で足す。
評価セットの文そのものは学習に入れない（作るときに、同じ文・よく似た文が無いことを確かめる）。

|種類|行|ラベル|中身|
|---|---|---|---|
|LG 長い文・前置きつき|54|各機能・雑談（9 クラス × 6）|理由や状況の前置きのあとに依頼が来る文|
|CQ 言いよどみ・口語|54|各機能・雑談（9 クラス × 6）|「えっと」「ねえ」、くだけた語尾|
|PL 敬語・方言・カタカナ語|27|各機能・雑談（9 クラス × 3）|1 クラスにつき敬語 1・方言 1・カタカナ語 1|
|OS 機能にない依頼|60|UNKNOWN（棄却ヘッドの教師）|20 の機能 × 3 文。うち 10 は評価セット H の半分（15 行）と同じ機能を別の言い方で、10 は H に無い機能|
|AM 意図が不明|24|UNKNOWN（棄却ヘッドの教師）|途中で切れる・自己矛盾・指示語・取り消し・曖昧|

ラベルの決め方は 714 行（EdgeSLM_MultiHead_Multi `data/synthetic_v2`）に合わせた:
- UNKNOWN は分類ヘッド（行き先）のクラスに入れない。棄却ヘッドの教師にだけ使う（`route=UNKNOWN`・`trainable=false`）
- 「少し上げて」は空調、「もっと大きく」「止めて」はメディア（714 行の決め方）。対象の無い度合いだけの文は、ここでは足さない
- 足した行には ArgMode を付けない（`argmode=null`）。ArgMode のヘッドは 714 行の教師のまま

分け方（split）: 3 文で 1 つの単位（`split_unit`）。単位ごとに train・validation・test を決める（クラスごとに 3:1:1 で回す）。
交差検証は `split_unit` とラベルで分けるので、単位が fold をまたがない。

    python -m app.p5_train_data     # data/train_add_v1/added.jsonl（足した行）・utterances.jsonl（714 行＋足した行）
"""
from __future__ import annotations

import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, ROUTER_DATA

OUT = PROJECT_ROOT / "data" / "train_add_v1"
CHALLENGE = PROJECT_ROOT / "data" / "challenge_v1" / "utterances.jsonl"
MAX_CHARS = 40          # 評価セットの最長と同じ（64 トークンに収まる長さ）
NEAR = 0.60             # 評価セットの文との似かた（2 文字組の重なり）がこれ以上なら止める
CLASSES = ["hvac", "media", "nav", "status", "manual", "poi", "minfo", "cam", "chat"]
TOOL = {"hvac": "control_hvac", "media": "control_media", "nav": "set_navigation", "status": "get_vehicle_status",
        "manual": "search_vehicle_manual", "poi": "search_poi", "minfo": "get_media_info", "cam": "capture_camera"}
TYPE_NOTE = {"LG": "長い文・前置きつき", "CQ": "言いよどみ・口語", "PL": "敬語・方言・カタカナ語",
             "OS": "機能にない依頼", "AM": "意図が不明"}

# ---------------------------------------------------------------- 行き先のある行（3 文で 1 単位）
LG = {
    "hvac": [["日差しが強くて車内がむっとしてきたから、冷房を入れてもらえる",
              "足元が冷えてつらいので、暖房の温度をもう少し上げてほしい",
              "助手席の人が寒がっているから、風量を少し弱めてあげて"],
             ["もうすぐ家に着くから、エアコンはここで止めてしまって大丈夫",
              "朝は冷え込んでフロントが真っ白だから、デフロスターを入れて",
              "みんな汗をかいているみたいだから、設定温度を23度まで下げて"]],
    "media": [["静かすぎて落ち着かないから、何か曲を流してもらえるかな",
               "話し声が聞こえにくいので、音楽のボリュームを少し絞ってほしい",
               "この歌はあまり好きじゃないから、ひとつ先の曲に送って"],
              ["ニュースが聴きたい時間だから、ラジオに切り替えてくれる",
               "いまから大事な話をするので、流れている曲を一時停止にして",
               "さっきのサビをもう一度聴きたいから、ひとつ前の曲に戻してほしい"]],
    "nav": [["午後から病院の予約があるので、市民病院までの道案内を始めて",
             "子どもを迎えに行く時間だから、小学校を目的地に入れてほしい",
             "週末は実家に顔を出すことにしたから、実家までのルートを設定して"],
            ["道が混んできたみたいだから、高速を使わないルートに変えてくれる",
             "忘れ物を取りに戻りたいので、行き先をいったん自宅に変更して",
             "もう場所はわかったから、いまやっているルート案内は終了していいよ"]],
    "status": [["遠出の前に念のため知りたいんだけど、ガソリンはどれくらい入ってる",
                "しばらく空気を入れていないから、タイヤの空気圧の今の値を見せて",
                "半ドアの警告音が鳴った気がするので、ドアの状態を確認してくれる"],
               ["次の給油までたどり着けるか不安だから、あと何キロ走れるか教えて",
                "そろそろ点検の時期かもしれないから、今の総走行距離を知りたい",
                "朝エンジンのかかりが悪かったので、バッテリーの状態を見ておいて"]],
    "manual": [["メーターに見慣れないマークが出ているんだけど、これの意味を調べて",
                "雪道を走る予定があるので、チェーンの付け方を説明書で確認したい",
                "初めてこの車で遠出するから、推奨されている燃料の種類を教えて"],
               ["スマートキーの電池が切れそうなので、電池交換の手順を知りたい",
                "ウォッシャー液が出なくなったから、どこから補充するのか教えて",
                "パンクしたときに困らないように、スペアタイヤの場所を調べておいて"]],
    "poi": [["少し休憩したくなってきたから、この先にある道の駅を探してくれる",
             "郵便を出したいので、近くにある郵便局を調べて",
             "お見舞いに行く途中なので、この辺りで開いている花屋を探して"],
            ["小腹が空いたので、近所でテイクアウトできるお店を検索してほしい",
             "洗車をしておきたいから、この近くの洗車場を探してもらえる",
             "子どもが飽きてきたみたいだから、近くに公園がないか調べてみて"]],
    "minfo": [["どこかで聴いたことがある気がするんだけど、いま流れているのは何て曲",
               "この声がすごく好みなので、歌っている人の名前を知りたい",
               "あとで買いたいから、この曲が入っているアルバムを教えてくれる"],
              ["ずいぶん長い曲みたいだけど、あとどのくらいで終わるのか教えて",
               "いつの間にか切り替わったみたいだけど、いま何の放送局が流れてるの",
               "いま聴いているのがラジオなのかスマホの音楽なのか、どっちか教えて"]],
    "cam": [["後ろに自転車がいないか心配だから、リアカメラの映像を映して",
             "縁石にこすりそうで怖いので、左前のタイヤのあたりを見せてほしい",
             "狭い道ですれ違うから、右のミラーの下あたりを映してほしい"],
            ["車の真後ろに何か置いてある気がするから、後方の画面を出してくれる",
             "ぐるっと一周確認したいので、車のまわり全体の映像を表示して",
             "前に低い障害物があるかもしれないから、フロントカメラで見せて"]],
    "chat": [["今日は道がすいていて気持ちいいね、こういう日は遠くまで走りたくなる",
              "子どものころは家族でよくドライブに出かけて、後ろの席で寝てたんだよ",
              "最近は仕事が忙しくて、なかなかゆっくり出かける時間が取れないんだ"],
             ["この前食べたラーメンがおいしくて、また行きたいなと思ってるんだよね",
              "昔は音楽をかけながら夜の海沿いを走るのが好きだったな",
              "エアコンの効いた車内から出ると、外の暑さにびっくりするよね"]],
}

CQ = {
    "hvac": [["あー暑っ、冷房入れて", "えっと、暖房、もうちょい強めで", "ねえ、風、止めてくんない"],
             ["んー、なんか寒いな、温度上げてよ", "あ、そうだ、24度にしといて", "ちょ、エアコン強すぎ、弱めて"]],
    "media": [["あ、なんか曲流してよ", "えーと、音、もうちょい上げて", "ねえ、これ飛ばして、次いって"],
              ["んーと、いっこ前の曲に戻して", "あ、うるさい、音小さくして", "えっと、FMつけてくんない"]],
    "nav": [["えーと、駅、最寄りの駅まで連れてって", "あ、そうだ、実家まで案内よろしく", "んー、やっぱ会社じゃなくて家に向かって"],
            ["ねえ、ナビ、渋谷でセットして", "あのさ、もう案内いらないや、やめて", "えっと、大阪駅、大阪駅まで行きたい"]],
    "status": [["あれ、ガソリンまだあったっけ", "えーと、タイヤの空気、今どんな感じ", "ねえ、トランクちゃんと閉まってる"],
               ["んー、電池あと何パーくらい", "あ、そういえば今まで何キロ走ったんだっけ", "えっと、あとどんだけ走れそう"]],
    "manual": [["えっと、このマークなに、どういうこと", "ねえ、ワイパーのゴムってどうやって替えんの", "あのさ、給油ってレギュラーでいいんだっけ"],
               ["んー、タイヤっていつ替えればいいの", "あ、そうだ、チャイルドシートってどこに付けんの", "えーと、キーの電池ってどう替えるんだっけ"]],
    "poi": [["あ、腹へった、近くにメシ屋ない", "えーと、この辺、トイレあるとこない", "ねえ、近くにスーパーあったっけ"],
            ["んー、コーヒー飲みたい、カフェ探して", "あのさ、近くで車停めれるとこ調べて", "えっと、近所にパン屋ってある"]],
    "minfo": [["あ、これ何て曲だっけ", "ねえ、これ歌ってんの誰", "えーと、今かかってんのどのアルバム"],
              ["んー、今どのへんまで再生した", "あのさ、今ラジオ、どこの局", "えっと、さっきからかかってるやつ、曲名わかる"]],
    "cam": [["あ、ちょい、後ろ映して", "えーと、左、左のほう見して", "ねえ、前のカメラ出してくんない"],
            ["んー、周りどうなってる、映して", "あのさ、バックの画面出して", "えっと、右の後ろ、カメラで見たい"]],
    "chat": [["あー疲れた、今日ほんと長かったわ", "ねえ、昨日のドラマ見た、すごかったよね", "んー、今日は月曜だからなんかだるいなあ"],
             ["えっと、そういえば来週って連休だっけ", "あのさ、最近ちょっと運動不足なんだよね", "あ、見て、空がすごいきれい"]],
}

# 1 クラスにつき [敬語, 方言, カタカナ語]
PL = {
    "hvac": [["冷房を少し弱めていただけますでしょうか", "暑うてかなわん、冷房つけてえな", "クーラーのパワーをアップして"]],
    "media": [["次の曲へ進めていただけますか", "音ちょっと大きゅうしてくれへん", "プレイバックをポーズして"]],
    "nav": [["目的地を市役所に設定していただけますか", "実家まで連れてってくれへんか", "デスティネーションを横浜駅にセット"]],
    "status": [["現在の走行可能距離をお教えいただけますか", "タイヤの空気、今どんくらい入っとるん", "バッテリーレベルをチェックして"]],
    "manual": [["オイル交換の時期についてご教示いただけますか", "この警告灯、どげん意味ね", "ワイパーブレードのリプレイス方法をガイドして"]],
    "poi": [["この付近の飲食店を検索していただけますか", "近くにうまい飯屋あらへんか", "ニアバイのカフェをリストアップして"]],
    "minfo": [["ただいまの曲のアーティスト名を教えていただけますか", "これ誰の歌やったっけ", "ナウプレイングのタイトルは"]],
    "cam": [["前方カメラの映像を映していただけますか", "右のほう映してくれんかね", "サイドカメラをディスプレイに出して"]],
    "chat": [["いつも丁寧に対応していただいて感謝しております", "今日はほんまにええ天気やなあ", "ドライブってリフレッシュになるよね"]],
}

# ---------------------------------------------------------------- 機能にない依頼（UNKNOWN）
# (単位の名前, 評価セット H に同じ機能があるか, 3 文)
OS = [
    ("window", True, ["窓を閉めて", "運転席の窓を少し下げて", "助手席の窓を全開にして"]),
    ("horn", False, ["クラクションを鳴らして", "ホーンを鳴らしてほしい", "警笛を一回鳴らして"]),
    ("light", True, ["ヘッドライトを消して", "室内灯をつけて", "フォグランプを点けて"]),
    ("display", False, ["画面を明るくして", "メーターの明るさを下げて", "ディスプレイを夜間モードにして"]),
    ("phone", True, ["妻に電話をかけて", "かかってきた電話に出て", "会社に電話をつないで"]),
    ("network", False, ["Wi-Fiに接続して", "テザリングをオンにして", "ネットにつないで"]),
    ("message", True, ["田中さんにLINEを送って", "届いたメッセージを読んで", "返信を打って"]),
    ("update", False, ["ソフトウェアを更新して", "地図データを最新にして", "システムを再起動して"]),
    ("lock", True, ["ドアをロックして", "全部のドアの鍵を閉めて", "ロックを解除して"]),
    ("home", False, ["家の電気を消して", "お風呂を沸かしておいて", "家のテレビを録画して"]),
    ("trunk", True, ["トランクを閉めて", "バックドアを開けて", "荷室を開けてほしい"]),
    ("memo", False, ["メモを取って", "今の会話を録音して", "音声メモを残して"]),
    ("seat", True, ["シートを前に出して", "背もたれを起こして", "座席を一番後ろまで下げて"]),
    ("sns", False, ["SNSに投稿して", "この景色をみんなにシェアして", "写真をアップして"]),
    ("wiper", True, ["ワイパーを止めて", "ワイパーを速くして", "後ろのワイパーも使って"]),
    ("steering", False, ["ハンドルの位置を上げて", "ステアリングを手前に引いて", "ペダルの位置を調整して"]),
    ("engine", True, ["エンジンをかけて", "エンジンを止めておいて", "アイドリングストップをオフにして"]),
    ("settings", False, ["表示の言語を英語に変えて", "時計を合わせて", "文字を大きくして"]),
    ("sunroof", True, ["サンルーフを閉めて", "ルーフを少しだけ開けて", "サンシェードを閉じて"]),
    ("app", False, ["ゲームを起動して", "ブラウザを開いて", "アプリをインストールして"]),
]
# 評価セット H のうち、上の「同じ機能がある」10 単位と同じ機能の行（15 行）。残りの 15 行は、学習に同じ機能が無い
CHALLENGE_H_SEEN = [
    "窓を開けて", "後ろの席の窓をロックして", "ライトをつけて", "ヘッドライトをハイビームにして", "ルームランプを消して",
    "母さんに電話して", "電話を切って", "メッセージを送って", "メールを読み上げて", "ドアのロックを開けて",
    "トランクを開けて", "シートを倒して", "ワイパー動かして", "エンジンを切って", "サンルーフを開けて",
]
# 評価セット G・V のうち、ラベルが 714 行の決め方と食い違う行（714 行では「少し上げて」は空調、「もっと大きく」はメディア）。
# 評価セットは作り直さず、集計のときに別に数える
CHALLENGE_DISPUTED = ["もっと上げて", "もうちょっと強く"]

# ---------------------------------------------------------------- 意図が不明（UNKNOWN）
AM = [
    ("incomplete_1", ["風量を", "行き先は", "次の"]),
    ("incomplete_2", ["カメラで", "ボリュームが", "曲を、あの"]),
    ("selfcancel", ["探して、あ、もういいや", "下げて、あ、やっぱ上げて、いやいいや", "戻って、いや、そのまま進んで"]),
    ("deictic", ["あれをもう一度", "向こうのにして", "そっちじゃないほう"]),
    ("prevplace", ["この前の店に向かって", "さっき言ったところまで", "例の場所に案内して"]),
    ("cancel", ["今のはキャンセル", "さっきのは間違い", "それは無しにして"]),
    ("vague", ["いいようにしといて", "そこそこで頼む", "なんとかならないかな"]),
    ("opposite", ["反対にして", "入れ替えて", "別のにして"]),
]

SPLIT_CYCLE = ["train", "train", "validation", "test", "train"]     # 3:1:1


def bigrams(s: str) -> set:
    s = s.replace("、", "").replace(" ", "")
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def near(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y)


def build() -> list[dict]:
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
                lab = {"route": "TOOLCALL", "category": t["category"], "effect": t["effect"], "tool": t["name"],
                       "action": f"TOOL_{t['name']}"}
            rows.append({"id": f"{unit}:{k}", "group": unit, "split_unit": unit, "split": split, "utterance": u, "history": [],
                         "state": {"motion": "stopped", "permission": "allowed"}, "last_tool": None, **lab, "argmode": None,
                         "difficulty": "hard", "confusable": None, "note": TYPE_NOTE[kind],
                         "trainable": cls is not None, "purpose": "classification" if cls is not None else "rejection_eval",
                         "added": "train_add_v1", "add_type": kind, **(extra or {})})

    # 行き先のある行: クラスごとに 5 単位（LG 2・CQ 2・PL 1）。3:1:1 の回し方を、クラスごとに 1 つずつずらす
    for ci, cls in enumerate(CLASSES):
        units = [("LG", u) for u in LG[cls]] + [("CQ", u) for u in CQ[cls]] + [("PL", u) for u in PL[cls]]
        if len(units) != 5:
            raise SystemExit(f"{cls}: 単位が 5 つでない")
        for ui, (kind, utts) in enumerate(units):
            n = sum(1 for k, _ in units[:ui] if k == kind)
            add(kind, f"add_{kind.lower()}_{cls}_{n}", SPLIT_CYCLE[(ui + ci) % 5], utts, cls)
    for ui, (name, seen, utts) in enumerate(OS):
        add("OS", f"add_os_{name}", SPLIT_CYCLE[ui % 5], utts, None, {"os_function_in_challenge": seen})
    for ui, (name, utts) in enumerate(AM):
        add("AM", f"add_am_{name}", SPLIT_CYCLE[(ui + 2) % 5], utts, None)
    return rows


def main() -> None:
    rows = build()
    base = [json.loads(l) for l in ROUTER_DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    chal = [json.loads(l) for l in CHALLENGE.read_text(encoding="utf-8").splitlines() if l.strip()]
    old = {r["utterance"] for r in base}
    ch = sorted({r["utterance"] for r in chal})

    # 確かめ 1: 足した行の中・714 行・評価セットとの重なり（同じ文は入れない）
    c = collections.Counter(r["utterance"] for r in rows)
    dup = [u for u, k in c.items() if k > 1]
    hit_old = sorted({r["utterance"] for r in rows if r["utterance"] in old})
    hit_ch = sorted({r["utterance"] for r in rows if r["utterance"] in set(ch)})
    long = [r["utterance"] for r in rows if len(r["utterance"]) > MAX_CHARS]
    if dup or hit_old or hit_ch or long:
        raise SystemExit(f"重なり・長さの問題: 足した行の中 {dup} / 714 行と同じ {hit_old} / 評価セットと同じ {hit_ch} / {MAX_CHARS} 文字超 {long}")
    # 確かめ 2: 評価セットとよく似た文（2 文字組の重なり）。機能にない依頼は、同じ機能を別の言い方で入れる単位があるので別に見る
    pairs = sorted(((near(r["utterance"], u), r["utterance"], u, r["add_type"]) for r in rows for u in ch), reverse=True)
    print("評価セットの文と一番似ている組（2 文字組の重なり）:")
    for s, a, b, k in pairs[:12]:
        print(f"  {s:.2f} [{k}]「{a}」↔「{b}」")
    bad = [(round(s, 2), a, b) for s, a, b, k in pairs if s >= NEAR]
    if bad:
        raise SystemExit(f"評価セットの文と似すぎている（{NEAR} 以上）: {bad}")
    # 確かめ 3: 評価セット H の「同じ機能がある」行が、評価セットにあること
    hset = {r["utterance"] for r in chal if r["category"] == "H"}
    miss = [u for u in CHALLENGE_H_SEEN if u not in hset]
    if miss or len(CHALLENGE_H_SEEN) != 15:
        raise SystemExit(f"CHALLENGE_H_SEEN が評価セット H と合わない: {miss}")
    # 確かめ 4: split ごと・クラスごとに行がある
    by = collections.Counter((r["split"], r["action"]) for r in rows)
    for sp in ("train", "validation", "test"):
        for a in sorted({r["action"] for r in rows}):
            if by[(sp, a)] == 0:
                raise SystemExit(f"split={sp} に {a} の行が無い")

    OUT.mkdir(parents=True, exist_ok=True)
    dump = lambda rs: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    (OUT / "added.jsonl").write_text(dump(rows), encoding="utf-8")
    (OUT / "utterances.jsonl").write_text(dump(base + rows), encoding="utf-8")     # 714 行はそのまま・同じ並びで先頭に置く
    t = collections.Counter(r["add_type"] for r in rows)
    print(f"足した行 {len(rows)}: " + "・".join(f"{k} {TYPE_NOTE[k]} {t[k]}" for k in TYPE_NOTE))
    print("行き先: " + "・".join(f"{k} {v}" for k, v in collections.Counter(r["action"] for r in rows).most_common()))
    print("split: " + "・".join(f"{k} {v}" for k, v in collections.Counter(r["split"] for r in rows).most_common()))
    ln = [len(r["utterance"]) for r in rows]
    print(f"発話の長さ: 平均 {sum(ln) / len(ln):.1f} 文字・最長 {max(ln)} 文字")
    kn = sum(r["trainable"] for r in base + rows)
    print(f"合わせた学習データ: {len(base) + len(rows)} 行（714 行＋{len(rows)} 行）= 行き先のある行 {kn}・UNKNOWN {len(base) + len(rows) - kn}")
    print(f"保存しました: {OUT}（added.jsonl・utterances.jsonl。714 行・評価セットとの重なり 0）")


if __name__ == "__main__":
    main()
