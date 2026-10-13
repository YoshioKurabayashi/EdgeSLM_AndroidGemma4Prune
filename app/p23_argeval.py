# -*- coding: utf-8 -*-
"""P23 手順 3: 引数の値だけを測る、2 つ目の新しい行（arg_eval_v2・240 行）。端末なし。

設計: DoC/設計書_P22_引数の値を測って直す.md の 8 節。

arg_eval_v1 は、P22 の結果を見て `p23_arguments` を作ったので「直す用」になった。**直しが、見ていない行でどこまで効くか**を、もう一度新しい行で測る。
`p23_arguments` を決めた後に書いた。正解の引数は、文の意味だけから付けた（取り出しの結果を見る前）。

形は 10 個 × 3 機能 × 8 行。形 1〜8 は arg_eval_v1 と同じ（文は別）。**形 9・10 は、arg_eval_v1 に無かった形**（直しを作るときに見ていない形）。

|形|中身|例|
|---|---|---|
|1〜8|arg_eval_v1 と同じ形（ふつう・理由が前・理由が後ろ・ていねい・くだけた・条件や時つき・数や場所の言い方・2 つのこと）|—|
|9 質問・願望|〜できる・〜したいな・〜ならないかな|冷房つけられる／東京駅に行きたいんだけど|
|10 長い文・言いよどみ|えっと・あの、で始まり、理由を言ってから頼む|えっと、ちょっと寒くなってきたんだけど、暖房を入れてもらってもいいかな|

    python -m app.p23_argeval build       # data/arg_eval_v2/utterances.jsonl（重なりを確かめる）
    python -m app.p23_argeval measure     # results/P23/argeval.md・argeval.json
"""
from __future__ import annotations

import argparse
import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT, use_multi_library
from .p5_train_data import near
from .p7_data import CH1, CH2, read_jsonl
from .p13_data import CH3
from .p18_data import CH4, TR3, md5_of
from .p20_data import CH5
from .p21_data import CH6
from .p22_argeval import FORMS as FORMS1, J, NEAR, OUT as ARG1
from .p22_labels import match, parse

OUT = PROJECT_ROOT / "data" / "arg_eval_v2"
RES = PROJECT_ROOT / "results" / "P23"
FORMS = FORMS1 + ["9 質問・願望", "10 長い文・言いよどみ"]
NEW_FORMS = FORMS[8:]
PER = 8

HVAC = [
    [("冷房を24度に", "t24|t24,m:cool"), ("風量を6に上げて", "f6"), ("空調を動かして", "m:auto"), ("暖房を消して", "m:off"), ("曇り取りを入れて", "m:defrost"), ("温度を下げてほしい", "t24"),
     ("風を強くして", "f4"), ("ヒーターを入れて", "m:heat")],
    [("西日がまぶしくて暑いので温度を下げて", "t24"), ("膝が冷えるんでヒーターをつけて", "m:heat"), ("窓の内側が曇ってきたため曇りを取って", "m:defrost"), ("のどが渇くし風量を落として", "f2"),
     ("雨に濡れて寒いから暖房を強めて", "f4"), ("もうすぐ降りるんで空調を切って", "m:off"), ("人が増えて蒸すので冷房を入れて", "m:cool"), ("犬が暑そうだから後ろの風量を5にして", "f5")],
    [("暖房を入れて、指先が冷たいんだ", "m:heat"), ("温度を2度下げて、眠くなりそうだから", "t23"), ("風量を下げて、書類が飛ぶから", "f2"), ("曇りを取って、何も見えないよ", "m:defrost"),
     ("冷房を強くして、全然冷えないね", "f4"), ("エアコンを止めて、窓を開けるから", "m:off"), ("26度にして、ちょっと肌寒い", "t26"), ("風を弱めて、髪が乱れるの", "f2")],
    [("暖房を入れていただけますでしょうか", "m:heat"), ("温度を1度下げていただけるとありがたいです", "t24"), ("風量を3にしていただけますか", "f3"), ("冷房を止めていただいて大丈夫です", "m:off"),
     ("曇りを取ってもらえると助かります", "m:defrost"), ("もう少し涼しくしていただけませんか", "t24"), ("よろしければ冷房を強めてください", "f4"), ("エアコンをつけてもらってもよろしいでしょうか", "m:auto")],
    [("暖房つけて", "m:heat"), ("冷房きって", "m:off"), ("風つよめで", "f4"), ("さむ、暖房強くして", "f4"), ("26度で頼む", "t26"), ("曇りとって", "m:defrost"), ("クーラーつけてよー", "m:cool"), ("ちょい温度下げて", "t24")],
    [("会社に着くまで暖房を入れておいて", "m:heat"), ("霧が晴れるまで曇り取りをつけといて", "m:defrost"), ("赤ちゃんが寝てるうちに風量を1にして", "f1"), ("給油してるあいだはエアコンを止めておいて", "m:off"),
     ("ドアを閉めたら温度を23度にして", "t23"), ("荷物を積み終わったら冷房を入れて", "m:cool"), ("出発する前に車内を暖かくしておいて", "t26"), ("高速に乗るまで風を弱くしといて", "f2")],
    [("温度を十八度にして", "t18"), ("風量を三にして", "f3"), ("温度を2度だけ上げて", "t27"), ("風量をひとつ下げて", "f2"), ("温度を最低にして", "t16"), ("風量をマックスにして", "f7"), ("温度を1.5度上げて", "t26.5"),
     ("風を二段強くして", "f5")],
    [("風量を上げて温度も下げて", "f4,t24"), ("暖房ではなく冷房に切り替えて", "m:cool"), ("温度は24度、風量は2にして", "t24,f2"), ("冷房は切らないで弱めて", "f2"), ("風量じゃなくて温度を下げて", "t24"),
     ("デフロスターを止めて暖房を入れて", "m:heat"), ("温度を上げるんじゃなくて風を弱めて", "f2"), ("エアコンは止めずに温度だけ上げて", "t26")],
    [("冷房つけられる", "m:cool"), ("暖房を入れてもいいかな", "m:heat"), ("もう少し暖かくならないかな", "t26"), ("風を弱くしたいんだけど", "f2"), ("温度を22度にできる", "t22"), ("曇りを取りたいな", "m:defrost"),
     ("エアコンを止めたい", "m:off"), ("冷房を強くできないかな", "f4")],
    [("えっと、ちょっと寒くなってきたんだけど、暖房を入れてもらってもいいかな", "m:heat"), ("あのさ、後ろの人が暑いって言ってるから、冷房をもう少し強くしてあげて", "f4"),
     ("うーん、なんか窓が曇ってきて前が見づらいから、曇りを取ってくれると助かる", "m:defrost"), ("そうだな、さっきから風が強すぎる気がするので、風量を2まで下げてくれる", "f2"),
     ("えー、もう十分涼しくなったと思うから、そろそろ冷房は止めてもらえるかな", "m:off"), ("あの、いまの温度だと少し暑いので、設定を23度に変えてほしいんだけど", "t23"),
     ("ねえ、子どもが寝ちゃったみたいだから、風が当たらないように風量を一番弱くして", "f0|f1"), ("んー、やっぱり少し寒い気がするから、温度を1度だけ上げておいてくれる", "t26")],
]
MEDIA = [
    [("曲をかけて", "play"), ("いまの再生を止めて", "pause"), ("次の曲を流して", "next"), ("ひとつ前に戻って", "prev"), ("音量を22に", "v22"), ("音をもっと大きく", "v14"), ("音を消音に切り替えて", "mute"), ("ラジオが聞きたい", "play")],
    [("眠気覚ましに何かかけて", "play"), ("電話が鳴ってるので音を下げて", "v10"), ("この歌は苦手だから次に飛ばして", "next"), ("窓を開けたら聞こえないんで音量を上げて", "v14"), ("寝かしつけたいし音を小さくして", "v10"),
     ("話しかけられたため音楽を止めて", "pause"), ("歌詞を聞き逃したから前の曲に戻って", "prev"), ("うるさいって言われたから音量を6にして", "v6")],
    [("音楽を止めて、ちょっと考えごとをするから", "pause"), ("音量を上げて、ここのサビが好きなんだ", "v14"), ("次の曲にして、この歌は暗いね", "next"), ("音を小さくして、頭が痛いの", "v10"), ("曲をかけて、退屈だよ", "play"),
     ("ミュートにして、後ろで電話してるから", "mute"), ("前の曲に戻して、聞きそびれた", "prev"), ("音量を10にして、夜だから", "v10")],
    [("曲を止めていただけますか", "pause"), ("音量を上げていただけると助かります", "v14"), ("次の曲に変えていただけますでしょうか", "next"), ("音楽を流してもらえるとうれしいです", "play"),
     ("少し音を小さくしていただけませんか", "v10"), ("音量を16にしていただけますか", "v16"), ("前の曲に戻ってもらえますか", "prev"), ("よろしければ消音にしてください", "mute")],
    [("曲かけて", "play"), ("音さげて", "v10"), ("次の曲いって", "next"), ("とめて", "pause"), ("音量15で", "v15"), ("音おおきく", "v14"), ("ミュートで", "mute"), ("前の曲もどって", "prev")],
    [("家に着くまで音楽をかけておいて", "play"), ("電話してるあいだは音楽を止めておいて", "pause"), ("子どもが起きるまで音量を下げておいて", "v10"), ("信号が変わる前に次の曲にして", "next"),
     ("トンネルを出るまでラジオを止めておいて", "pause"), ("会話が終わるまでミュートにしておいて", "mute"), ("料金所を過ぎたら音量を上げて", "v14"), ("休憩が終わったら音楽を再開して", "play")],
    [("音量を十五にして", "v15"), ("音を2つ分下げて", "v10"), ("音量を3倍にして", "v36"), ("ボリュームをいちばん下まで下げて", "v0"), ("音量をいまの半分くらいにして", "v6"), ("音を気持ち上げて", "v14"), ("音量を2だけ下げて", "v10"),
     ("ボリュームをマックスにして", "v40")],
    [("止めるんじゃなくて次の曲にして", "next"), ("音量は下げないで上げて", "v14"), ("曲は変えないで音だけ小さくして", "v10"), ("ラジオはやめて音楽をかけて", "play"), ("消音じゃなくて一時停止にして", "pause"),
     ("次の曲じゃなくて前の曲に戻して", "prev"), ("音楽を止めずに音量を8にして", "v8"), ("再生じゃなくて停止して", "pause")],
    [("音楽かけられる", "play"), ("音を大きくできる", "v14"), ("次の曲にしたいな", "next"), ("この曲止めたいんだけど", "pause"), ("音量を20にできるかな", "v20"), ("もう少し音が小さくならないかな", "v10"),
     ("前の曲をもう一回聞きたい", "prev"), ("ラジオを聞きたいな", "play")],
    [("えっと、後ろの子がそろそろ寝そうだから、音楽の音を少し小さくしてもらえるかな", "v10"), ("あの、いま流れてるのはあまり好みじゃないので、次の曲に変えてくれると助かる", "next"),
     ("うーん、ちょっと静かすぎて眠くなりそうだから、何か音楽をかけてほしいんだけど", "play"), ("ねえ、電話がかかってきたみたいだから、いったん音楽を止めてもらってもいい", "pause"),
     ("そうだ、さっきの曲がすごく良かったから、ひとつ前の曲に戻してくれる", "prev"), ("えー、ナビの声が全然聞こえないから、音量を8くらいまで下げてほしいな", "v8"),
     ("あのさ、いま大事な話をしてるところだから、しばらく消音にしておいて", "mute"), ("んー、高速に入ってから音が聞こえにくいので、ボリュームを少し上げてもらえる", "v14")],
]
NAV = [
    [("名古屋城まで案内して", "set:名古屋城"), ("中央図書館に向かって", "set:中央図書館"), ("目的地を病院に設定して", "set:病院"), ("道案内はもう終わって", "cancel"), ("市民プールへ行って", "set:市民プール"),
     ("別の道を探して", "reroute"), ("経由地に銀行を追加して", "way:銀行"), ("会社に戻って", "set:会社")],
    [("歯が痛いから歯医者まで案内して", "set:歯医者"), ("牛乳が切れたんでスーパーに向かって", "set:スーパー"), ("充電が減ってきたため充電スタンドへ向かって", "set:充電スタンド"), ("時間が余ったし美術館まで案内して", "set:美術館"),
     ("祖母が待ってるので老人ホームを目的地にして", "set:老人ホーム"), ("道が混んでるから高速を使うルートにして", "reroute"), ("のどが渇いたからコンビニに寄るルートにして", "way:コンビニ"), ("財布を忘れたんで家に戻って", "set:家")],
    [("歯医者に向かって、予約の時間なんだ", "set:歯医者"), ("駅まで案内して、電車に間に合わないよ", "set:駅"), ("目的地をホテルにして、もう疲れた", "set:ホテル"), ("スーパーに寄るルートにして、卵を買い忘れた", "way:スーパー"),
     ("案内を止めて、ここからは分かるから", "cancel"), ("下道にして、高速は混んでるね", "reroute"), ("学校へ向かって、子どもを迎えに行くの", "set:学校"), ("経由地に花屋を追加して、お見舞いだから", "way:花屋")],
    [("市役所まで案内していただけますでしょうか", "set:市役所"), ("空港に向かっていただけますか", "set:空港"), ("目的地を駅前のホテルに設定していただけると助かります", "set:駅前のホテル"), ("案内を中止していただけますか", "cancel"),
     ("よろしければ美術館まで連れていってください", "set:美術館"), ("経由地に郵便局を追加していただけませんか", "way:郵便局"), ("渋滞を避けるルートにしていただけるとありがたいです", "reroute"),
     ("自宅へのルートを出していただけますか", "set:自宅")],
    [("新宿駅お願い", "set:新宿駅"), ("会社戻る", "set:会社"), ("病院まで", "set:病院"), ("案内おわり", "cancel"), ("郵便局よって", "set:郵便局|way:郵便局"), ("駅いって", "set:駅"), ("目的地、羽田空港", "set:羽田空港"),
     ("道変えて", "reroute")],
    [("買い物が終わったら自宅まで案内して", "set:自宅"), ("子どもを乗せたら塾へ向かって", "set:塾"), ("9時までに着きたいから会社へ向かって", "set:会社"), ("日が暮れる前に展望台まで案内して", "set:展望台"),
     ("雨が上がったら公園に向かって", "set:公園"), ("渋滞が終わるまで別のルートで案内して", "reroute"), ("お店が閉まる前にホームセンターへ向かって", "set:ホームセンター"), ("給油したら高速のインターまで案内して", "set:高速のインター")],
    [("県立美術館の第二駐車場まで案内して", "set:県立美術館の第二駐車場"), ("すき家に向かって", "set:すき家"), ("かっぱ寿司まで案内して", "set:かっぱ寿司"), ("びっくりドンキーへ行って", "set:びっくりドンキー"),
     ("国道1号線沿いのガソリンスタンドに向かって", "set:国道1号線沿いのガソリンスタンド"), ("いろは坂まで案内して", "set:いろは坂"), ("東口のロータリーへ向かって", "set:東口のロータリー"), ("3丁目の交差点まで案内して", "set:3丁目の交差点")],
    [("空港じゃなくて駅に向かって", "set:駅"), ("自宅はやめて実家に向かって", "set:実家"), ("下道じゃなくて高速で行って", "reroute"), ("目的地はそのままで経由地にコンビニを追加して", "way:コンビニ"),
     ("病院ではなく薬局を目的地にして", "set:薬局"), ("案内は止めないでルートだけ変えて", "reroute"), ("銀行に寄ってから市役所まで案内して", "set:市役所|way:銀行"), ("大阪駅から京都駅まで案内して", "set:京都駅")],
    [("東京駅に行きたいんだけど", "set:東京駅"), ("空港まで行ける", "set:空港"), ("病院に向かってもらえるかな", "set:病院"), ("コンビニに寄りたいな", "set:コンビニ|way:コンビニ"), ("案内をやめてもいいかな", "cancel"),
     ("ほかの道はないかな", "reroute"), ("会社まで連れてってくれない", "set:会社"), ("図書館に行きたい", "set:図書館")],
    [("えっと、子どもが熱を出したみたいなので、急いで市民病院まで案内してほしい", "set:市民病院"), ("あの、会議に間に合いそうにないから、できるだけ早く会社に向かってもらえるかな", "set:会社"),
     ("うーん、この道はずっと混んでるみたいだから、別のルートに変えてくれると助かる", "reroute"), ("ねえ、お腹がすいてきたから、途中でコンビニに寄るルートにしてくれる", "way:コンビニ"),
     ("そうだ、荷物を届けないといけないので、先に郵便局を目的地にしてほしいんだけど", "set:郵便局"), ("えー、もう道は覚えたと思うから、そろそろ案内は終わりにしてもらえる", "cancel"),
     ("あのさ、帰りに買い物をしたいから、経由地にスーパーを追加しておいて", "way:スーパー"), ("んー、やっぱり電車で行くことにしたので、駅まで送ってもらえるかな", "set:駅")],
]
ALL = (("control_hvac", HVAC), ("control_media", MEDIA), ("set_navigation", NAV))


def rows_all() -> list[dict]:
    rows = []
    for tool, forms in ALL:
        if len(forms) != len(FORMS):
            raise SystemExit(f"{tool}: 形が {len(forms)} 個（{len(FORMS)} 個のはず）")
        for form, items in zip(FORMS, forms):
            if len(items) != PER:
                raise SystemExit(f"{tool}・{form}: {len(items)} 行（{PER} 行のはず）")
            for u, label in items:
                rows.append({"id": f"a2_{tool.split('_')[-1][:5]}_{len(rows):03d}", "tool": tool, "form": form, "utterance": u, "label": label, "expected": parse(tool, label),
                             "history": [], "action": f"TOOL_{tool}", "purpose": "arg_eval", "author": "claude", "trainable": False})
    return rows


def build() -> None:
    rows = rows_all()
    problems = [f"この中で重なる: {u}" for u, k in collections.Counter(r["utterance"] for r in rows).items() if k > 1]
    old = {"学習": read_jsonl(TR3 / "utterances.jsonl"), "v1": read_jsonl(CH1), "v2": read_jsonl(CH2 / "utterances.jsonl"), "v3": read_jsonl(CH3 / "utterances.jsonl"),
           "v4": read_jsonl(CH4 / "utterances.jsonl"), "v5": read_jsonl(CH5 / "utterances.jsonl"), "v6": read_jsonl(CH6 / "utterances.jsonl"), "arg_eval_v1": read_jsonl(ARG1 / "utterances.jsonl")}
    worst = []
    for name, rs in old.items():
        us = sorted({r["utterance"] for r in rs})
        for r in rows:
            for x in us:
                s = 1.0 if r["utterance"] == x else near(r["utterance"], x)
                if s >= NEAR:
                    problems.append(f"{name} と{'同じ' if s == 1.0 else f'似すぎ（{s:.2f}）'}:「{r['utterance']}」↔「{x}」")
                worst.append((s, r["utterance"], x, name))
    worst.sort(reverse=True)
    print("これまでの行で一番似ている組: " + " / ".join(f"{s:.2f}「{a}」↔「{b}」（{n}）" for s, a, b, n in worst[:4]))
    if problems:
        raise SystemExit(f"止める（{len(problems)} 件）:\n  " + "\n  ".join(problems))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "utterances.jsonl").write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    print(f"arg_eval_v2: {len(rows)} 行（" + "・".join(f"{J[t]} {sum(r['tool'] == t for r in rows)}" for t in J) + f"）/ 形 {len(FORMS)} 個 × {PER} 行 / md5 {md5_of(OUT / 'utterances.jsonl')}")
    print(f"保存しました: {OUT / 'utterances.jsonl'}")


def measure() -> None:
    use_multi_library()
    from edge_router.dispatch import arguments
    from .p21_arguments import make_resolve as args_p21
    from .p22_arguments import make_resolve as args_p22
    from .p23_arguments import make_resolve as args_p23
    rows = read_jsonl(OUT / "utterances.jsonl")
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    here = PROJECT_ROOT / "app"
    sums = {"arg_eval_v2": md5_of(OUT / "utterances.jsonl"), "p23_arguments.py": md5_of(here / "p23_arguments.py"), "p22_arguments.py": md5_of(here / "p22_arguments.py")}
    print(f"\n===== P23 引数の値（arg_eval_v2・{len(rows)} 行。見ていない行）=====")
    print("測ったもの（md5）: " + "・".join(f"{k} {v}" for k, v in sums.items()))
    md = [f"arg_eval_v2: {len(rows)} 行（形 {len(FORMS)} 個 × 3 機能 × {PER} 行）。`p23_arguments` を決めた後に書いた、見ていない行。形 9・10 は、直しを作るときに見ていない形。引数の段だけを測る\n",
          "測ったもの（md5）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "\n",
          "結果は 3 つに分ける: **合っている**／**聞き返す**（違う実行はしない）／**違う値で実行**\n"]
    res = {"md5": sums, "configs": {}}
    configs = (("P21（直す前）", args_p21), ("P22（主節を先に見る）", args_p22), ("P23（整えてから渡す）", args_p23))
    summary = ["\n## まとめ（合っている／聞き返す／違う値で実行）\n", "|引数|形 1〜8（192 行）|形 9・10（48 行。見ていない形）|合計（240 行）|", "|---|---|---|---|"]
    body = []
    for cname, make in configs:
        resolve = make(arguments)
        cnt = collections.Counter()
        detail = []
        for r in rows:
            x = resolve(r["tool"], None, r["utterance"], state)
            ok = match(r["tool"], r["expected"], x.complete, x.values)
            kind = "合っている" if ok else "聞き返す" if not x.complete else "違う値で実行"
            for key in ("all", "新" if r["form"] in NEW_FORMS else "旧", r["tool"], r["form"], r["tool"] + "|" + r["form"]):
                cnt[(key, kind)] += 1
                cnt[(key, "n")] += 1
            detail.append({"id": r["id"], "tool": r["tool"], "form": r["form"], "utterance": r["utterance"], "label": r["label"],
                           "got": json.dumps(x.values, ensure_ascii=False) if x.complete else "聞き返す", "kind": kind})
        res["configs"][cname] = {"counts": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, "rows": detail}
        cellf = lambda key: f"{cnt[(key, '合っている')]}／{cnt[(key, '聞き返す')]}／{cnt[(key, '違う値で実行')]}"
        pct = lambda key: f"{cellf(key)}（合っている {cnt[(key, '合っている')] / cnt[(key, 'n')]:.3f}・違う値で実行 {cnt[(key, '違う値で実行')] / cnt[(key, 'n')]:.3f}）"
        summary.append(f"|{cname}|{pct('旧')}|{pct('新')}|{pct('all')}|")
        print(f"\n--- {cname} ---\n形 1〜8: {pct('旧')}\n形 9・10: {pct('新')}\n合計: {pct('all')}")
        body += [f"\n## {cname}\n", f"表の中は「合っている／聞き返す／違う値で実行」（{PER} 行ずつ）\n", "|形|" + "|".join(J.values()) + f"|合計（{PER * 3} 行）|", "|---|---|---|---|---|"]
        for form in FORMS:
            body.append(f"|{form}|" + "|".join(cellf(t + "|" + form) for t in J) + f"|{cellf(form)}|")
            print(f"{form:<16} " + "  ".join(f"{J[t]} {cellf(t + '|' + form)}" for t in J) + f"  合計 {cellf(form)}")
        body.append("|**合計**|" + "|".join(cellf(t) for t in J) + f"|{cellf('all')}|")
        if make is args_p23:
            for kind in ("違う値で実行", "聞き返す"):
                body += [f"\n### {kind}の行（P23）\n", "|機能|形|発話|取り出した値|正解|", "|---|---|---|---|---|"]
                body += [f"|{J[d['tool']]}|{d['form']}|{d['utterance']}|{d['got']}|{d['label']}|" for d in detail if d["kind"] == kind]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "argeval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "argeval.md").write_text("\n".join(md + summary + body) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（argeval.md・argeval.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "measure"])
    a = ap.parse_args()
    build() if a.step == "build" else measure()


if __name__ == "__main__":
    main()
