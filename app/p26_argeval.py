# -*- coding: utf-8 -*-
"""P26 手順 3: 引数の値だけを測る、4 つ目の新しい行（arg_eval_v4・240 行）。端末なし。

設計: DoC/設計書_P26_歯止めの2段目.md の 5 節。

arg_eval_v3 は、P25 の結果と歯止めの 2 段目（`p26_arguments`）を作るときに見たので「直す用」になった。
**2 段目を固定した後に**（md5 は `results/P26/preregistration.md`）、新しい 240 行を書いた。正解の引数は、文の意味だけから付けた（取り出しの結果を見る前）。
形は arg_eval_v2・v3 と同じ 10 個 × 3 機能 × 8 行。

    python -m app.p26_argeval build       # data/arg_eval_v4/utterances.jsonl（重なりを確かめる）
    python -m app.p26_argeval measure     # results/P26/argeval.md・argeval.json
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
from .p22_argeval import J, NEAR, OUT as ARG1
from .p22_labels import match, parse
from .p23_argeval import FORMS, OUT as ARG2
from .p25_argeval import OUT as ARG3

OUT = PROJECT_ROOT / "data" / "arg_eval_v4"
RES = PROJECT_ROOT / "results" / "P26"
PER = 8

HVAC = [
    [("冷房を23度にセットして", "t23|t23,m:cool"), ("風量を6段階にして", "f6"), ("暖房をオフにして", "m:off"), ("ガラスの曇り取りを始めて", "m:defrost"),
     ("温度をちょっと下げて", "t24"), ("暖房をもう少し強めて", "f4"), ("空調をおまかせにして", "m:auto"), ("冷房のスイッチを入れてください", "m:cool")],
    [("西日が入って暑いので冷房をつけて", "m:cool"), ("背中が寒いから温度を2度上げて", "t27"), ("息で窓が曇ったので曇り止めを入れて", "m:defrost"), ("喉が乾燥するから風を弱めて", "f2"),
     ("雪で冷え込んでいるので暖房を入れて", "m:heat"), ("ほこりっぽいので風量を上げて", "f4"), ("もうすぐ駐車するから空調を切って", "m:off"), ("子どもが暑がるので温度を23度にして", "t23")],
    [("暖房をつけて、凍えそう", "m:heat"), ("温度を3度下げて、蒸し風呂みたい", "t22"), ("風量を1にして、乾燥する", "f1"), ("窓の曇りを取って、外が見えない", "m:defrost"),
     ("冷房を弱めて、足が冷たい", "f2"), ("エアコンを止めて、静かにしたい", "m:off"), ("21度にして、暑くてたまらない", "t21"), ("風を強めて、空気がよどんでる", "f4")],
    [("ヒーターをつけていただけますか", "m:heat"), ("温度を1度下げてもらってもいいですか", "t24"), ("風量を4にしていただけると助かります", "f4"), ("エアコンを切っていただいて構いません", "m:off"),
     ("前のガラスの曇りを取っていただけますか", "m:defrost"), ("車内を少し冷やしていただけますか", "t24"), ("お手数ですが風を弱くしてください", "f2"), ("冷房を入れていただいてもいいでしょうか", "m:cool")],
    [("暖房いれて", "m:heat"), ("クーラーきって", "m:off"), ("風つよく", "f4"), ("さむっ、温度あげて", "t26"),
     ("22度でいい", "t22"), ("フロント曇り取り", "m:defrost"), ("冷房たのむ", "m:cool"), ("風ちょい弱めて", "f2")],
    [("渋滞を抜けたら冷房を弱めて", "f2"), ("ガソリンスタンドに着くまで暖房をつけておいて", "m:heat"), ("トンネルの中では風量を2にして", "f2"), ("目的地に着いたら空調を止めて", "m:off"),
     ("雨がやむまで曇り取りを入れておいて", "m:defrost"), ("夜になったら温度を24度にして", "t24"), ("高速を降りたら温度を上げて", "t26"), ("休憩中はエアコンをオートにしておいて", "m:auto")],
    [("温度を二十六度にして", "t26"), ("風量を七にして", "f7"), ("温度を4度上げて", "t29"), ("風量をみっつ下げて", "f0"),
     ("設定温度を最低まで下げて", "t16"), ("風を全開にして", "f7"), ("温度を二十二度半にして", "t22.5"), ("風量を二段上げて", "f5")],
    [("温度を上げて風は弱めて", "t26,f2"), ("ヒーターをやめてクーラーをつけて", "m:cool"), ("風量を5にして、温度は21度で", "f5,t21"), ("冷房は止めずに温度だけ上げて", "t26"),
     ("風量はそのままで温度を下げて", "t24"), ("曇り取りを止めて冷房を入れて", "m:cool"), ("温度じゃなくて風を弱めて", "f2"), ("エアコンはつけたまま風だけ弱くして", "f2")],
    [("冷房入れられる", "m:cool"), ("暖房を強くしてもいいかな", "f4"), ("車内をもっと暖かくできる", "t26"), ("風量を下げたいんだけど", "f2"),
     ("温度を25度にできる", "t25"), ("曇り止めをかけたいな", "m:defrost"), ("エアコンを消したいな", "m:off"), ("冷房を弱くできないかな", "f2")],
    [("えっと、さっき外から戻ったばかりで体が冷えてるから、暖房を入れてもらえるかな", "m:heat"), ("あのさ、日焼けしそうなくらい日差しが強いから、冷房をつけてくれる", "m:cool"),
     ("うーん、湿気でガラスが曇ってきて危ないから、曇り取りを入れてほしいんだけど", "m:defrost"), ("そうだな、ちょっと風が直接当たって寒いから、風量を2にしてもらえる", "f2"),
     ("えー、窓を開けて走りたいから、エアコンは止めてもらっていいかな", "m:off"), ("あの、もう少しだけ涼しいほうがいいので、温度を24度にしてもらえると助かる", "t24"),
     ("ねえ、後ろの席の人が寒いって言ってるから、温度を1度上げてくれる", "t26"), ("んー、空気がこもってる感じがするので、風を少し強めにしてほしいな", "f4")],
]
MEDIA = [
    [("曲を流して", "play"), ("音楽を一時停止して", "pause"), ("次の曲へ進んで", "next"), ("さっきの曲に戻してください", "prev"),
     ("音量を13にして", "v13"), ("ボリュームを少し上げて", "v14"), ("音量を少し落として", "v10"), ("音を無音にして", "mute")],
    [("運転に飽きてきたので曲をかけて", "play"), ("トンネルで電話が入ったので曲を止めて", "pause"), ("この曲は好みじゃないから飛ばして", "next"), ("歌詞を聞き逃したのでもう一つ前に戻して", "prev"),
     ("エンジン音で聞こえないから音を上げて", "v14"), ("会話がしにくいから音量を下げて", "v10"), ("駐車場の精算をするので消音にして", "mute"), ("好きな曲だから音量を24にして", "v24")],
    [("曲を止めて、電話に出る", "pause"), ("次の曲にして、飽きた", "next"), ("音を下げて、頭が痛い", "v10"), ("音量を上げて、窓を開けたから", "v14"),
     ("音楽を流して、静かすぎる", "play"), ("一つ前に戻して、いい曲だった", "prev"), ("音量を8にして、子どもが寝た", "v8"), ("消音にして、話を聞きたい", "mute")],
    [("何か曲を流していただけますか", "play"), ("音楽を一時停止していただけますか", "pause"), ("次の曲に飛ばしていただけますでしょうか", "next"), ("ボリュームを上げていただけると助かります", "v14"),
     ("音量を下げていただけませんでしょうか", "v10"), ("前の曲をもう一度流していただけますか", "prev|play"), ("音量を15にしていただけると嬉しいです", "v15"), ("音を消していただけますか", "mute|pause")],
    [("なんか流してよ", "play"), ("音楽とめて", "pause"), ("次いこ", "next"), ("音でかく", "v14"),
     ("ボリュームさげて", "v10"), ("音量20で", "v20"), ("さっきの曲もう一回", "prev|play"), ("ミュートね", "mute")],
    [("車が動き出したら音楽をかけて", "play"), ("コンビニに寄る間は曲を止めておいて", "pause"), ("サビが終わったら次の曲にして", "next"), ("トンネルを出たら音量を上げて", "v14"),
     ("料金所を通る間は音を小さくして", "v10"), ("電話の間は消音にしておいて", "mute"), ("駐車場に着いたら音楽を止めて", "pause"), ("朝のうちは音量を10にして", "v10")],
    [("ボリュームを二十二にして", "v22"), ("音を4つ下げて", "v8"), ("ボリュームを一番大きくして", "v40"), ("音の大きさを二倍にして", "v24"),
     ("音量を6だけ下げて", "v6"), ("音量をいちばん下まで絞って", "v0"), ("ボリュームをほんの少し上げて", "v14"), ("音量を30にして", "v30")],
    [("消音じゃなくて音を小さくして", "v10"), ("音量は変えないで次の曲にして", "next"), ("曲は止めずに音量を14にして", "v14"), ("ラジオはやめて曲を流して", "play"),
     ("飛ばすんじゃなくて一曲戻して", "prev"), ("音を上げるんじゃなくて下げて", "v10"), ("一時停止じゃなくて次の曲に飛ばして", "next"), ("止めないで音を下げて", "v10")],
    [("曲かけられる", "play"), ("音を大きくしたいな", "v14"), ("次の曲に行きたいんだけど", "next"), ("音楽を止めたいな", "pause"),
     ("音量を16にできるかな", "v16"), ("もうちょっと静かにならないかな", "v10"), ("前の曲をもう一回聞けるかな", "prev"), ("ラジオをつけられる", "play")],
    [("えっと、長い運転で眠くなってきたから、元気な曲を何か流してもらえるかな", "play"), ("あの、いまから大事な電話をするので、少しの間曲を止めておいてもらえる", "pause"),
     ("うーん、この曲はさっきも聞いたから、次の曲に進めてもらえると助かる", "next"), ("ねえ、トラックの音がうるさくて聞こえないから、ボリュームをもう少し上げて", "v14"),
     ("そうだ、さっき流れてた曲をもう一度聞きたいから、前に戻してくれる", "prev"), ("えー、助手席の人が寝ちゃったので、音量を4まで下げてもらえるかな", "v4"),
     ("あのさ、料金所で係の人と話すから、ちょっと音を消しておいて", "mute|pause"), ("んー、ちょっと音が割れてる気がするから、音量を少し下げてほしいな", "v10")],
]
NAV = [
    [("福岡空港まで案内して", "set:福岡空港"), ("おじいちゃんの家までナビして", "set:おじいちゃんの家"), ("市立図書館を目的地にして", "set:市立図書館"), ("道案内を終えてください", "cancel"),
     ("違う経路を出してください", "reroute"), ("ファミマを経由地に追加して", "way:ファミマ"), ("神戸駅まで連れて行って", "set:神戸駅"), ("動物病院に向かって", "set:動物病院")],
    [("お腹がすいたのでラーメン屋に向かって", "set:ラーメン屋"), ("タイヤの点検をしたいからディーラーまで案内して", "set:ディーラー"), ("この道は狭いから別のルートにして", "reroute"),
     ("もう着くので案内を終わりにして", "cancel"), ("お茶が欲しいのでセブンに立ち寄って", "way:セブン|set:セブン"), ("書類を出すので区役所を目的地にして", "set:区役所"),
     ("財布を忘れたので家に戻って", "set:家"), ("打ち合わせがあるから支店まで案内して", "set:支店")],
    [("空港まで案内して、飛行機に乗るんだ", "set:空港"), ("案内を中止して、道はわかる", "cancel"), ("ルートを変えて、渋滞してるみたい", "reroute"), ("歯医者に向かって、予約があるんだ", "set:歯医者"),
     ("スタンドに寄って、ガソリンが少ない", "way:スタンド|set:スタンド"), ("駅を目的地にして、友達を迎えに行く", "set:駅"), ("自宅まで案内して、もう帰る", "set:自宅"),
     ("下道のルートにして、高速代を節約したい", "reroute")],
    [("恐れ入りますが熊本城までお願いできますか", "set:熊本城"), ("ナビを終了していただけますか", "cancel"), ("別の道を探していただけると助かります", "reroute"), ("市民病院を目的地にしていただけますか", "set:市民病院"),
     ("途中で郵便局に寄っていただけますか", "way:郵便局"), ("会社まで案内をお願いしてもよろしいですか", "set:会社"), ("混んでいない道にしていただけますか", "reroute"), ("科学館への道を教えていただけますか", "set:科学館")],
    [("空港まで頼むわ", "set:空港"), ("家帰りたい", "set:家"), ("ナビおわり", "cancel"), ("別ルートで", "reroute"),
     ("駅前ロータリーまで頼むね", "set:駅前ロータリー"), ("スタバ寄りたい", "way:スタバ|set:スタバ"), ("行き先、大阪駅", "set:大阪駅"), ("病院いって", "set:病院")],
    [("インターを降りたらコンビニに寄って", "way:コンビニ|set:コンビニ"), ("駐車場に入ったら案内を止めて", "cancel"), ("事故渋滞ならほかの道にして", "reroute"), ("夕方になったらスーパーに向かって", "set:スーパー"),
     ("帰りは実家まで案内して", "set:実家"), ("雪が降ってきたら高速を使わないルートにして", "reroute"), ("会社に着く前に銀行に寄って", "way:銀行"), ("明日の朝は駅に向かって", "set:駅")],
    [("国道2号沿いのガソリンスタンドに向かって", "set:国道2号沿いのガソリンスタンド"), ("3番ゲートの駐車場まで案内して", "set:3番ゲートの駐車場"), ("イオンの立体駐車場の2階へ向かって", "set:イオンの立体駐車場の2階"),
     ("スシローに向かって", "set:スシロー"), ("4丁目の郵便局まで案内して", "set:4丁目の郵便局"), ("南口のタクシー乗り場へ行って", "set:南口のタクシー乗り場"), ("コストコの第2駐車場まで案内して", "set:コストコの第2駐車場"),
     ("あべのハルカスまで案内して", "set:あべのハルカス")],
    [("駅ではなく港に向かって", "set:港"), ("実家はやめて会社に向かって", "set:会社"), ("高速ではなく一般道で行って", "reroute"), ("目的地はそのままでスーパーを経由地にして", "way:スーパー"),
     ("病院じゃなくて自宅に戻って", "set:自宅"), ("ナビは続けて道だけ変えて", "reroute"), ("銀行に寄ってから空港に向かって", "set:空港|way:銀行"), ("東京から横浜まで案内して", "set:横浜")],
    [("札幌駅に行きたいな", "set:札幌駅"), ("港まで行けるかな", "set:港"), ("病院まで案内してもらえる", "set:病院"), ("コンビニに寄れるかな", "way:コンビニ|set:コンビニ"),
     ("ナビを止めてもいい", "cancel"), ("もっと空いてる道はないかな", "reroute"), ("実家まで送ってくれない", "set:実家"), ("美術館に行きたいんだけど", "set:美術館")],
    [("えっと、映画の時間に間に合わせたいので、駅前の映画館まで案内してもらえるかな", "set:駅前の映画館"), ("あの、ここからは歩いて行けるから、案内はもう終わりにしてもらって大丈夫", "cancel"),
     ("うーん、この先が通行止めみたいだから、ほかの道を探してくれると助かる", "reroute"), ("ねえ、電気が残り少ないから、途中で充電スタンドに寄ってくれる", "way:充電スタンド"),
     ("そうだ、明日の朝ごはんを買いたいので、帰りにパン屋に寄ってほしいんだけど", "way:パン屋|set:パン屋"), ("えー、習い事のお迎えがあるから、スイミングスクールまで案内してくれる", "set:スイミングスクール"),
     ("あのさ、今日はもう疲れたので、自宅まで一番早いルートにしてほしいな", "set:自宅"), ("んー、少し時間が余ってるから、川沿いの公園に向かってもらえるかな", "set:川沿いの公園")],
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
                rows.append({"id": f"a4_{tool.split('_')[-1][:5]}_{len(rows):03d}", "tool": tool, "form": form, "utterance": u, "label": label, "expected": parse(tool, label),
                             "history": [], "action": f"TOOL_{tool}", "purpose": "arg_eval", "author": "claude", "trainable": False})
    return rows


def build() -> None:
    rows = rows_all()
    problems = [f"この中で重なる: {u}" for u, k in collections.Counter(r["utterance"] for r in rows).items() if k > 1]
    old = {"学習": read_jsonl(TR3 / "utterances.jsonl"), "v1": read_jsonl(CH1), "v2": read_jsonl(CH2 / "utterances.jsonl"), "v3": read_jsonl(CH3 / "utterances.jsonl"),
           "v4": read_jsonl(CH4 / "utterances.jsonl"), "v5": read_jsonl(CH5 / "utterances.jsonl"), "v6": read_jsonl(CH6 / "utterances.jsonl"),
           "arg_eval_v1": read_jsonl(ARG1 / "utterances.jsonl"), "arg_eval_v2": read_jsonl(ARG2 / "utterances.jsonl"),
           "arg_eval_v3": read_jsonl(ARG3 / "utterances.jsonl")}
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
    print(f"arg_eval_v4: {len(rows)} 行（" + "・".join(f"{J[t]} {sum(r['tool'] == t for r in rows)}" for t in J) + f"）/ 形 {len(FORMS)} 個 × {PER} 行 / md5 {md5_of(OUT / 'utterances.jsonl')}")
    print(f"保存しました: {OUT / 'utterances.jsonl'}")


def measure() -> None:
    use_multi_library()
    from edge_router.dispatch import arguments
    from .p21_arguments import make_resolve as args_p21
    from .p23_arguments import make_resolve as args_p23
    from .p25_arguments import make_resolve as args_p25
    from .p26_arguments import make_resolve as args_p26
    rows = read_jsonl(OUT / "utterances.jsonl")
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    here = PROJECT_ROOT / "app"
    sums = {"arg_eval_v4": md5_of(OUT / "utterances.jsonl"), "p26_arguments.py": md5_of(here / "p26_arguments.py"), "p25_arguments.py": md5_of(here / "p25_arguments.py"),
            "p23_arguments.py": md5_of(here / "p23_arguments.py")}
    print(f"\n===== P26 引数の値（arg_eval_v4・{len(rows)} 行。見ていない行）=====")
    print("測ったもの（md5）: " + "・".join(f"{k} {v}" for k, v in sums.items()))
    md = [f"arg_eval_v4: {len(rows)} 行（形 {len(FORMS)} 個 × 3 機能 × {PER} 行）。歯止めの 2 段目（`p26_arguments`）を固定した後に書いた、見ていない行。引数の段だけを測る\n",
          "測ったもの（md5）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "（固定した md5 は `results/P26/preregistration.md`）\n",
          "結果は 3 つに分ける: **合っている**／**聞き返す**（違う実行はしない）／**違う値で実行**\n"]
    res = {"md5": sums, "configs": {}}
    configs = (("P21（直す前）", args_p21), ("P23（いまの引数）", args_p23), ("P25（歯止め 1 段目）", args_p25), ("P26（歯止め 2 段目）", args_p26))
    summary = ["\n## まとめ（合っている／聞き返す／違う値で実行）\n", "|引数|空調（80 行）|メディア（80 行）|ナビ（80 行）|合計（240 行）|", "|---|---|---|---|---|"]
    body, details = [], {}
    for cname, make in configs:
        resolve = make(arguments)
        cnt = collections.Counter()
        detail = []
        for r in rows:
            x = resolve(r["tool"], None, r["utterance"], state)
            ok = match(r["tool"], r["expected"], x.complete, x.values)
            kind = "合っている" if ok else "聞き返す" if not x.complete else "違う値で実行"
            for key in ("all", r["tool"], r["form"], r["tool"] + "|" + r["form"]):
                cnt[(key, kind)] += 1
                cnt[(key, "n")] += 1
            detail.append({"id": r["id"], "tool": r["tool"], "form": r["form"], "utterance": r["utterance"], "label": r["label"],
                           "got": json.dumps(x.values, ensure_ascii=False) if x.complete else "聞き返す", "kind": kind, "note": x.notes[0] if x.notes else ""})
        details[cname] = detail
        res["configs"][cname] = {"counts": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, "rows": detail}
        cellf = lambda key: f"{cnt[(key, '合っている')]}／{cnt[(key, '聞き返す')]}／{cnt[(key, '違う値で実行')]}"
        pct = lambda key: f"{cellf(key)}（違う値で実行 {cnt[(key, '違う値で実行')] / cnt[(key, 'n')]:.3f}）"
        summary.append(f"|{cname}|" + "|".join(pct(t) for t in J) + f"|{pct('all')}|")
        print(f"\n--- {cname} ---\n" + "  ".join(f"{J[t]} {pct(t)}" for t in J) + f"\n合計: {pct('all')}")
        body += [f"\n## {cname}\n", f"表の中は「合っている／聞き返す／違う値で実行」（{PER} 行ずつ）\n", "|形|" + "|".join(J.values()) + f"|合計（{PER * 3} 行）|", "|---|---|---|---|---|"]
        for form in FORMS:
            body.append(f"|{form}|" + "|".join(cellf(t + "|" + form) for t in J) + f"|{cellf(form)}|")
        body.append("|**合計**|" + "|".join(cellf(t) for t in J) + f"|{cellf('all')}|")
    # P23 → P25 で変わった行（歯止めが止めた行）と、P25 でも残った違う値での実行
    body += ["\n## P23 → P26 で、どの段で止めたか\n", "|段|止めた行（P23 で違う値で実行だった／合っていた）|", "|---|---|"]
    p23d, p25d, p26d = details[configs[1][0]], details[configs[2][0]], details[configs[3][0]]
    for name, prev, cur in (("1 段目（P23 → P25）", p23d, p25d), ("2 段目（P25 → P26）", p25d, p26d)):
        bad = sum(1 for x, y in zip(prev, cur) if x["kind"] == "違う値で実行" and y["kind"] == "聞き返す")
        ok = sum(1 for x, y in zip(prev, cur) if x["kind"] == "合っている" and y["kind"] == "聞き返す")
        body.append(f"|{name}|{bad}／{ok}|")
    a, b = p25d, p26d
    body += ["\n## P25 → P26 で変わった行（2 段目が止めた行）\n", "|機能|形|発話|P25 の値|P25 の採点|正解|止めた理由|", "|---|---|---|---|---|---|---|"]
    changed = collections.Counter()
    for x, y in zip(a, b):
        if x["kind"] != y["kind"]:
            changed[(x["kind"], y["kind"])] += 1
            body.append(f"|{J[x['tool']]}|{x['form']}|{x['utterance']}|{x['got']}|{x['kind']}|{x['label']}|{y['note'].replace('P26: ', '').replace('P25: ', '').replace(' → 聞き返す', '')}|")
    body.append("\n変わり方（P25 → P26）: " + ("・".join(f"{k[0]} → {k[1]} {v} 行" for k, v in changed.most_common()) if changed else "なし"))
    print("\nP25 → P26 で変わった行: " + ("・".join(f"{k[0]} → {k[1]} {v} 行" for k, v in changed.most_common()) if changed else "なし"))
    for kind in ("違う値で実行", "聞き返す"):
        body += [f"\n## P26 で{kind}の行\n", "|機能|形|発話|取り出した値|正解|メモ|", "|---|---|---|---|---|---|"]
        body += [f"|{J[d['tool']]}|{d['form']}|{d['utterance']}|{d['got']}|{d['label']}|{d['note']}|" for d in b if d["kind"] == kind]
    res["changed"] = {f"{k[0]}→{k[1]}": v for k, v in changed.items()}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "argeval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "argeval.md").write_text("\n".join(md + summary + body) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（argeval.md・argeval.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "measure"])
    a = ap.parse_args()
    {"build": build, "measure": measure}[a.step]()


if __name__ == "__main__":
    main()
