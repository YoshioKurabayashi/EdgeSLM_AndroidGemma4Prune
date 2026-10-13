# -*- coding: utf-8 -*-
"""P22 手順 3: 引数の値だけを測る、新しい行（arg_eval_v1・240 行）。**形の違う文**を、形ごとに 10 行ずつ入れた。端末なし。

設計: DoC/設計書_P22_引数の値を測って直す.md の 6 節。

`p22_arguments` を決めた後に書いた。正解の引数は、文の意味だけから付けた（取り出しの結果を見る前）。学習にも、規則を作るのにも使っていない。
引数の段だけを測る（正しいツールを選べたとして、値が合うか）。流れ全体（BERT・ヘッド・規則）は通さないので、端末は要らない。

|形|中身|例|
|---|---|---|
|1 ふつう|短い依頼|横浜駅まで案内して|
|2 理由が前|〜だから・ので・んで・ため・し|熱が出たから病院まで案内して|
|3 理由が後ろ|依頼を先に言い、理由を後に足す|病院に向かって、子どもが熱を出した|
|4 ていねい・遠回し|〜していただけますか・〜してもらえると助かります|冷房を入れていただけますか|
|5 くだけた|助詞を落とす・一言・方言|東京駅お願い／音でかい、下げて|
|6 条件・時つき|〜まで・〜あいだ・〜たら・〜前に|着くまで冷房を強くしておいて|
|7 数・程度・場所の言い方|漢数字・半分・倍・一段／ひらがなの店名・「の」でつなぐ場所|音量を倍にして／はま寿司に向かって|
|8 2 つのことを言う|A じゃなくて B・A して B|駅じゃなくて空港に向かって|

    python -m app.p22_argeval build       # data/arg_eval_v1/utterances.jsonl（重なりを確かめる）
    python -m app.p22_argeval measure     # results/P22/argeval.md・argeval.json
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
from .p22_labels import match, parse

OUT = PROJECT_ROOT / "data" / "arg_eval_v1"
RES = PROJECT_ROOT / "results" / "P22"
FORMS = ["1 ふつう", "2 理由が前", "3 理由が後ろ", "4 ていねい・遠回し", "5 くだけた", "6 条件・時つき", "7 数・程度・場所の言い方", "8 2 つのことを言う"]
J = {"control_hvac": "空調", "control_media": "メディア", "set_navigation": "ナビ"}
NEAR = 0.75          # 学習・これまでの評価セットの文と、これ以上似ていたら止める

HVAC = [
    [("エアコンを26度にして", "t26"), ("風量を2にして", "f2"), ("暖房のスイッチを入れて", "m:heat"), ("冷房はもう止めて", "m:off"), ("ガラスの曇りを取って", "m:defrost"),
     ("温度を上げて", "t26"), ("風を弱めて", "f2"), ("エアコンを消して", "m:off"), ("設定温度を20度に", "t20"), ("送風を最大に", "f7")],
    [("日が差してきて暑いから温度を2度下げて", "t23"), ("足元が冷えるんで暖房入れて", "m:heat"), ("フロントガラスが真っ白で見えないため曇り取りを作動させて", "m:defrost"),
     ("子どもが寝たし風を弱くして", "f2"), ("汗が引かないから冷房をもっと強くして", "f4"), ("鼻が乾くんでエアコン切って", "m:off"), ("外が涼しくなったので冷房を止めてください", "m:off"),
     ("雨で窓が曇るから曇り止めお願い", "m:defrost"), ("手がかじかむから温度を27度にして", "t27"), ("風が顔に当たって寒いから風量を1にして", "f1")],
    [("冷房を入れて、蒸し暑いから", "m:cool"), ("温度を1度上げて、ちょっと寒いので", "t26"), ("暖房を強めてほしい、足が冷たくて", "f4"), ("風を止めて、目が乾くから", "m:off|f0"),
     ("デフロスターをつけて、前が見えない", "m:defrost"), ("エアコンを切って、もう着くから", "m:off"), ("風量を上げて、後ろまで届いてないから", "f4"), ("温度を下げてくれる、暑がりなんだ", "t24"),
     ("暖房にして、朝は冷えるね", "m:heat"), ("24度にして、それくらいがちょうどいい", "t24")],
    [("冷房を入れていただけますか", "m:cool"), ("少し暖かくしていただけると助かります", "t26"), ("風をもう少し弱くしてもらってもいいですか", "f2"), ("温度を23度に設定していただけますでしょうか", "t23"),
     ("暖房をつけてもらえるとうれしいな", "m:heat"), ("エアコンを止めてもらってかまいません", "m:off"), ("曇りを取っていただきたいのですが", "m:defrost"), ("できれば風量を上げてほしいです", "f4"),
     ("もしよければ温度を下げてもらえませんか", "t24"), ("冷房を少し弱めてもらえるとありがたい", "f2")],
    [("冷房つけて", "m:cool"), ("暖房たのむ", "m:heat"), ("エアコン止めて", "m:off"), ("寒っ、温度上げて", "t26"), ("あっつ、冷房強くして", "f4"), ("風よわめで", "f2"), ("22度で", "t22"),
     ("曇り取ってー", "m:defrost"), ("エアコンいらんわ、切って", "m:off"), ("もうちょい涼しくして", "t24")],
    [("着くまで冷房を強くしておいて", "f4"), ("雨が止むまでデフロスターをつけておいて", "m:defrost"), ("子どもが寝てるあいだは風を弱くしておいて", "f2"), ("トンネルを抜けるまで風を止めておいて", "m:off|f0"),
     ("窓を閉めたら冷房を入れて", "m:cool"), ("エンジンが温まったら暖房を入れて", "m:heat"), ("走り出す前に曇りを取っておいて", "m:defrost"), ("みんなが乗ったら温度を24度にして", "t24"),
     ("休憩が終わったからエアコンをつけ直して", "m:auto"), ("高速に乗ってるうちに温度を1度下げておいて", "t24")],
    [("温度を二十二度にして", "t22"), ("温度を3度ほど下げて", "t22"), ("風の量を半分に落として", "f1|f2"), ("風量を一段階上げて", "f4"), ("風量を2段下げて", "f1"), ("温度をちょっとだけ上げて", "t26"),
     ("冷房を一気に強くして", "f4|f7"), ("温度を0.5度下げて", "t24.5"), ("風を最強にして", "f7"), ("温度を一番低くして", "t16")],
    [("温度を下げて風も強くして", "t24,f4"), ("冷房じゃなくて暖房にして", "m:heat"), ("暖房を切って窓の曇りを取って", "m:defrost"), ("温度は22度で風量は4にして", "t22,f4"), ("風を弱めて温度を1度上げて", "f2,t26"),
     ("オートはやめて冷房にして", "m:cool"), ("暖房を入れて25度にして", "t25,m:heat|t25"), ("風量は2でいいから温度を下げて", "f2,t24|t24"), ("温度じゃなくて風量を上げて", "f4"), ("冷房を弱めて風は止めないで", "f2")],
]
MEDIA = [
    [("何か曲をお願い", "play"), ("流れてる曲を止めて", "pause"), ("次へ", "next"), ("一曲前に戻して", "prev"), ("音量を18にして", "v18"), ("音を下げて", "v10"), ("消音モードにして", "mute"), ("曲をスタートさせて", "play"),
     ("ボリュームを大きくして", "v14"), ("いったんポーズして", "pause")],
    [("会話が聞こえないから音を小さくして", "v10"), ("眠くなってきたんで音楽かけて", "play"), ("着信があったので音楽を止めて", "pause"), ("この曲は飽きたから次にして", "next"),
     ("いまのサビをもう一回聞きたいから最初から流して", "prev|play"), ("赤ちゃんが起きるし音量を下げて", "v10"), ("高速でうるさいため音量を20に上げて", "v20"), ("さっきの曲が良かったから前の曲に戻って", "prev"),
     ("ニュースが聞きたいんでラジオにして", "play"), ("耳が痛いから音量を8にして", "v8")],
    [("音を上げて、よく聞こえない", "v14"), ("音楽を止めて、電話するから", "pause"), ("次の曲にして、これ好きじゃない", "next"), ("音量を下げてくれる、ちょっとうるさい", "v10"), ("何かかけて、静かすぎる", "play"),
     ("ミュートして、インターホンで話すから", "mute"), ("ひとつ戻して、今のもう一回聞きたい", "prev"), ("再生を止めて、もう降りるから", "pause"), ("音を消して、子どもが寝たから", "mute|pause"),
     ("ボリュームを15にして、それくらいがいい", "v15")],
    [("音楽をかけていただけますか", "play"), ("音量を少し下げていただけると助かります", "v10"), ("次の曲に進めてもらえますか", "next"), ("再生を止めていただいてもいいですか", "pause"),
     ("前の曲に戻していただけますでしょうか", "prev"), ("音量を12に合わせてもらえるとありがたいです", "v12"), ("消音にしていただけますか", "mute"), ("もしよければ音を少し大きくしてもらえませんか", "v14"),
     ("ラジオを流していただきたいのですが", "play"), ("できれば曲を変えてほしいです", "next|play")],
    [("なんか曲かけて", "play"), ("でかいな、音下げて", "v10"), ("これ飛ばして", "next"), ("ストップ", "pause"), ("音ちっちゃい、上げて", "v14"), ("ボリューム20", "v20"), ("次いって", "next"), ("ちょっと止めといて", "pause"),
     ("音ミュート", "mute"), ("もっかい最初から", "prev|play")],
    [("着くまで音楽を流しておいて", "play"), ("電話が終わるまで音を消しておいて", "mute|pause"), ("トンネルのあいだは音量を上げておいて", "v14"), ("信号で止まってるうちに次の曲にして", "next"),
     ("ナビがしゃべってるあいだは音楽を止めて", "pause"), ("休憩に入る前に音楽を止めて", "pause"), ("発車する前に音量を10にして", "v10"), ("高速を降りるまでラジオをかけといて", "play"),
     ("寝てる子が起きるまで音は小さくしておいて", "v10"), ("会議の電話のあいだはミュートにしといて", "mute")],
    [("音量をいまの半分にして", "v6"), ("音量を二十にして", "v20"), ("音を3つ下げて", "v9"), ("音量を倍にして", "v24"), ("ボリュームを最大にして", "v40"), ("音をちょっとだけ上げて", "v14"), ("音量を1だけ上げて", "v13"),
     ("音を最小にして", "v0"), ("ボリュームを5下げて", "v7"), ("音量を一気に上げて", "v14|v40")],
    [("この曲は止めて次の曲をかけて", "next"), ("ラジオじゃなくてスマホの曲をかけて", "play"), ("音量は上げないで下げて", "v10"), ("ミュートじゃなくて音を小さくして", "v10"), ("止めるんじゃなくて音を下げて", "v10"),
     ("次じゃなくて前の曲にして", "prev"), ("音楽はやめてラジオにして", "play"), ("前の曲じゃなくて次の曲にして", "next"), ("音は消さないで小さくして", "v10"), ("再生を止めないで音量だけ下げて", "v10")],
]
NAV = [
    [("横浜駅まで案内して", "set:横浜駅"), ("羽田空港に向かって", "set:羽田空港"), ("目的地を市役所にして", "set:市役所"), ("案内を中止して", "cancel"), ("図書館へ行って", "set:図書館"), ("ルートを変えて", "reroute"),
     ("コンビニを経由地に追加して", "way:コンビニ"), ("自宅に帰りたい", "set:自宅"), ("大阪城をナビして", "set:大阪城"), ("ナビはもう止めていいよ", "cancel")],
    [("熱が出たから病院まで案内して", "set:病院"), ("お腹がすいたんでファミレスに向かって", "set:ファミレス"), ("忘れ物をしたため自宅へ戻って", "set:自宅"), ("ガソリンが少ないしガソリンスタンドまで案内して", "set:ガソリンスタンド"),
     ("子どもが熱を出したので小児科を目的地にして", "set:小児科"), ("会議に遅れそうだから会社まで急いで", "set:会社"), ("雨が降ってきたんで駅まで迎えに行って", "set:駅"), ("荷物が重いからスーパーの駐車場に向かって", "set:スーパーの駐車場"),
     ("お土産を買いたいから道の駅に寄るルートにして", "way:道の駅"), ("事故で混んでるから別の道にして", "reroute")],
    [("病院に向かって、子どもが熱を出した", "set:病院"), ("空港まで案内して、飛行機に遅れそう", "set:空港"), ("郵便局を目的地にして、荷物を出したいから", "set:郵便局"), ("コンビニに寄るルートにして、トイレに行きたい", "way:コンビニ"),
     ("案内をやめて、もう道はわかるから", "cancel"), ("ルートを引き直して、渋滞してるみたい", "reroute"), ("実家へ向かって、母に呼ばれたんだ", "set:実家"), ("銀行まで連れていって、三時に閉まるから", "set:銀行"),
     ("経由地に薬局を入れて、薬を取りに行くから", "way:薬局"), ("高速は使わないで、お金がかかるから", "reroute")],
    [("東京タワーまで案内していただけますか", "set:東京タワー"), ("博物館に向かっていただけると助かります", "set:博物館"), ("目的地を空港に変更していただけますでしょうか", "set:空港"), ("案内を終了していただいてかまいません", "cancel"),
     ("よろしければ駅まで連れていってもらえませんか", "set:駅"), ("経由地にガソリンスタンドを加えていただけますか", "way:ガソリンスタンド"), ("別のルートを探していただけるとありがたいです", "reroute"),
     ("市民ホールへのルートを出してもらえますか", "set:市民ホール"), ("できれば下道で行っていただきたいのですが", "reroute"), ("ホテルを目的地に設定してもらってもいいですか", "set:ホテル")],
    [("東京駅お願い", "set:東京駅"), ("家帰る", "set:家"), ("会社まで", "set:会社"), ("ナビやめて", "cancel"), ("コンビニ寄って", "set:コンビニ|way:コンビニ"), ("空港いって", "set:空港"), ("次、区役所な", "set:区役所"),
     ("行き先、新宿駅", "set:新宿駅"), ("ルート変えてくれへん", "reroute"), ("駅までよろしく", "set:駅")],
    [("給油が終わったら空港まで案内して", "set:空港"), ("荷物を積んだら倉庫へ向かって", "set:倉庫"), ("5時までに着くように市役所まで案内して", "set:市役所"), ("暗くなる前にキャンプ場へ向かって", "set:キャンプ場"),
     ("休憩が済んだら案内を再開して", "reroute"), ("渋滞が解けるまで下道で案内して", "reroute"), ("着く前にコンビニに寄るルートにして", "way:コンビニ"), ("帰りは自宅までナビして", "set:自宅"),
     ("明日の朝までに着きたいから空港へ急いで", "set:空港"), ("お昼を食べたら動物園に向かって", "set:動物園")],
    [("国道16号沿いのホームセンターまで案内して", "set:国道16号沿いのホームセンター"), ("北海道庁まで案内して", "set:北海道庁"), ("駅の北口へ向かって", "set:駅の北口"), ("東京都庁の第一庁舎に向かって", "set:東京都庁の第一庁舎"),
     ("イオンモールの立体駐車場まで案内して", "set:イオンモールの立体駐車場"), ("はま寿司に向かって", "set:はま寿司"), ("ららぽーとまで案内して", "set:ららぽーと"), ("おでん屋まで連れてって", "set:おでん屋"),
     ("道の駅を行き先にして", "set:道の駅"), ("さくら通りの郵便局へ行って", "set:さくら通りの郵便局")],
    [("駅じゃなくて空港に向かって", "set:空港"), ("会社はやめて自宅に向かって", "set:自宅"), ("コンビニに寄ってから会社まで案内して", "set:会社|way:コンビニ"), ("高速じゃなくて下道で行って", "reroute"),
     ("目的地は変えずにルートだけ変えて", "reroute"), ("病院の前に薬局に寄るルートにして", "way:薬局"), ("図書館ではなく市役所を目的地にして", "set:市役所"), ("案内は続けて経由地だけ消して", "ASK|reroute"),
     ("空港へ行く途中でガソリンスタンドに寄って", "way:ガソリンスタンド"), ("東京駅から新宿駅まで案内して", "set:新宿駅")],
]
ALL = (("control_hvac", HVAC), ("control_media", MEDIA), ("set_navigation", NAV))


def rows_all() -> list[dict]:
    rows = []
    for tool, forms in ALL:
        if len(forms) != len(FORMS):
            raise SystemExit(f"{tool}: 形が {len(forms)} 個（{len(FORMS)} 個のはず）")
        for form, items in zip(FORMS, forms):
            if len(items) != 10:
                raise SystemExit(f"{tool}・{form}: {len(items)} 行（10 行のはず）")
            for u, label in items:
                rows.append({"id": f"a1_{tool.split('_')[-1][:5]}_{len(rows):03d}", "tool": tool, "form": form, "utterance": u, "label": label, "expected": parse(tool, label),
                             "history": [], "action": f"TOOL_{tool}", "purpose": "arg_eval", "author": "claude", "trainable": False})
    return rows


def build() -> None:
    rows = rows_all()
    problems = [f"この中で重なる: {u}" for u, k in collections.Counter(r["utterance"] for r in rows).items() if k > 1]
    old = {"学習": read_jsonl(TR3 / "utterances.jsonl"), "v1": read_jsonl(CH1), "v2": read_jsonl(CH2 / "utterances.jsonl"), "v3": read_jsonl(CH3 / "utterances.jsonl"),
           "v4": read_jsonl(CH4 / "utterances.jsonl"), "v5": read_jsonl(CH5 / "utterances.jsonl"), "v6": read_jsonl(CH6 / "utterances.jsonl")}
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
    print(f"arg_eval_v1: {len(rows)} 行（" + "・".join(f"{J[t]} {sum(r['tool'] == t for r in rows)}" for t in J) + f"）/ 形 {len(FORMS)} 個 × 10 行 / md5 {md5_of(OUT / 'utterances.jsonl')}")
    print(f"保存しました: {OUT / 'utterances.jsonl'}")


def measure() -> None:
    use_multi_library()
    from edge_router.dispatch import arguments
    from .p21_arguments import make_resolve as args_p21
    from .p22_arguments import make_resolve as args_p22
    rows = read_jsonl(OUT / "utterances.jsonl")
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    here = PROJECT_ROOT / "app"
    sums = {"arg_eval_v1": md5_of(OUT / "utterances.jsonl"), "p22_arguments.py": md5_of(here / "p22_arguments.py")}
    print(f"\n===== P22 引数の値（arg_eval_v1・{len(rows)} 行。見ていない行）=====")
    print("測ったもの（md5）: " + "・".join(f"{k} {v}" for k, v in sums.items()))
    md = [f"arg_eval_v1: {len(rows)} 行（形 {len(FORMS)} 個 × 3 機能 × 10 行）。`p22_arguments` を決めた後に書いた、見ていない行。引数の段だけを測る（正しいツールを選べたとして）\n",
          "測ったもの（md5）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "\n",
          "結果は 3 つに分ける: **合っている**／**聞き返す**（引数が足りないと返した。違う実行はしない）／**違う値で実行**（値は出たが、正解と違う）\n"]
    res = {"md5": sums, "configs": {}}
    for cname, make in (("直す前（p21_arguments）", args_p21), ("直した後（p22_arguments）", args_p22)):
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
                           "got": json.dumps(x.values, ensure_ascii=False) if x.complete else "聞き返す", "kind": kind, "note": "; ".join(x.notes)})
        res["configs"][cname] = {"counts": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, "rows": detail}
        cellf = lambda key: f"{cnt[(key, '合っている')]}／{cnt[(key, '聞き返す')]}／{cnt[(key, '違う値で実行')]}"
        n = cnt[("all", "n")]
        print(f"\n--- {cname} ---")
        print(f"合計 {n} 行: 合っている {cnt[('all', '合っている')]}（{cnt[('all', '合っている')] / n:.3f}）・聞き返す {cnt[('all', '聞き返す')]}・違う値で実行 {cnt[('all', '違う値で実行')]}（{cnt[('all', '違う値で実行')] / n:.3f}）")
        md += [f"\n## {cname}\n", f"合計 {n} 行: **合っている {cnt[('all', '合っている')]}（{cnt[('all', '合っている')] / n:.3f}）**・聞き返す {cnt[('all', '聞き返す')]}（{cnt[('all', '聞き返す')] / n:.3f}）・"
               f"**違う値で実行 {cnt[('all', '違う値で実行')]}（{cnt[('all', '違う値で実行')] / n:.3f}）**\n",
               "表の中は「合っている／聞き返す／違う値で実行」（10 行ずつ）\n", "|形|" + "|".join(J.values()) + "|合計（30 行）|", "|---|---|---|---|---|"]
        for form in FORMS:
            md.append(f"|{form}|" + "|".join(cellf(t + "|" + form) for t in J) + f"|{cellf(form)}|")
            print(f"{form:<16} " + "  ".join(f"{J[t]} {cellf(t + '|' + form)}" for t in J) + f"  合計 {cellf(form)}")
        md.append("|**合計（80 行）**|" + "|".join(cellf(t) for t in J) + f"|{cellf('all')}|")
        print("合計             " + "  ".join(f"{J[t]} {cellf(t)}" for t in J))
        if make is args_p22:
            for kind in ("違う値で実行", "聞き返す"):
                md += [f"\n### {kind}の行（直した後）\n", "|機能|形|発話|取り出した値|正解|", "|---|---|---|---|---|"]
                md += [f"|{J[d['tool']]}|{d['form']}|{d['utterance']}|{d['got']}|{d['label']}|" for d in detail if d["kind"] == kind]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "argeval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "argeval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n保存しました: {RES}（argeval.md・argeval.json）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "measure"])
    a = ap.parse_args()
    build() if a.step == "build" else measure()


if __name__ == "__main__":
    main()
