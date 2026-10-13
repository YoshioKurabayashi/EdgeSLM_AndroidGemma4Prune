# -*- coding: utf-8 -*-
"""P25 手順 3: 引数の値だけを測る、3 つ目の新しい行（arg_eval_v3・240 行）。端末なし。

設計: DoC/設計書_P25_引数の歯止め.md の 5 節。

arg_eval_v2 は、P23 の結果と歯止め（`p25_arguments`）を作るときに見たので「直す用」になった。
**歯止めを固定した後に**（md5 は `results/P25/preregistration.md`）、新しい 240 行を書いた。正解の引数は、文の意味だけから付けた（取り出しの結果を見る前）。
形は arg_eval_v2 と同じ 10 個 × 3 機能 × 8 行（比べられるように。新しい形は入れていない）。

    python -m app.p25_argeval build       # data/arg_eval_v3/utterances.jsonl（重なりを確かめる）
    python -m app.p25_argeval measure     # results/P25/argeval.md・argeval.json
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

OUT = PROJECT_ROOT / "data" / "arg_eval_v3"
RES = PROJECT_ROOT / "results" / "P25"
PER = 8

HVAC = [
    [("暖房の温度を27度に設定して", "t27|t27,m:heat"), ("風量を4にしてください", "f4"), ("エアコンを切ってください", "m:off"), ("デフロスターを作動させて", "m:defrost"),
     ("温度を少し上げて", "t26"), ("冷房の風を弱くしてほしい", "f2"), ("エアコンを自動にしてください", "m:auto"), ("クーラーを入れてください", "m:cool")],
    [("日差しが強いから冷房を入れて", "m:cool"), ("足が冷えるので温度を上げて", "t26"), ("フロントガラスが白くなったので曇りを取って", "m:defrost"), ("寒気がするから暖房を強くして", "f4"),
     ("トンネルを出たら蒸すから風量を5にして", "f5"), ("音が気になるから送風を弱くして", "f2"), ("もう着くのでエアコンを止めて", "m:off"), ("汗が引かないから温度を22度にして", "t22")],
    [("冷房を入れて、蒸し暑いんだ", "m:cool"), ("温度を1度上げて、手がかじかむ", "t26"), ("風量を落として、寒いくらいだ", "f2"), ("曇りを取って、ミラーが見えない", "m:defrost"),
     ("暖房を弱めて、のぼせそう", "f2"), ("エアコンを切って、もう十分だから", "m:off"), ("24度にして、ちょっと暑い", "t24"), ("風を強くして、こもってる感じがする", "f4")],
    [("冷房をつけていただいてもよろしいですか", "m:cool"), ("温度を2度上げていただけると助かります", "t27"), ("風量を2にしていただいてもいいですか", "f2"), ("暖房を止めていただけますでしょうか", "m:off"),
     ("曇り止めをお願いできますか", "m:defrost"), ("少し暖かくしていただけませんか", "t26"), ("差し支えなければ風を弱めてください", "f2"), ("空調を入れていただけると嬉しいです", "m:auto")],
    [("冷房いれて", "m:cool"), ("エアコンけして", "m:off"), ("送風よわめにして", "f2"), ("あつ、温度さげて", "t24"),
     ("27度でよろしく", "t27"), ("デフロスターよろしく", "m:defrost"), ("ヒーターつけてー", "m:heat"), ("風もうちょい強く", "f4")],
    [("渋滞の間はエアコンをオートにして", "m:auto"), ("高速に乗ったら冷房を強めて", "f4"), ("信号で止まってる間に温度を23度にして", "t23"), ("駐車場に着くまで暖房を入れておいて", "m:heat"),
     ("寝ている間は風量を1にして", "f1"), ("雨が降ってきたら曇りを取って", "m:defrost"), ("帰り道は温度を下げておいて", "t24"), ("家に着いたらエアコンを切って", "m:off")],
    [("温度を二十三度にして", "t23"), ("風量を五にして", "f5"), ("設定温度を三度ほど下げて", "t22"), ("風量をふたつ上げて", "f5"),
     ("温度を最高にして", "t30"), ("風の強さを一番下にして", "f0"), ("温度を二十四度半にして", "t24.5"), ("風を一段弱くして", "f2")],
    [("温度を下げて風量も上げて", "t24,f4"), ("クーラーではなくヒーターを入れて", "m:heat"), ("温度は22度、風量は5にして", "t22,f5"), ("暖房は止めないで温度だけ下げて", "t24"),
     ("温度はいいから風を強くして", "f4"), ("冷房を切って曇り取りを入れて", "m:defrost"), ("風を弱めるんじゃなくて温度を上げて", "t26"), ("エアコンは切らずに風だけ止めて", "f0")],
    [("暖房つけられるかな", "m:heat"), ("温度を上げてもいい", "t26"), ("もうちょっと涼しくならないかな", "t24"), ("風を強くしたいな", "f4"),
     ("温度を20度にできるかな", "t20"), ("デフロスターを入れたいんだけど", "m:defrost"), ("冷房を切りたいな", "m:off"), ("風量を下げられる", "f2")],
    [("えっと、さっきから足元がすごく冷えるので、暖房をつけてもらってもいいかな", "m:heat"), ("あのさ、日が差してきて暑くなったから、冷房を少し強めにしてくれる", "f4"),
     ("うーん、外が雨で窓が真っ白になってきたから、デフロスターを入れてほしいな", "m:defrost"), ("そうだな、風の音がうるさく感じるから、風量を1つ下げてもらえるかな", "f2"),
     ("えー、そろそろ目的地に着くから、エアコンはもう切ってしまっていいよ", "m:off"), ("あの、ちょっと肌寒いので、設定温度を26度に上げてもらえると助かるんだけど", "t26"),
     ("ねえ、助手席の人が寒がってるから、温度を2度上げてあげてくれる", "t27"), ("んー、なんだか空気がよどんでる気がするので、風量を4にしてほしいな", "f4")],
]
MEDIA = [
    [("音楽をスタートして", "play"), ("再生を一時停止してください", "pause"), ("曲を一つ先に送って", "next"), ("前の曲に戻って", "prev"),
     ("音量を17に設定して", "v17"), ("ボリュームを大きくしてください", "v14"), ("音を下げてください", "v10"), ("音をミュートしてください", "mute")],
    [("眠くなってきたから音楽をかけて", "play"), ("電話が来たので曲を止めて", "pause"), ("この歌は聞き飽きたので飛ばして", "next"), ("聞き逃したので一曲戻して", "prev"),
     ("エアコンの音で聞こえないから音を大きくして", "v14"), ("赤ちゃんが寝たので音を小さくして", "v10"), ("料金所なので音を消して", "mute|pause"), ("お気に入りだから音量を20にして", "v20")],
    [("音楽を止めて、話がしたいんだ", "pause"), ("次の曲にして、これは苦手", "next"), ("音量を下げて、ちょっとうるさい", "v10"), ("ボリュームを上げて、ほとんど聞こえない", "v14"),
     ("音楽をかけて、退屈なんだ", "play"), ("前の曲にして、もう一回聞きたい", "prev"), ("音量を10にして、大きすぎる", "v10"), ("ミュートにして、着信だ", "mute")],
    [("音楽を流していただけますか", "play"), ("曲を止めていただけますでしょうか", "pause"), ("次の曲にしていただけると助かります", "next"), ("音量を少し上げていただけませんか", "v14"),
     ("音を少し下げていただいてもよろしいですか", "v10"), ("ひとつ前の曲に戻していただけますか", "prev"), ("音量を19に変えていただけますか", "v19"), ("消音にしていただけると嬉しいです", "mute")],
    [("音楽ながして", "play"), ("曲いったんとめて", "pause"), ("次のやつ", "next"), ("音ちっちゃく", "v10"),
     ("ボリュームあげて", "v14"), ("音量12で", "v12"), ("前のにして", "prev"), ("音けして", "mute|pause")],
    [("トンネルを抜けたら音楽をかけて", "play"), ("電話が終わるまで曲を止めておいて", "pause"), ("この曲が終わったら次にして", "next"), ("高速に入ったら音量を上げて", "v14"),
     ("子どもが寝ている間は音を小さくして", "v10"), ("駐車するまで消音にして", "mute"), ("家に着く前に前の曲をもう一度流して", "prev|play"), ("夜の間は音量を8にして", "v8")],
    [("音量を十にして", "v10"), ("音を4つ分上げて", "v16"), ("音量をいっぱいまで上げて", "v40"), ("ボリュームを今の半分に下げて", "v6"),
     ("音量を4だけ上げて", "v16"), ("音量をゼロにして", "v0"), ("音量を気持ち下げて", "v10"), ("音量を25にして", "v25")],
    [("一時停止じゃなくて消音にして", "mute"), ("ボリュームは上げずに下げて", "v10"), ("曲はそのままで音量だけ上げて", "v14"), ("ラジオじゃなくて音楽を流して", "play"),
     ("戻るんじゃなくて先の曲にして", "next"), ("音を消すんじゃなくて小さくして", "v10"), ("音楽を止めないで音量を6にして", "v6"), ("停止じゃなくて再生して", "play")],
    [("音楽流せる", "play"), ("音を下げられる", "v10"), ("前の曲に戻りたいな", "prev"), ("この曲を飛ばしたいんだけど", "next"),
     ("音量を22にできる", "v22"), ("もう少し音が大きくならないかな", "v14"), ("曲を止めてもいいかな", "pause"), ("何か曲を聴きたいな", "play")],
    [("えっと、渋滞で退屈になってきたから、何か音楽を流してもらえるかな", "play"), ("あの、これから電話をかけるので、いったん曲を止めておいてほしいんだけど", "pause"),
     ("うーん、この曲はちょっと暗いから、次の曲に飛ばしてくれると嬉しい", "next"), ("ねえ、窓を開けたら全然聞こえなくなったから、音量を少し上げてくれる", "v14"),
     ("そうだ、さっきの歌がもう一度聞きたいから、前の曲に戻してほしいな", "prev"), ("えー、後ろで子どもが寝ているから、音量を6まで下げてもらえる", "v6"),
     ("あのさ、ナビの案内を聞きたいから、少しの間音を消しておいて", "mute|pause"), ("んー、さっきから音が大きすぎる気がするので、ボリュームを少し下げてもらえるかな", "v10")],
]
NAV = [
    [("仙台駅まで案内して", "set:仙台駅"), ("祖母の家を目的地にして", "set:祖母の家"), ("市役所を目的地にして", "set:市役所"), ("ルート案内を終了してください", "cancel"),
     ("別のルートを探して", "reroute"), ("ローソンを経由地に入れて", "way:ローソン"), ("大阪城まで連れて行って", "set:大阪城"), ("横浜港に向かって", "set:横浜港")],
    [("ガソリンが少ないのでスタンドに向かって", "set:スタンド"), ("薬を買いたいから薬局まで案内して", "set:薬局"), ("渋滞してるから違うルートにして", "reroute"), ("道はわかったので案内をやめて", "cancel"),
     ("トイレに行きたいからサービスエリアに寄って", "way:サービスエリア|set:サービスエリア"), ("荷物を受け取るので郵便局を目的地にして", "set:郵便局"), ("忘れ物をしたので自宅に戻って", "set:自宅"),
     ("会議があるから本社まで案内して", "set:本社")],
    [("駅まで案内して、電車に乗るんだ", "set:駅"), ("案内を止めて、もう近いから", "cancel"), ("ルートを変えて、この道は工事中みたい", "reroute"), ("病院に向かって、具合が悪いんだ", "set:病院"),
     ("コンビニに寄って、飲み物を買いたい", "way:コンビニ|set:コンビニ"), ("空港を目的地にして、出張なんだ", "set:空港"), ("実家まで案内して、久しぶりなんだ", "set:実家"),
     ("高速を使うルートにして、急いでるから", "reroute")],
    [("京都駅まで案内していただけますか", "set:京都駅"), ("案内を終わらせていただけますか", "cancel"), ("ほかの道で案内していただけると助かります", "reroute"), ("市民会館を目的地にしていただけますか", "set:市民会館"),
     ("途中で銀行に寄っていただけますか", "way:銀行"), ("自宅まで案内をお願いできますでしょうか", "set:自宅"), ("空いている道を選んでいただけますか", "reroute"), ("美術館まで連れて行っていただけますか", "set:美術館")],
    [("駅まで頼む", "set:駅"), ("家まで", "set:家"), ("案内ストップ", "cancel"), ("ルート変更で", "reroute"),
     ("空港よろしく", "set:空港"), ("コンビニ寄りたい", "way:コンビニ|set:コンビニ"), ("目的地、東京タワー", "set:東京タワー"), ("会社いって", "set:会社")],
    [("高速を降りたらガソリンスタンドに寄って", "way:ガソリンスタンド|set:ガソリンスタンド"), ("次の信号を過ぎたら案内を止めて", "cancel"), ("渋滞が続くなら別の道にして", "reroute"),
     ("お昼になったらファミレスに向かって", "set:ファミレス"), ("帰りは自宅まで案内して", "set:自宅"), ("雨がひどくなったら下道で案内して", "reroute"), ("駅に着く前に花屋に寄って", "way:花屋"),
     ("夕方には実家に向かって", "set:実家")],
    [("国道6号沿いのコンビニに向かって", "set:国道6号沿いのコンビニ"), ("5番出口の駐車場まで案内して", "set:5番出口の駐車場"), ("イオンモールの屋上駐車場へ向かって", "set:イオンモールの屋上駐車場"),
     ("くら寿司に向かって", "set:くら寿司"), ("2丁目の公園まで案内して", "set:2丁目の公園"), ("北口のバス停へ行って", "set:北口のバス停"), ("ららぽーとの第3駐車場まで案内して", "set:ららぽーとの第3駐車場"),
     ("みなとみらいまで案内して", "set:みなとみらい")],
    [("ホテルではなく旅館に向かって", "set:旅館"), ("ジムはやめて自宅に帰って", "set:自宅"), ("有料道路じゃなくて一般道で案内して", "reroute"), ("目的地は変えずにコンビニを経由地に入れて", "way:コンビニ"),
     ("映画館じゃなくて図書館に行って", "set:図書館"), ("案内は続けたまま別のルートにして", "reroute"), ("郵便局に寄ってから会社に向かって", "set:会社|way:郵便局"), ("名古屋から大阪まで案内して", "set:大阪")],
    [("新宿駅に行きたいな", "set:新宿駅"), ("病院まで行けるかな", "set:病院"), ("空港まで案内してもらえる", "set:空港"), ("スーパーに寄りたいんだけど", "way:スーパー|set:スーパー"),
     ("案内を止めてもいい", "cancel"), ("もっと早い道はないかな", "reroute"), ("実家まで連れていってくれない", "set:実家"), ("動物園に行きたいんだけど", "set:動物園")],
    [("えっと、友達と待ち合わせをしているので、渋谷駅まで案内してもらえるかな", "set:渋谷駅"), ("あの、もうこの辺りは知ってる道だから、案内は止めてもらって大丈夫", "cancel"),
     ("うーん、この先で事故があったみたいだから、別の道を探してくれると助かる", "reroute"), ("ねえ、ガソリンがそろそろ心配だから、途中でスタンドに寄ってくれる", "way:スタンド"),
     ("そうだ、母に頼まれた物があるので、帰りにドラッグストアに寄ってほしいんだけど", "way:ドラッグストア|set:ドラッグストア"), ("えー、子どもを迎えに行かないといけないから、保育園まで案内してくれる", "set:保育園"),
     ("あのさ、今日は疲れたから、まっすぐ家に帰るルートにしてほしいな", "set:家"), ("んー、ちょっと時間があるから、海の見える公園に向かってもらえるかな", "set:海の見える公園")],
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
                rows.append({"id": f"a3_{tool.split('_')[-1][:5]}_{len(rows):03d}", "tool": tool, "form": form, "utterance": u, "label": label, "expected": parse(tool, label),
                             "history": [], "action": f"TOOL_{tool}", "purpose": "arg_eval", "author": "claude", "trainable": False})
    return rows


def build() -> None:
    rows = rows_all()
    problems = [f"この中で重なる: {u}" for u, k in collections.Counter(r["utterance"] for r in rows).items() if k > 1]
    old = {"学習": read_jsonl(TR3 / "utterances.jsonl"), "v1": read_jsonl(CH1), "v2": read_jsonl(CH2 / "utterances.jsonl"), "v3": read_jsonl(CH3 / "utterances.jsonl"),
           "v4": read_jsonl(CH4 / "utterances.jsonl"), "v5": read_jsonl(CH5 / "utterances.jsonl"), "v6": read_jsonl(CH6 / "utterances.jsonl"),
           "arg_eval_v1": read_jsonl(ARG1 / "utterances.jsonl"), "arg_eval_v2": read_jsonl(ARG2 / "utterances.jsonl")}
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
    print(f"arg_eval_v3: {len(rows)} 行（" + "・".join(f"{J[t]} {sum(r['tool'] == t for r in rows)}" for t in J) + f"）/ 形 {len(FORMS)} 個 × {PER} 行 / md5 {md5_of(OUT / 'utterances.jsonl')}")
    print(f"保存しました: {OUT / 'utterances.jsonl'}")


def measure() -> None:
    use_multi_library()
    from edge_router.dispatch import arguments
    from .p21_arguments import make_resolve as args_p21
    from .p23_arguments import make_resolve as args_p23
    from .p25_arguments import make_resolve as args_p25
    rows = read_jsonl(OUT / "utterances.jsonl")
    state = json.loads((MULTI_ROOT / "data" / "samples" / "vehicle_state.json").read_text(encoding="utf-8"))
    here = PROJECT_ROOT / "app"
    sums = {"arg_eval_v3": md5_of(OUT / "utterances.jsonl"), "p25_arguments.py": md5_of(here / "p25_arguments.py"), "p23_arguments.py": md5_of(here / "p23_arguments.py")}
    print(f"\n===== P25 引数の値（arg_eval_v3・{len(rows)} 行。見ていない行）=====")
    print("測ったもの（md5）: " + "・".join(f"{k} {v}" for k, v in sums.items()))
    md = [f"arg_eval_v3: {len(rows)} 行（形 {len(FORMS)} 個 × 3 機能 × {PER} 行）。歯止め（`p25_arguments`）を固定した後に書いた、見ていない行。引数の段だけを測る\n",
          "測ったもの（md5）: " + "・".join(f"{k} `{v}`" for k, v in sums.items()) + "（固定した md5 は `results/P25/preregistration.md`）\n",
          "結果は 3 つに分ける: **合っている**／**聞き返す**（違う実行はしない）／**違う値で実行**\n"]
    res = {"md5": sums, "configs": {}}
    configs = (("P21（直す前）", args_p21), ("P23（いまの引数）", args_p23), ("P25（歯止めを足す）", args_p25))
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
    a, b = details[configs[1][0]], details[configs[2][0]]
    body += ["\n## P23 → P25 で変わった行（歯止めが止めた行）\n", "|機能|形|発話|P23 の値|P23 の採点|正解|止めた理由|", "|---|---|---|---|---|---|---|"]
    changed = collections.Counter()
    for x, y in zip(a, b):
        if x["kind"] != y["kind"]:
            changed[(x["kind"], y["kind"])] += 1
            body.append(f"|{J[x['tool']]}|{x['form']}|{x['utterance']}|{x['got']}|{x['kind']}|{x['label']}|{y['note'].replace('P25: ', '').replace(' → 聞き返す', '')}|")
    body.append("\n変わり方: " + ("・".join(f"{k[0]} → {k[1]} {v} 行" for k, v in changed.most_common()) if changed else "なし"))
    print("\nP23 → P25 で変わった行: " + ("・".join(f"{k[0]} → {k[1]} {v} 行" for k, v in changed.most_common()) if changed else "なし"))
    for kind in ("違う値で実行", "聞き返す"):
        body += [f"\n## P25 で{kind}の行\n", "|機能|形|発話|取り出した値|正解|メモ|", "|---|---|---|---|---|---|"]
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
