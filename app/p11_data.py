# -*- coding: utf-8 -*-
"""P11 手順 1: 文の分類器（BERT）の学習だけに足す、対象側（機能にある依頼）の発話を作る。

設計: DoC/設計書_P11_BERTのフラグが対象の依頼に立つ問題.md の 3 節。

P10 で残った不要な聞き返しの最大の原因は、BERT のフラグ（確率 0.5 以上で、操作系の実行を止めて確認へ回す）が
空調の依頼に立つことだった。学習側の交差検証でも、空調 10〜14%・メディア 11〜15%・ナビ 4% の行にフラグが立つ
（「エアコンを24度にして」のような基本の文を含む。seed で結果が変わる）。

足す行（bert_add_v1。90 行 = 30 単位 × 3 文。ラベルは「機能にない依頼ではない」= 0）:
|種類|単位|中身|
|---|---|---|
|空調|14|入れる・切る・温度（数値／相対）・暖める・風量・風向き・席の指定（運転席・助手席・後ろ）・自動・曇り取り・理由つき・くだけた言い方|
|メディア|8|止める・流す・曲送り・音量・消音・切り替え・数値・くだけた言い方|
|ナビ|5|やめる・目的地・ルートの変更・経由地・再開|
|カメラ|3|後ろ・前と横・車内と周囲|

**ヘッドの学習には使わない**（文だけを読む BERT 用。端末で `h` を集める必要がない）。
評価セット（challenge_v1・v2）とは、同じ文なし・文字 2-gram の重なり 0.5 未満（P5・P7 の 0.6 より厳しくした）。

    python -m app.p11_data      # data/bert_add_v1/added.jsonl
"""
from __future__ import annotations

import collections
import json

from .common import MULTI_ROOT, PROJECT_ROOT
from .p5_train_data import TOOL, near
from .p7_data import CH1, CH2, TR2, read_jsonl

ADD = PROJECT_ROOT / "data" / "bert_add_v1"
MAX_CHARS, NEAR = 40, 0.50
# シートヒーター・ハンドルヒーター（train_add_v2 の単位 add2_os_seatheat・add2_os_steerheat、評価セット v2 の H の同じ機能）
RELABEL = ("seatheat", "steerheat")

TG = {
    "hvac": [
        ("on", ["エアコンを入れて", "冷房をつけておいて", "空調をオンにして"]),
        ("off", ["エアコンを切って", "暖房を止めて", "空調をオフにして"]),
        ("tempabs", ["温度を22度にして", "26度に設定し直して", "エアコンの温度を20度まで下げて"]),
        ("temprel", ["もう少し涼しくしてほしい", "車内を冷やして", "温度を2度上げて"]),
        ("warm", ["車の中を暖かくしておいて", "暖房を使いたい", "ヒーターで車内を暖めて"]),
        ("fan", ["風をもっと強めに出して", "風量を3にして", "風を少し弱めて"]),
        ("vent", ["風向きを上にして", "風を顔のほうに向けて", "吹き出し口を足元に切り替えて"]),
        ("zone_driver", ["運転席の温度を上げて", "運転席側の風を強くして", "運転席だけ冷房を強めて"]),
        ("zone_passenger", ["助手席の温度を下げて", "助手席側だけ暖房を入れて", "助手席の風を弱くして"]),
        ("zone_rear", ["後部座席のエアコンをつけて", "後ろの席の温度を上げて", "後席の風量を下げて"]),
        ("auto", ["空調をオートにして", "エアコンをおまかせ運転にして", "温度を自動で調整して"]),
        ("defrost", ["ガラスの曇りを消して", "デフロスターをつけて", "曇り止めを入れて"]),
        ("reason", ["暑いから冷房を強めにかけて", "寒いので暖房をつけて", "蒸し暑いから風を強めて"]),
        ("casual", ["エアコンつけてくれる", "冷房を止めてもらえる", "暖房を弱くしてほしい"]),
    ],
    "media": [
        ("stop", ["音楽を止めておいて", "曲を一時停止して", "再生をストップして"]),
        ("play", ["音楽を流して", "再生を始めて", "止めた曲の続きをかけて"]),
        ("skip", ["次のトラックにして", "ひとつ前の曲にして", "一曲飛ばして"]),
        ("volume", ["音量を上げて", "ボリュームを少し絞って", "音を少し大きくして"]),
        ("mute", ["音を消しておいて", "ミュートにして", "消音にして"]),
        ("source", ["ラジオをかけて", "Bluetoothの音楽にして", "USBの曲を流して"]),
        ("volnum", ["音量を10にして", "ボリュームを半分にして", "音量を20まで上げて"]),
        ("casual", ["音楽を止めてくれる", "音を小さくしてほしい", "次の曲に変えてもらえる"]),
    ],
    "nav": [
        ("cancel", ["ナビを終了して", "案内を止めて", "ルート案内をやめて"]),
        ("set", ["駅を目的地にして", "自宅までの案内を始めて", "会社へのルートを出して"]),
        ("reroute", ["高速を使わないルートにして", "別の道に変えて", "下道で行くルートに切り替えて"]),
        ("waypoint", ["途中でコンビニに寄るようにして", "経由地にガソリンスタンドを追加して", "次のサービスエリアを経由地にして"]),
        ("resume", ["案内をもう一度始めて", "ルート案内を再開してほしい", "道案内をスタートして"]),
    ],
    "cam": [
        ("rear", ["バックカメラに切り替えて", "後ろのカメラをつけて", "リアカメラの映像を表示して"]),
        ("front_side", ["フロントカメラを映して", "左側のカメラを見せて", "周りの映像を画面に出して"]),
        ("cabin_around", ["車内カメラの映像を見せて", "右のカメラを画面に出して", "全周囲カメラをつけて"]),
    ],
}


def build() -> list[dict]:
    tools = {t["name"]: t for t in json.loads((MULTI_ROOT / "configs" / "tools.json").read_text(encoding="utf-8"))["tools"]}
    rows = []
    for cls, units in TG.items():
        t = tools[TOOL[cls]]
        for name, utts in units:
            if len(utts) != 3:
                raise SystemExit(f"{cls}_{name}: 3 文で 1 単位（{len(utts)} 文）")
            unit = f"bert_tg_{cls}_{name}"
            for k, u in enumerate(utts):
                rows.append({"id": f"{unit}:{k}", "group": unit, "split_unit": unit, "split": "train", "utterance": u, "history": [],
                             "state": {"motion": "stopped", "permission": "allowed"}, "last_tool": None, "route": "TOOLCALL",
                             "category": t["category"], "effect": t["effect"], "tool": t["name"], "action": f"TOOL_{t['name']}",
                             "argmode": None, "difficulty": "easy", "confusable": None, "note": "対象側の文（BERT の学習だけに使う）",
                             "trainable": True, "purpose": "classification", "added": "bert_add_v1", "add_type": "TG"})
    return rows


def main() -> None:
    rows = build()
    train = read_jsonl(TR2 / "utterances.jsonl")
    chal = read_jsonl(CH1) + read_jsonl(CH2 / "utterances.jsonl")
    c = collections.Counter(r["utterance"] for r in rows)
    dup = [u for u, v in c.items() if v > 1]
    old = {r["utterance"] for r in train}
    same_train = [r["utterance"] for r in rows if r["utterance"] in old]
    ch = sorted({r["utterance"] for r in chal})
    same_ch = [r["utterance"] for r in rows if r["utterance"] in set(ch)]
    close = [(r["utterance"], x, round(near(r["utterance"], x), 2)) for r in rows for x in ch if near(r["utterance"], x) >= NEAR]
    long = [r["utterance"] for r in rows if len(r["utterance"]) > MAX_CHARS]
    by = collections.Counter(r["tool"] for r in rows)
    print(f"足す行: {len(rows)} 行（" + "・".join(f"{k} {v}" for k, v in by.items()) + f"）/ 単位 {len({r['split_unit'] for r in rows})}")
    print(f"確かめ: 足す行の中の同じ文 {len(dup)} / 1,101 行と同じ文 {len(same_train)} / 評価セットと同じ文 {len(same_ch)} / "
          f"評価セットと近い文（2-gram の重なり {NEAR} 以上）{len(close)} / {MAX_CHARS} 字を超える文 {len(long)}")
    for name, bad in (("足す行の中の同じ文", dup), ("1,101 行と同じ文", same_train), ("評価セットと同じ文", same_ch), ("評価セットと近い文", close),
                      ("長い文", long)):
        if bad:
            raise SystemExit(f"{name}: {bad}")
    relabel = [r for r in train if r.get("os_function") in RELABEL]
    print(f"ラベルを変える案で対象側へ移す行（1,101 行のうち）: {len(relabel)} 行 = " + "・".join(r["utterance"] for r in relabel))
    ADD.mkdir(parents=True, exist_ok=True)
    (ADD / "added.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"保存しました: {ADD / 'added.jsonl'}")


if __name__ == "__main__":
    main()
