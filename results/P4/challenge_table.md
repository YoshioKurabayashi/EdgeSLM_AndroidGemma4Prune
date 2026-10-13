
### 層 17 個

|種類|行|正解|確認・聞き返しへ|別の行き先|雑談へ|読むだけの機能へ|誤実行|1 位の正解率（閾値なし）|
|---|---|---|---|---|---|---|---|---|
|A 言い換え|60|56|0|4|0|0|0|0.933|
|B 長い文・前置きつき|30|19|5|3|0|0|3|0.633|
|C 言いよどみ・口語|30|15|3|11|0|0|1|0.567|
|D 音声認識の誤り風|30|25|3|1|0|0|1|0.867|
|E 敬語・方言・カタカナ語|30|20|6|3|0|0|1|0.767|
|I 読むだけか操作かの取り違え|30|28|0|2|0|0|0|0.933|
|F 機能の語を含む雑談|40|40|0|0|0|0|0|1.000|
|G 意図が不明|40|27|（正解）|0|2|0|11|—|
|H 機能にない依頼|30|7|（正解）|0|0|0|23|—|
|V 走行中|30|26|5|0|0|0|3|1.000|
|M 複数の依頼（参考）|10|—|0|0|0|0|0|—|

まとめ（M を除く 350 行）: 行き先のある行の正解率 0.824・確認へ回した割合 0.066 / 意図が不明・機能にない依頼を確認へ回した割合 0.494 / 誤実行 43 件

誤実行した行:

|種類|発話|正解|判定|確信度|棄却の得点|
|---|---|---|---|---|---|
|B|後ろの席の子どもが寒いって言ってるから、暖房をもう少し強くしてあげて|TOOL_control_hvac|TOOL_set_navigation|0.53|0.000|
|B|急にトイレに行きたくなったから、近くのコンビニを探して|TOOL_search_poi|TOOL_set_navigation|0.54|0.000|
|B|目的地に着いたら車を停めたいので、周辺の駐車場を調べておいて|TOOL_search_poi|TOOL_set_navigation|0.63|0.000|
|C|ちょ、音でかい、下げて|TOOL_control_media|TOOL_control_hvac|0.56|0.468|
|D|走行距離を押しえて|TOOL_get_vehicle_status|TOOL_set_navigation|0.68|0.000|
|E|リアカメラをオン|TOOL_capture_camera|TOOL_control_media|0.61|0.000|
|G|もっと上げて|UNKNOWN|TOOL_control_hvac|0.77|0.476|
|G|さっきのところに行って|UNKNOWN|TOOL_set_navigation|0.88|0.850|
|G|同じのをもう一回|UNKNOWN|TOOL_control_media|0.89|0.638|
|G|逆にして|UNKNOWN|TOOL_control_media|0.82|0.058|
|G|あっちに変えて|UNKNOWN|TOOL_set_navigation|0.67|0.276|
|G|いつもの設定にして|UNKNOWN|TOOL_control_media|0.66|0.008|
|G|前の状態に戻して|UNKNOWN|TOOL_set_navigation|0.74|0.001|
|G|もうちょっと強く|UNKNOWN|TOOL_control_hvac|0.90|0.041|
|G|音量を|UNKNOWN|TOOL_control_media|0.97|0.611|
|G|とりあえずお願い|UNKNOWN|TOOL_control_media|0.50|0.979|
|G|今のは取り消し|UNKNOWN|TOOL_control_media|0.52|0.910|
|H|窓を開けて|UNKNOWN|TOOL_control_hvac|0.64|0.000|
|H|シートを倒して|UNKNOWN|TOOL_control_hvac|0.85|0.000|
|H|母さんに電話して|UNKNOWN|TOOL_set_navigation|0.87|0.015|
|H|ライトをつけて|UNKNOWN|TOOL_control_hvac|0.66|0.000|
|H|サンルーフを開けて|UNKNOWN|TOOL_control_hvac|0.68|0.000|
|H|ハザードをつけて|UNKNOWN|TOOL_control_hvac|0.57|0.000|
|H|クルーズコントロールをセットして|UNKNOWN|TOOL_set_navigation|0.79|0.000|
|H|メッセージを送って|UNKNOWN|TOOL_control_media|0.74|0.042|
|H|ミラーをたたんで|UNKNOWN|TOOL_control_hvac|0.73|0.000|
|H|駐車ブレーキをかけて|UNKNOWN|TOOL_control_hvac|0.59|0.000|
|H|自動で駐車して|UNKNOWN|TOOL_set_navigation|0.80|0.000|
|H|車高を上げて|UNKNOWN|TOOL_control_hvac|0.89|0.000|
|H|給油口を開けて|UNKNOWN|TOOL_control_hvac|0.78|0.004|
|H|ボンネットを開けて|UNKNOWN|TOOL_control_hvac|0.76|0.000|
|H|ヘッドライトをハイビームにして|UNKNOWN|TOOL_control_hvac|0.75|0.000|
|H|料金所の支払いをして|UNKNOWN|TOOL_set_navigation|0.78|0.000|
|H|走行モードをスポーツにして|UNKNOWN|TOOL_set_navigation|0.61|0.000|
|H|アラームを7時にセットして|UNKNOWN|TOOL_set_navigation|0.86|0.000|
|H|メールを読み上げて|UNKNOWN|TOOL_control_media|0.97|0.000|
|H|電話を切って|UNKNOWN|TOOL_control_media|0.99|0.001|
|H|リマインダーを設定して|UNKNOWN|TOOL_set_navigation|0.89|0.000|
|H|ピザを注文して|UNKNOWN|TOOL_set_navigation|0.73|0.000|
|H|タクシーを呼んで|UNKNOWN|TOOL_set_navigation|0.92|0.000|
|V|案内して、いややめとく|UNKNOWN|TOOL_set_navigation|0.68|0.906|
|V|もっと上げて|UNKNOWN|TOOL_control_media|0.62|0.001|
|V|それ下げて|UNKNOWN|TOOL_control_media|0.57|0.135|

### 層 26 個

|種類|行|正解|確認・聞き返しへ|別の行き先|雑談へ|読むだけの機能へ|誤実行|1 位の正解率（閾値なし）|
|---|---|---|---|---|---|---|---|---|
|A 言い換え|60|57|0|3|0|0|0|0.950|
|B 長い文・前置きつき|30|22|0|5|0|0|3|0.733|
|C 言いよどみ・口語|30|22|4|3|0|0|1|0.800|
|D 音声認識の誤り風|30|26|3|0|0|0|1|0.933|
|E 敬語・方言・カタカナ語|30|27|1|1|0|0|1|0.900|
|I 読むだけか操作かの取り違え|30|25|1|3|0|0|1|0.867|
|F 機能の語を含む雑談|40|40|0|0|0|0|0|1.000|
|G 意図が不明|40|23|（正解）|0|4|0|13|—|
|H 機能にない依頼|30|12|（正解）|0|0|4|14|—|
|V 走行中|30|24|2|1|1|0|4|0.957|
|M 複数の依頼（参考）|10|—|0|0|0|0|0|—|

まとめ（M を除く 350 行）: 行き先のある行の正解率 0.883・確認へ回した割合 0.033 / 意図が不明・機能にない依頼を確認へ回した割合 0.481 / 誤実行 38 件

誤実行した行:

|種類|発話|正解|判定|確信度|棄却の得点|
|---|---|---|---|---|---|
|B|そろそろ充電が必要だから、近くにある急速充電器を調べてくれる|TOOL_search_poi|TOOL_set_navigation|0.91|0.000|
|B|急にトイレに行きたくなったから、近くのコンビニを探して|TOOL_search_poi|TOOL_set_navigation|0.73|0.000|
|B|目的地に着いたら車を停めたいので、周辺の駐車場を調べておいて|TOOL_search_poi|TOOL_set_navigation|0.96|0.000|
|C|ちょっと、エアコン消してくれる|TOOL_control_hvac|TOOL_control_media|0.55|0.000|
|D|充電スポット或る|TOOL_search_poi|TOOL_control_hvac|0.69|0.033|
|E|ACオンにして|TOOL_control_hvac|TOOL_control_media|0.68|0.000|
|I|風量を一段上げて|TOOL_control_hvac|TOOL_control_media|0.68|0.000|
|G|案内して、いややめとく|UNKNOWN|TOOL_set_navigation|0.71|0.996|
|G|もっと上げて|UNKNOWN|TOOL_control_media|0.63|0.335|
|G|それ下げて|UNKNOWN|TOOL_control_media|0.54|0.876|
|G|さっきのところに行って|UNKNOWN|TOOL_set_navigation|0.97|0.987|
|G|同じのをもう一回|UNKNOWN|TOOL_control_media|0.95|0.056|
|G|逆にして|UNKNOWN|TOOL_control_media|0.70|0.263|
|G|あっちに変えて|UNKNOWN|TOOL_set_navigation|0.63|0.811|
|G|いつもの設定にして|UNKNOWN|TOOL_control_media|0.71|0.002|
|G|前の状態に戻して|UNKNOWN|TOOL_control_media|0.60|0.000|
|G|もうちょっと強く|UNKNOWN|TOOL_control_hvac|0.82|0.254|
|G|音量を|UNKNOWN|TOOL_control_media|0.96|0.219|
|G|ほどほどにやって|UNKNOWN|TOOL_control_media|0.55|0.995|
|G|今のは取り消し|UNKNOWN|TOOL_set_navigation|0.51|0.011|
|H|窓を開けて|UNKNOWN|TOOL_control_hvac|0.86|0.000|
|H|ライトをつけて|UNKNOWN|TOOL_control_hvac|0.66|0.000|
|H|サンルーフを開けて|UNKNOWN|TOOL_control_hvac|0.87|0.000|
|H|クルーズコントロールをセットして|UNKNOWN|TOOL_set_navigation|0.63|0.000|
|H|メッセージを送って|UNKNOWN|TOOL_control_media|0.69|0.989|
|H|自動で駐車して|UNKNOWN|TOOL_set_navigation|0.59|0.000|
|H|ルームランプを消して|UNKNOWN|TOOL_control_media|0.73|0.000|
|H|料金所の支払いをして|UNKNOWN|TOOL_set_navigation|0.85|0.000|
|H|走行モードをスポーツにして|UNKNOWN|TOOL_set_navigation|0.55|0.000|
|H|メールを読み上げて|UNKNOWN|TOOL_control_media|0.97|0.000|
|H|電話を切って|UNKNOWN|TOOL_control_media|0.96|0.000|
|H|リマインダーを設定して|UNKNOWN|TOOL_set_navigation|0.59|0.000|
|H|ピザを注文して|UNKNOWN|TOOL_set_navigation|0.85|0.000|
|H|タクシーを呼んで|UNKNOWN|TOOL_set_navigation|0.95|0.000|
|V|つけて、いややっぱ消して|UNKNOWN|TOOL_control_media|0.56|0.425|
|V|案内して、いややめとく|UNKNOWN|TOOL_set_navigation|0.82|0.612|
|V|もっと上げて|UNKNOWN|TOOL_control_media|0.82|0.010|
|V|それ下げて|UNKNOWN|TOOL_control_media|0.58|0.060|

### 層 35 個

|種類|行|正解|確認・聞き返しへ|別の行き先|雑談へ|読むだけの機能へ|誤実行|1 位の正解率（閾値なし）|
|---|---|---|---|---|---|---|---|---|
|A 言い換え|60|58|1|1|0|0|0|0.967|
|B 長い文・前置きつき|30|24|1|2|0|0|3|0.800|
|C 言いよどみ・口語|30|22|2|5|0|0|1|0.767|
|D 音声認識の誤り風|30|24|4|2|0|0|0|0.867|
|E 敬語・方言・カタカナ語|30|26|1|2|0|0|1|0.900|
|I 読むだけか操作かの取り違え|30|25|0|5|0|0|0|0.833|
|F 機能の語を含む雑談|40|40|0|0|0|0|0|1.000|
|G 意図が不明|40|24|（正解）|0|7|0|9|—|
|H 機能にない依頼|30|11|（正解）|0|0|9|10|—|
|V 走行中|30|23|1|1|3|0|3|0.957|
|M 複数の依頼（参考）|10|—|0|0|0|0|0|—|

まとめ（M を除く 350 行）: 行き先のある行の正解率 0.883・確認へ回した割合 0.033 / 意図が不明・機能にない依頼を確認へ回した割合 0.468 / 誤実行 27 件

誤実行した行:

|種類|発話|正解|判定|確信度|棄却の得点|
|---|---|---|---|---|---|
|B|そろそろ充電が必要だから、近くにある急速充電器を調べてくれる|TOOL_search_poi|TOOL_set_navigation|0.75|0.000|
|B|急にトイレに行きたくなったから、近くのコンビニを探して|TOOL_search_poi|TOOL_set_navigation|0.52|0.000|
|B|目的地に着いたら車を停めたいので、周辺の駐車場を調べておいて|TOOL_search_poi|TOOL_set_navigation|0.96|0.000|
|C|あ、やば、ガソスタ探して|TOOL_search_poi|TOOL_set_navigation|0.55|0.010|
|E|ACオンにして|TOOL_control_hvac|TOOL_control_media|0.67|0.000|
|G|案内して、いややめとく|UNKNOWN|TOOL_set_navigation|0.70|0.971|
|G|さっきのところに行って|UNKNOWN|TOOL_set_navigation|0.88|0.917|
|G|同じのをもう一回|UNKNOWN|TOOL_control_media|0.93|0.480|
|G|逆にして|UNKNOWN|TOOL_control_media|0.59|0.372|
|G|いつもの設定にして|UNKNOWN|TOOL_control_media|0.75|0.163|
|G|もうちょっと強く|UNKNOWN|TOOL_control_hvac|0.72|0.482|
|G|ナビを|UNKNOWN|TOOL_set_navigation|0.96|0.400|
|G|音量を|UNKNOWN|TOOL_control_media|0.91|0.584|
|G|今のは取り消し|UNKNOWN|TOOL_control_media|0.62|0.207|
|H|窓を開けて|UNKNOWN|TOOL_control_hvac|0.86|0.000|
|H|ライトをつけて|UNKNOWN|TOOL_control_hvac|0.62|0.000|
|H|サンルーフを開けて|UNKNOWN|TOOL_control_hvac|0.57|0.000|
|H|クルーズコントロールをセットして|UNKNOWN|TOOL_set_navigation|0.54|0.000|
|H|エンジンを切って|UNKNOWN|TOOL_control_media|0.51|0.000|
|H|料金所の支払いをして|UNKNOWN|TOOL_set_navigation|0.80|0.000|
|H|メールを読み上げて|UNKNOWN|TOOL_control_media|0.77|0.000|
|H|電話を切って|UNKNOWN|TOOL_control_media|0.87|0.002|
|H|リマインダーを設定して|UNKNOWN|TOOL_control_media|0.60|0.000|
|H|タクシーを呼んで|UNKNOWN|TOOL_set_navigation|0.98|0.000|
|V|つけて、いややっぱ消して|UNKNOWN|TOOL_control_media|0.62|0.174|
|V|案内して、いややめとく|UNKNOWN|TOOL_set_navigation|0.84|0.018|
|V|もっと上げて|UNKNOWN|TOOL_control_media|0.52|0.005|

### 層 17 と層 26 で判定が違う行（73 / 360）

|種類|発話|正解|層 17|層 26|
|---|---|---|---|---|
|A|ドアが開いてないか確認して|TOOL_get_vehicle_status|TOOL_capture_camera（別の行き先）|TOOL_get_vehicle_status（正解）|
|A|周辺の駐車場を調べて|TOOL_search_poi|TOOL_get_vehicle_status（別の行き先）|TOOL_capture_camera（別の行き先）|
|B|さっきから少し暑い気がするんだけど、エアコンの温度を2度くらい下げてもらえるかな|TOOL_control_hvac|TOOL_search_poi（別の行き先）|TOOL_control_hvac（正解）|
|B|後ろの席の子どもが寒いって言ってるから、暖房をもう少し強くしてあげて|TOOL_control_hvac|TOOL_set_navigation（誤実行）|TOOL_control_hvac（正解）|
|B|雨で窓ガラスが曇って前が見えにくいから、曇りを取ってほしい|TOOL_control_hvac|CLARIFY（確認へ）|TOOL_control_hvac（正解）|
|B|この曲はもう何回も聴いて飽きたから、次の曲に進めてほしい|TOOL_control_media|TOOL_control_media（正解）|CHAT（別の行き先）|
|B|子どもが寝たから、流れている音楽をいったん止めておいて|TOOL_control_media|CLARIFY（確認へ）|TOOL_control_media（正解）|
|B|最近あまり乗っていなかったから、バッテリーの残量を見ておきたい|TOOL_get_vehicle_status|CLARIFY（確認へ）|CHAT（別の行き先）|
|B|さっきドアを閉めた音が変だったから、ちゃんと閉まっているか見て|TOOL_get_vehicle_status|CLARIFY（確認へ）|CHAT（別の行き先）|
|B|そろそろ充電が必要だから、近くにある急速充電器を調べてくれる|TOOL_search_poi|CLARIFY（確認へ）|TOOL_set_navigation（誤実行）|
|C|あ、そうだ、ガソリンってあとどんくらい|TOOL_get_vehicle_status|CLARIFY（確認へ）|TOOL_get_vehicle_status（正解）|
|C|んーと、エアコン、もうちょい下げてくんない|TOOL_control_hvac|CHAT（別の行き先）|TOOL_control_hvac（正解）|
|C|あのさ、音楽かけてよ|TOOL_control_media|CHAT（別の行き先）|TOOL_control_media（正解）|
|C|ねえ、今の曲なんだっけ|TOOL_get_media_info|CHAT（別の行き先）|TOOL_get_media_info（正解）|
|C|えー、このランプ何だっけ、どういう意味|TOOL_search_vehicle_manual|TOOL_get_media_info（別の行き先）|CLARIFY（確認へ）|
|C|ちょ、音でかい、下げて|TOOL_control_media|TOOL_control_hvac（誤実行）|TOOL_control_media（正解）|
|C|あー寒い寒い、暖房つけて|TOOL_control_hvac|CHAT（別の行き先）|TOOL_control_hvac（正解）|
|C|あれ、ドア閉まってるっけ|TOOL_get_vehicle_status|CHAT（別の行き先）|CLARIFY（確認へ）|
|C|あのー、25度にしてもらえる|TOOL_control_hvac|TOOL_control_hvac（正解）|CLARIFY（確認へ）|
|C|あ、やば、ガソスタ探して|TOOL_search_poi|TOOL_search_poi（正解）|CLARIFY（確認へ）|
|C|ねえ、あと何キロ走れんの|TOOL_get_vehicle_status|CHAT（別の行き先）|TOOL_get_vehicle_status（正解）|
|C|あ、そうそう、ラジオにして|TOOL_control_media|CLARIFY（確認へ）|TOOL_control_media（正解）|
|C|ちょっと、エアコン消してくれる|TOOL_control_hvac|CLARIFY（確認へ）|TOOL_control_media（誤実行）|
|C|あー、この曲のアルバムって何|TOOL_get_media_info|CHAT（別の行き先）|TOOL_get_media_info（正解）|
|D|近く コンビニ|TOOL_search_poi|CLARIFY（確認へ）|TOOL_search_poi（正解）|
|D|走行距離を押しえて|TOOL_get_vehicle_status|TOOL_set_navigation（誤実行）|CLARIFY（確認へ）|
|D|充電スポット或る|TOOL_search_poi|CLARIFY（確認へ）|TOOL_control_hvac（誤実行）|
|D|空港まで尾根がい|TOOL_set_navigation|TOOL_get_vehicle_status（別の行き先）|CLARIFY（確認へ）|
|E|温度を下げていただけますか|TOOL_control_hvac|CLARIFY（確認へ）|TOOL_control_hvac（正解）|
|E|音楽を再生していただけますか|TOOL_control_media|CLARIFY（確認へ）|TOOL_control_media（正解）|
|E|ガソリンあとなんぼ残っとる|TOOL_get_vehicle_status|CLARIFY（確認へ）|CHAT（別の行き先）|
|E|うしろ見せてくれんね|TOOL_capture_camera|CLARIFY（確認へ）|TOOL_capture_camera（正解）|
|E|ちょっと寒いけん暖房つけて|TOOL_control_hvac|CHAT（別の行き先）|TOOL_control_hvac（正解）|
|E|空気圧ってなんぼがええの|TOOL_search_vehicle_manual|CHAT（別の行き先）|TOOL_search_vehicle_manual（正解）|
|E|ACオンにして|TOOL_control_hvac|CLARIFY（確認へ）|TOOL_control_media（誤実行）|
|E|リアカメラをオン|TOOL_capture_camera|TOOL_control_media（誤実行）|CLARIFY（確認へ）|
|E|パーキングをサーチして|TOOL_search_poi|CLARIFY（確認へ）|TOOL_search_poi（正解）|
|E|マニュアルでタイヤプレッシャーを調べて|TOOL_search_vehicle_manual|TOOL_get_vehicle_status（別の行き先）|TOOL_search_vehicle_manual（正解）|
|I|今の風量はいくつ|TOOL_get_vehicle_status|TOOL_get_vehicle_status（正解）|TOOL_get_media_info（別の行き先）|
|I|風量を一段上げて|TOOL_control_hvac|TOOL_control_hvac（正解）|TOOL_control_media（誤実行）|
|I|充電できるところを一覧で見せて|TOOL_search_poi|TOOL_get_vehicle_status（別の行き先）|TOOL_search_vehicle_manual（別の行き先）|
|I|後ろのドアは閉まってる|TOOL_get_vehicle_status|TOOL_get_vehicle_status（正解）|CLARIFY（確認へ）|
|G|かけて、あ、やっぱいい|UNKNOWN|REJECT（確認へ）|CHAT（雑談へ）|
|G|案内して、いややめとく|UNKNOWN|REJECT（確認へ）|TOOL_set_navigation（誤実行）|
|G|もっと上げて|UNKNOWN|TOOL_control_hvac（誤実行）|TOOL_control_media（誤実行）|
|G|それ下げて|UNKNOWN|REJECT（確認へ）|TOOL_control_media（誤実行）|
|G|前の状態に戻して|UNKNOWN|TOOL_set_navigation（誤実行）|TOOL_control_media（誤実行）|
|G|えっと、この曲の|UNKNOWN|CHAT（雑談へ）|REJECT（確認へ）|
|G|ほどほどにやって|UNKNOWN|REJECT（確認へ）|TOOL_control_media（誤実行）|
|G|任せるよ|UNKNOWN|REJECT（確認へ）|CHAT（雑談へ）|
|G|とりあえずお願い|UNKNOWN|TOOL_control_media（誤実行）|REJECT（確認へ）|
|G|あ、なんでもない|UNKNOWN|REJECT（確認へ）|CHAT（雑談へ）|
|G|今のは取り消し|UNKNOWN|TOOL_control_media（誤実行）|TOOL_set_navigation（誤実行）|
|H|シートを倒して|UNKNOWN|TOOL_control_hvac（誤実行）|CLARIFY（確認へ）|
|H|母さんに電話して|UNKNOWN|TOOL_set_navigation（誤実行）|CLARIFY（確認へ）|
|H|ドアのロックを開けて|UNKNOWN|CLARIFY（確認へ）|TOOL_get_vehicle_status（読むだけの機能へ）|
|H|トランクを開けて|UNKNOWN|CLARIFY（確認へ）|TOOL_get_vehicle_status（読むだけの機能へ）|
|H|ハザードをつけて|UNKNOWN|TOOL_control_hvac（誤実行）|TOOL_capture_camera（読むだけの機能へ）|
|H|ミラーをたたんで|UNKNOWN|TOOL_control_hvac（誤実行）|CLARIFY（確認へ）|
|H|駐車ブレーキをかけて|UNKNOWN|TOOL_control_hvac（誤実行）|CLARIFY（確認へ）|
|H|車高を上げて|UNKNOWN|TOOL_control_hvac（誤実行）|CLARIFY（確認へ）|
|H|給油口を開けて|UNKNOWN|TOOL_control_hvac（誤実行）|CLARIFY（確認へ）|
|H|ボンネットを開けて|UNKNOWN|TOOL_control_hvac（誤実行）|CLARIFY（確認へ）|
|H|ルームランプを消して|UNKNOWN|CLARIFY（確認へ）|TOOL_control_media（誤実行）|
|H|ヘッドライトをハイビームにして|UNKNOWN|TOOL_control_hvac（誤実行）|TOOL_capture_camera（読むだけの機能へ）|
|H|アラームを7時にセットして|UNKNOWN|TOOL_set_navigation（誤実行）|CLARIFY（確認へ）|
|M|エアコンつけて、音楽もかけて|MULTI|CLARIFY（参考）|TOOL_control_hvac（参考）|
|M|充電スポットを探して、そこに向かって|MULTI|TOOL_set_navigation（参考）|TOOL_search_poi（参考）|
|M|エアコンを消して、ラジオをつけて|MULTI|CLARIFY（参考）|TOOL_control_hvac（参考）|
|V|何か音楽ほしいな|TOOL_control_media|CLARIFY（確認へ）|CHAT（別の行き先）|
|V|つけて、いややっぱ消して|UNKNOWN|CLARIFY（確認へ）|TOOL_control_media（誤実行）|
|V|上げて、やっぱり下げて、いやそのままで|UNKNOWN|REJECT（確認へ）|CLARIFY（確認へ）|
|V|かけて、あ、やっぱいい|UNKNOWN|REJECT（確認へ）|CHAT（雑談へ）|
