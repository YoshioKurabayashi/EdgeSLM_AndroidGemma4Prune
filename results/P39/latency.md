# P39: 判定 1 回の時間の内訳（判定デモ v3・測定）

作成: 2026-10-15 10:00（`python -m app.p39_latency report`）｜ 設計: [DoC/設計書_P39_時間の内訳を測る.md](../../DoC/設計書_P39_時間の内訳を測る.md）

数字は ms・中央値（括弧は p90）。par = アプリ v2 と同じ（BERT を NPU と並行）、serial = BERT を先に単独で動かしてから Gemma。35 層は層 26 のヘッドを借りたので**判定は見ない（時間だけ）**。

## 1. 概要（グループごと）

|グループ|L17 par|L17 serial|L26 par|L26 serial|L35 par|L35 serial|
|---|---|---|---|---|---|---|
|Gemma の層（NPU）|42.9（49.0）|43.4（47.6）|79.9（81.5）|80.4（81.7）|85.6（91.8）|92.0（105.0）|
|埋め込み・PLE の表引き（CPU・毎回）|6.3（12.5）|0.6（0.6）|0.7（0.8）|0.7（0.8）|6.9（14.3）|1.0（1.0）|
|RoPE・マスク（毎回作り直し）|7.2（8.3）|5.9（6.0）|7.2（9.0）|7.0（7.2）|6.7（7.4）|6.5（6.7）|
|表を開く・スケールを読む・器（毎回）|52.4（63.4）|41.8（48.7）|56.6（82.7）|51.2（53.4）|70.2（100.5）|51.4（61.6）|
|受け渡し（JSON・JNI）|27.1（35.4）|16.5（24.0）|23.7（30.3）|21.0（24.3）|28.8（34.9）|23.4（29.1）|
|トークン化・ヘッド・h の平均・規則|3.4（4.9）|2.4（3.0）|3.4（5.5）|3.2（4.3）|3.8（10.1）|3.6（6.7）|
|BERT（並行: 待った分／単独: 全部）|0.0（0.0）|32.7（43.9）|0.0（0.0）|60.6（73.8）|0.0（0.0）|89.4（111.2）|
|測っていない残り|0.1（0.2）|0.0（0.0）|0.1（0.1）|0.0（0.0）|0.1（2.9）|0.0（0.0）|
|**判定 1 回の全体（wall）**|**142.9**（160.0）|**147.5**（162.5）|**173.6**（205.6）|**222.4**（241.6）|**207.7**（240.6）|**271.6**（303.5）|
|行の数|300|300|300|300|300|300|

## 2. 部品ごと

|部品|L17 par|L17 serial|L26 par|L26 serial|L35 par|L35 serial|
|---|---|---|---|---|---|---|
|Gemma のトークン化（Java）|1.20（1.8）|0.89（1.1）|1.14（2.8）|1.13（1.4）|1.58（8.0）|1.43（4.6）|
|job の JSON を作る（Java）|3.32（11.0）|1.66（2.0）|2.86（10.4）|2.13（2.2）|4.97（9.3）|2.37（5.5）|
|JNI: 文字列の受け取り＋job の JSON を読む（C++）|0.06（0.1）|0.05（0.1）|0.07（0.2）|0.06（0.1）|0.07（0.2）|0.07（0.2）|
|毎回の describe・versions（C++）|0.30（0.4）|0.23（0.3）|0.33（0.9）|0.29（0.3）|0.33（0.8）|0.33（0.9）|
|埋め込みの表を開く＋スケールを読む（C++）|0.83（1.5）|0.56（0.6）|0.90（2.2）|0.69（0.7）|0.85（2.1）|0.76（2.0）|
|PLE の表を開く＋スケールを読む（C++・35 層ぶん）|50.85（61.5）|40.84（47.6）|54.86（79.6）|49.96（52.1）|67.92（97.6）|49.70（58.6）|
|入力の器を作る（C++）|0.24（0.3）|0.22（0.2）|0.19（0.2）|0.19（0.2）|0.20（0.2）|0.20（0.2）|
|RoPE の表を読んで写す（C++）|7.16（8.3）|5.86（6.0）|7.24（9.0）|7.04（7.2）|6.65（7.4）|6.54（6.7）|
|マスクを作る（C++）|0.00（0.0）|0.00（0.0）|0.00（0.0）|0.00（0.0）|0.00（0.0）|0.00（0.0）|
|埋め込みの表引き（C++・T=64 行）|0.21（4.0）|0.16（0.2）|0.16（0.2）|0.16（0.2）|0.24（4.7）|0.23（0.3）|
|PLE の表引き（C++・T=64 行 × 35 層）|5.11（9.4）|0.43（0.5）|0.56（0.6）|0.57（0.6）|5.81（11.0）|0.72（0.8）|
|NPU（ORT の Run）|42.93（49.0）|43.38（47.6）|79.94（81.5）|80.43（81.7）|85.65（91.8）|91.99（105.0）|
|Run の前後（入力の包み・出力の写し）|0.19（0.2）|0.16（0.2）|0.19（0.2）|0.19（0.3）|0.19（0.2）|0.19（0.2）|
|h の平均（C++）|0.09（0.2）|0.04（0.1）|0.06（0.1）|0.06（0.2）|0.13（0.1）|0.07（0.1）|
|h を JSON の配列にする（C++）|0.38（0.5）|0.16（0.3）|0.21（0.3）|0.21（0.4）|0.44（0.5）|0.26（0.3）|
|job の残り（結果の組み立てなど・C++）|13.68（17.1）|9.76（16.2）|10.64（13.6）|10.96（12.6）|14.82（16.3）|10.64（14.9）|
|結果を文字列にして Java へ返す（dump・JNI）|1.41（2.7）|0.88（1.9）|1.46（1.7）|1.50（1.6）|2.30（2.8）|1.54（2.8）|
|結果の JSON を読む（Java）|4.91（8.5）|3.75（5.7）|5.53（6.9）|5.52（6.7）|6.04（9.2）|6.04（7.3）|
|h を float にする（Java）|0.11（0.1）|0.08（0.1）|0.11（0.1）|0.10（0.1）|0.12（0.1）|0.12（0.1）|
|ヘッド（Java）|1.43（2.1）|1.19（1.4）|1.68（2.3）|1.66（2.2）|1.84（2.1）|1.87（2.2）|
|BERT を待った時間（並行のとき）|0.00（0.0）|0.00（0.0）|0.00（0.0）|0.00（0.0）|0.00（0.0）|0.00（0.0）|
|BERT（単独・先に動かしたとき）|0.00（0.0）|32.73（43.9）|0.00（0.0）|60.58（73.8）|0.00（0.0）|89.44（111.2）|
|規則・引数・検査（Java）|0.63（0.9）|0.28（0.6）|0.36（0.8）|0.32（0.7）|0.01（0.0）|0.02（0.0）|
|測っていない残り|0.12（0.2）|0.02（0.0）|0.12（0.1）|0.03（0.0）|0.13（2.9）|0.02（0.0）|

## 3. 参考

- L17 par: BERT（トークン化＋推論）56.3ms・Gemma 側（トークン化〜ヘッド）142.1ms・本物のトークン 42 個（T=64）・PC と同じ判定 300/300
- L17 serial: BERT（トークン化＋推論）32.7ms・Gemma 側（トークン化〜ヘッド）114.4ms・本物のトークン 42 個（T=64）・PC と同じ判定 300/300
- L26 par: BERT（トークン化＋推論）66.1ms・Gemma 側（トークン化〜ヘッド）173.2ms・本物のトークン 42 個（T=64）・PC と同じ判定 300/300
- L26 serial: BERT（トークン化＋推論）60.6ms・Gemma 側（トークン化〜ヘッド）162.3ms・本物のトークン 42 個（T=64）・PC と同じ判定 300/300
- L35 par: BERT（トークン化＋推論）83.6ms・Gemma 側（トークン化〜ヘッド）207.6ms・本物のトークン 42 個（T=64）
- L35 serial: BERT（トークン化＋推論）89.4ms・Gemma 側（トークン化〜ヘッド）180.9ms・本物のトークン 42 個（T=64）

## 4. QNN の計測（NPU の中）

### 層 17


かたまりごとのサイクル（30 回の中央値・全体 83.0M サイクルに対する割合）:

|かたまり|中身|M サイクル|割合|
|---|---|---|---|
|A|入口＋層 0〜8|51.8|62.4%|
|c1|層 9〜16＋最後の正規化|31.3|37.7%|
|（A のうち入口）|埋め込みの変換・PLE の投影（35 層ぶん）と正規化・ple_tok の変換（OpId ≤ 97）|2.61|3.1%|

計測を付けたときの時間（中央値・ms。計測の分だけ遅い）: Accelerator (execute) time 35.2・QNN accelerator (execute) time 103.8・RPC (execute) time 106.2・QNN (execute) time 111.5

ノードごと（サイクルの合計が多い順）:

- `qnnprof_L17.csv`: 51,965 行・4278KiB
- 列: Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event Level, Event Identifier

|Event Identifier|単位|回数|中央値|合計|
|---|---|---|---|---|
|Accelerator (execute) time (cycles)|CYCLES|30|83,040,806.0|2,477,031,288|
|c1_node_linear_65:OpId_4931 (cycles)|CYCLES|30|1,208,379.5|35,038,775|
|c1_node_linear_58:OpId_4707 (cycles)|CYCLES|30|1,175,768.0|33,934,826|
|c1_node_add_83:OpId_4175 (cycles)|CYCLES|30|907,288.0|27,241,000|
|rms_norm_node___511906:OpId_38 (cycles)|CYCLES|30|626,338.0|18,830,054|
|c1_node_linear_33:OpId_3892 (cycles)|CYCLES|30|609,024.5|17,378,747|
|c1_node_linear_24:OpId_3597 (cycles)|CYCLES|30|597,990.0|17,259,580|
|c1_node_linear_15:OpId_3303 (cycles)|CYCLES|30|604,216.5|17,201,951|
|node_linear_79:OpId_2673 (cycles)|CYCLES|30|606,711.5|16,820,722|
|c1_node_linear_51:OpId_4482 (cycles)|CYCLES|30|571,929.0|16,363,487|
|c1_node_linear_6:OpId_3006 (cycles)|CYCLES|30|587,717.5|16,293,548|
|c1_node_linear_42:OpId_4187 (cycles)|CYCLES|30|555,723.0|15,986,483|
|node_linear_70:OpId_2378 (cycles)|CYCLES|30|582,368.0|15,488,348|
|c1_node_linear_56:OpId_4665 (cycles)|CYCLES|30|514,133.5|15,408,464|
|node_linear_43:OpId_1499 (cycles)|CYCLES|30|490,175.0|15,150,947|
|node_linear_61:OpId_2083 (cycles)|CYCLES|30|544,838.0|15,143,154|
|c1_node_linear_64:OpId_4876 (cycles)|CYCLES|30|501,265.0|15,097,905|
|c1_node_linear_57:OpId_4652 (cycles)|CYCLES|30|502,749.5|15,086,791|
|c1_node_linear_63:OpId_4889 (cycles)|CYCLES|30|490,202.5|14,735,341|
|node_linear_52:OpId_1792 (cycles)|CYCLES|30|504,128.0|14,717,299|
|node_linear:OpId_31 (cycles)|CYCLES|30|463,382.0|14,574,757|
|node_linear_34:OpId_1196 (cycles)|CYCLES|30|417,045.0|13,925,094|
|node_linear_16:OpId_605 (cycles)|CYCLES|30|431,671.0|13,860,550|
|node_linear_25:OpId_901 (cycles)|CYCLES|30|394,832.0|12,893,755|
|rms_norm_c1_node___629849:OpId_2844 (cycles)|CYCLES|30|417,522.5|12,555,533|
|rms_norm_c1_node___698624:OpId_4324 (cycles)|CYCLES|30|415,076.0|12,481,714|
|rms_norm_node___575056:OpId_1299 (cycles)|CYCLES|30|413,298.5|12,410,306|
|node_linear_7:OpId_310 (cycles)|CYCLES|30|383,420.0|12,135,455|
|QNN_DATATYPE_FLOAT_32_converted_ple_tok_QNN_DATATYPE_UFIXED_POINT_16:OpId_97 (cy|CYCLES|30|382,235.5|11,693,704|
|node_mul_148:OpId_1340 (cycles)|CYCLES|30|383,041.0|11,514,318|
|c1_node_linear_48:OpId_4401 (cycles)|CYCLES|30|387,907.0|11,315,161|
|c1_node_tanh_12:OpId_4688 (cycles)|CYCLES|30|370,705.0|11,149,302|
|node_linear_40:OpId_1418 (cycles)|CYCLES|30|376,500.0|11,063,964|
|node_mul_149:OpId_1385 (cycles)|CYCLES|30|362,852.0|10,906,675|
|c1_node_tanh_14:OpId_4912 (cycles)|CYCLES|30|359,934.5|10,815,610|
|c1_node_linear_3:OpId_2921 (cycles)|CYCLES|30|384,643.0|10,803,792|
|node_linear_23:OpId_846 (cycles)|CYCLES|30|304,702.0|9,255,905|
|node_linear_33:OpId_1154 (cycles)|CYCLES|30|296,196.5|8,891,709|
|c1_node_linear_40:OpId_4145 (cycles)|CYCLES|30|294,202.5|8,844,811|
|c1_node_linear_14:OpId_3248 (cycles)|CYCLES|30|293,944.5|8,812,739|

- 計測を付けたときの NPU（ORT の Run）中央値 121.0ms（30 行）

### 層 26


かたまりごとのサイクル（30 回の中央値・全体 145.9M サイクルに対する割合）:

|かたまり|中身|M サイクル|割合|
|---|---|---|---|
|A|入口＋層 0〜8|65.4|44.8%|
|c1|層 9〜17|37.7|25.8%|
|c2|層 18〜25＋最後の正規化|42.8|29.3%|
|（A のうち入口）|埋め込みの変換・PLE の投影（35 層ぶん）と正規化・ple_tok の変換（OpId ≤ 97）|3.45|2.4%|

計測を付けたときの時間（中央値・ms。計測の分だけ遅い）: Accelerator (execute) time 59.5・QNN accelerator (execute) time 138.3・RPC (execute) time 140.7・QNN (execute) time 146.7

ノードごと（サイクルの合計が多い順）:

- `qnnprof_L26.csv`: 73,535 行・6069KiB
- 列: Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event Level, Event Identifier

|Event Identifier|単位|回数|中央値|合計|
|---|---|---|---|---|
|Accelerator (execute) time (cycles)|CYCLES|30|145,889,333.0|4,409,319,513|
|c2_node_linear_32:OpId_6338 (cycles)|CYCLES|30|1,200,043.5|37,454,955|
|c2_node_linear_46:OpId_6786 (cycles)|CYCLES|30|1,168,206.0|36,622,908|
|c1_node_linear_65:OpId_4945 (cycles)|CYCLES|30|1,174,199.0|36,330,432|
|c1_node_linear_72:OpId_5169 (cycles)|CYCLES|30|1,168,148.0|36,286,606|
|c2_node_linear_4:OpId_5439 (cycles)|CYCLES|30|1,148,554.0|35,794,332|
|rms_norm_node___140411:OpId_38 (cycles)|CYCLES|30|1,106,025.5|33,189,493|
|c2_node_linear_11:OpId_5665 (cycles)|CYCLES|30|1,032,293.5|32,678,413|
|c2_node_linear_18:OpId_5889 (cycles)|CYCLES|30|970,617.0|31,477,606|
|c2_node_linear_39:OpId_6562 (cycles)|CYCLES|30|922,923.0|30,668,880|
|c2_node_linear_25:OpId_6114 (cycles)|CYCLES|30|926,158.0|30,469,756|
|c2_node_linear_53:OpId_7010 (cycles)|CYCLES|30|892,731.5|29,618,815|
|c1_node_linear_58:OpId_4721 (cycles)|CYCLES|30|877,027.5|28,875,082|
|c1_node_add_83:OpId_4184 (cycles)|CYCLES|30|838,828.5|25,142,798|
|QNN_DATATYPE_FLOAT_32_converted_ple_tok_QNN_DATATYPE_UFIXED_POINT_16:OpId_97 (cy|CYCLES|30|630,068.5|19,480,941|
|c1_node_linear_51:OpId_4496 (cycles)|CYCLES|30|612,996.0|18,779,505|
|c1_node_linear_15:OpId_3307 (cycles)|CYCLES|30|597,974.0|18,119,665|
|c1_node_linear_33:OpId_3896 (cycles)|CYCLES|30|586,396.5|17,979,435|
|c1_node_linear_24:OpId_3601 (cycles)|CYCLES|30|593,655.0|17,862,690|
|node_linear_43:OpId_1499 (cycles)|CYCLES|30|586,560.0|17,593,417|
|c2_node_linear_24:OpId_6072 (cycles)|CYCLES|30|583,559.0|17,472,892|
|node_linear_7:OpId_310 (cycles)|CYCLES|30|570,200.0|17,444,260|
|node_linear_52:OpId_1792 (cycles)|CYCLES|30|562,957.5|17,156,289|
|c2_node_linear_17:OpId_5847 (cycles)|CYCLES|30|568,439.5|17,064,485|
|c2_node_linear_3:OpId_5393 (cycles)|CYCLES|30|562,357.5|17,034,199|
|c2_node_linear_31:OpId_6296 (cycles)|CYCLES|30|566,481.0|16,998,761|
|c2_node_linear_10:OpId_5623 (cycles)|CYCLES|30|564,159.0|16,996,989|
|c2_node_linear_45:OpId_6744 (cycles)|CYCLES|30|564,057.5|16,940,726|
|c1_node_linear_64:OpId_4890 (cycles)|CYCLES|30|560,465.5|16,909,970|
|c1_node_linear_57:OpId_4666 (cycles)|CYCLES|30|563,004.5|16,889,472|
|c2_node_linear_38:OpId_6520 (cycles)|CYCLES|30|561,846.5|16,852,280|
|c1_node_linear_71:OpId_5114 (cycles)|CYCLES|30|559,837.5|16,791,911|
|c2_node_linear_52:OpId_6968 (cycles)|CYCLES|30|558,418.5|16,757,102|
|c1_node_linear_70:OpId_5127 (cycles)|CYCLES|30|552,575.0|16,727,236|
|c2_node_linear_44:OpId_6731 (cycles)|CYCLES|30|548,931.0|16,691,523|
|c2_node_linear_30:OpId_6283 (cycles)|CYCLES|30|551,398.5|16,648,416|
|node_linear_79:OpId_2673 (cycles)|CYCLES|30|556,031.5|16,633,798|
|c2_node_linear_2:OpId_5380 (cycles)|CYCLES|30|544,987.5|16,553,655|
|c2_node_linear_23:OpId_6059 (cycles)|CYCLES|30|539,532.5|16,404,478|
|node_linear_34:OpId_1196 (cycles)|CYCLES|30|528,091.5|16,354,981|

- 計測を付けたときの NPU（ORT の Run）中央値 163.4ms（30 行）

### 層 35


かたまりごとのサイクル（30 回の中央値・全体 207.1M サイクルに対する割合）:

|かたまり|中身|M サイクル|割合|
|---|---|---|---|
|A|入口＋層 0〜8|75.7|36.6%|
|c1|層 9〜17|39.3|19.0%|
|c2|かたまり 3|27.3|13.2%|
|c3|かたまり 4|32.4|15.6%|
|c4|かたまり 5（最後の正規化まで）|33.1|16.0%|
|（A のうち入口）|埋め込みの変換・PLE の投影（35 層ぶん）と正規化・ple_tok の変換（OpId ≤ 97）|3.56|1.7%|

計測を付けたときの時間（中央値・ms。計測の分だけ遅い）: Accelerator (execute) time 98.6・QNN accelerator (execute) time 202.1・RPC (execute) time 204.0・QNN (execute) time 217.7

ノードごと（サイクルの合計が多い順）:

- `qnnprof_L35.csv`: 95,345 行・7877KiB
- 列: Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event Level, Event Identifier

|Event Identifier|単位|回数|中央値|合計|
|---|---|---|---|---|
|Accelerator (execute) time (cycles)|CYCLES|30|207,062,749.5|6,334,745,556|
|c3_node_linear_32:OpId_7497 (cycles)|CYCLES|30|1,252,002.5|41,541,613|
|c3_node_linear_11:OpId_6824 (cycles)|CYCLES|30|1,260,143.0|41,337,464|
|c2_node_linear_18:OpId_5877 (cycles)|CYCLES|30|1,259,607.0|40,892,484|
|c2_node_linear_25:OpId_6102 (cycles)|CYCLES|30|1,247,441.5|40,263,598|
|c1_node_linear_65:OpId_4935 (cycles)|CYCLES|30|1,216,681.5|39,861,914|
|c4_node_linear_11:OpId_8219 (cycles)|CYCLES|30|1,195,702.0|38,924,569|
|c3_node_linear_39:OpId_7721 (cycles)|CYCLES|30|1,111,301.5|37,201,165|
|c4_node_linear_32:OpId_8892 (cycles)|CYCLES|30|1,136,955.0|36,782,736|
|c4_node_linear_4:OpId_7993 (cycles)|CYCLES|30|1,108,976.5|36,769,734|
|c3_node_linear_4:OpId_6598 (cycles)|CYCLES|30|1,129,558.0|36,450,583|
|c2_node_linear_32:OpId_6326 (cycles)|CYCLES|30|1,166,905.0|36,337,948|
|c3_node_linear_25:OpId_7273 (cycles)|CYCLES|30|1,086,048.5|36,205,013|
|c1_node_linear_58:OpId_4711 (cycles)|CYCLES|30|1,085,423.0|34,959,253|
|c4_node_linear_39:OpId_9116 (cycles)|CYCLES|30|1,090,774.5|34,444,615|
|c2_node_linear_11:OpId_5653 (cycles)|CYCLES|30|1,125,616.5|34,143,463|
|c3_node_linear_18:OpId_7048 (cycles)|CYCLES|30|1,102,031.5|32,944,667|
|rms_norm_node___244961:OpId_38 (cycles)|CYCLES|30|1,097,696.5|32,932,339|
|c4_node_linear_18:OpId_8443 (cycles)|CYCLES|30|1,203,000.5|32,426,010|
|c4_node_linear_25:OpId_8668 (cycles)|CYCLES|30|1,129,332.0|32,337,007|
|c1_node_linear_72:OpId_5159 (cycles)|CYCLES|30|977,544.5|32,177,394|
|c2_node_linear_4:OpId_5427 (cycles)|CYCLES|30|958,439.5|30,982,788|
|c1_node_linear_56:OpId_4669 (cycles)|CYCLES|30|829,325.5|25,797,368|
|c1_node_add_83:OpId_4179 (cycles)|CYCLES|30|851,145.0|25,521,669|
|c1_node_linear_57:OpId_4656 (cycles)|CYCLES|30|787,633.5|24,401,469|
|c1_node_linear_42:OpId_4191 (cycles)|CYCLES|30|635,414.0|20,594,498|
|c1_node_linear_24:OpId_3601 (cycles)|CYCLES|30|635,441.0|20,552,241|
|c1_node_linear_33:OpId_3896 (cycles)|CYCLES|30|630,410.0|20,375,196|
|QNN_DATATYPE_FLOAT_32_converted_ple_tok_QNN_DATATYPE_UFIXED_POINT_16:OpId_97 (cy|CYCLES|30|676,041.0|20,227,561|
|node_linear_79:OpId_2673 (cycles)|CYCLES|30|619,161.0|20,071,956|
|node_linear_61:OpId_2083 (cycles)|CYCLES|30|611,743.0|19,986,006|
|c1_node_linear_15:OpId_3307 (cycles)|CYCLES|30|609,888.5|19,800,757|
|c1_node_linear_51:OpId_4486 (cycles)|CYCLES|30|606,208.0|19,789,738|
|node_linear_52:OpId_1792 (cycles)|CYCLES|30|604,673.0|19,689,536|
|node_linear_43:OpId_1499 (cycles)|CYCLES|30|592,439.0|19,162,203|
|node_linear_16:OpId_605 (cycles)|CYCLES|30|582,153.5|18,510,546|
|node_linear:OpId_31 (cycles)|CYCLES|30|562,822.5|18,135,640|
|node_linear_7:OpId_310 (cycles)|CYCLES|30|583,507.5|18,135,288|
|c2_node_linear_2:OpId_5368 (cycles)|CYCLES|30|591,641.5|17,917,723|
|c4_node_linear_10:OpId_8177 (cycles)|CYCLES|30|583,587.5|17,858,574|

- 計測を付けたときの NPU（ORT の Run）中央値 239.6ms（30 行）

