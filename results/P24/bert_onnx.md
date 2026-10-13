BERT（案 F・seed 0。`work/p20/bert_F/bert_s0`）を ONNX にしたもの（`work/p24/device/router/bert/model.onnx`）の確かめ。行は評価セット v6・v5 の 1200 行

|項目|値|
|---|---|
|書き出し|TorchScript の書き出し（opset 17）・opset 17|
|大きさ|424MiB|
|ONNX ↔ torch（同じ入力・CPU）の確率の差の最大|5.40e-06|
|ONNX ↔ 保存済みの確率（P21。GPU・まとめて計算）の差の最大|2.25e-06|
|閾値（断る 0.9943・フラグ 0.5）をまたいだ行|0|
