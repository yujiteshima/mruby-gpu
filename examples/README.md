# examples/ — API の使い方

mruby-gpu の各 API の**典型的な使い方**を示す短いスクリプト集。

## スクリプト一覧

| ファイル | 内容 |
|---|---|
| [gpu_add.rb](gpu_add.rb) | 最小例: `GPU.add` で 3 要素ベクトル加算 |
| [mnist.rb](mnist.rb) | MNIST 推論 10 枚 (batch=1、1 枚ずつ CPU 往復) |
| [mnist_inference.rb](mnist_inference.rb) | MNIST 推論 10000 枚 (batched、精度・スループット計測) |
| [train.rb](train.rb) | MNIST 学習 (batch=1 SGD、2 層 MLP) |
| [train_minibatch.rb](train_minibatch.rb) | MNIST 学習 (ミニバッチ SGD、引数で batch size 指定) |

## データ準備

学習・推論スクリプトは `data/` 以下の MNIST 変換済みファイルを参照します:

```bash
# MNIST IDX ファイルをダウンロード(初回のみ)
bash tools/download_mnist.sh

# float32 .bin に変換(CRuby で実行)
ruby tools/prepare_mnist.rb
```

生成されるファイル:
- `data/train_images.bin` (60000 × 784 float32)
- `data/train_labels.bin` (60000 float32)
- `data/test_images.bin` (10000 × 784 float32)
- `data/test_labels.bin` (10000 float32)
- `data/test_img_0.bin` 〜 `data/test_img_9.bin`(`mnist.rb` 用)

## 学習 → 推論の流れ

```bash
MRUBY=/path/to/mruby

# 1. 学習(重みを weights/ に保存)
$MRUBY examples/train_minibatch.rb 64

# 2. 推論(weights/ の重みで 10000 枚分類)
$MRUBY examples/mnist_inference.rb 64
```

## 関連

- 単純な `GPU.add` / `Camera` などの API 仕様は [../README.md](../README.md) の API リファレンス節を参照
- ベンチマーク(Before / After 比較など)は [../bench/README.md](../bench/README.md)
