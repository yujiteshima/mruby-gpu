# bench/ — ベンチマーク

mruby-gpu のパフォーマンス特性を計測・比較するスクリプト。

## スクリプト一覧

| ファイル | 目的 |
|---|---|
| [basic.rb](basic.rb) | `GPU.add` の Transfer / Compute / Readback の内訳計測(10k 要素) |
| [scaling.rb](scaling.rb) | 要素数を変えた `GPU.add` のスケーリング特性 |
| [packing.rb](packing.rb) | 学習の packing 効果: batch=1 vs batch=64(forward + backward + SGD) |
| [packing_inference.rb](packing_inference.rb) | 推論の packing 効果: batch=1 vs batch=64(forward のみ) |

## 実行例

```bash
MRUBY=/path/to/mruby

# 単発計測
$MRUBY bench/basic.rb

# 要素数スケーリング
$MRUBY bench/scaling.rb

# packing Before/After(学習)
$MRUBY bench/packing.rb

# packing Before/After(推論)
$MRUBY bench/packing_inference.rb
```

## Packing ベンチの仕組み

`packing.rb` / `packing_inference.rb` は、MNIST 学習/推論のループを

- **Before**: batch=1 パターン(1 サンプルごとに upload / compute / readback)
- **After**: batch=64 パターン(64 サンプル束ねて 1 回ずつ)

で少量サンプル(200)実行して `N_TRAIN = 60000` または `N_TEST = 10000` に
線形外挿します。合成データで自己完結するので MNIST 実データ不要。

## 関連ドキュメント

実機での詳細分析:
- [../docs/PERFORMANCE.md](../docs/PERFORMANCE.md) — ボトルネック分析と高速化アイデア
- [../docs/GPU_VS_CPU_ANALYSIS.md](../docs/GPU_VS_CPU_ANALYSIS.md) — GPU vs CPU 推論の実測比較
