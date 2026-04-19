# GPU vs CPU 推論パフォーマンス分析

Raspberry Pi 5 上で UltraFace-slim 顔検出モデルの GPU (Vulkan V3D) と CPU (ARM NEON) の性能を比較し、なぜ CPU が 14 倍速いのかを調査した記録。

---

## 実測ベンチマーク

### 推論バックエンドの比較

| 構成 | 推論時間 | 備考 |
|------|---------|------|
| GPU default | 164.4 ms | V3D 7.1 Vulkan Compute |
| GPU + FP16 all | 159.6 ms | fp16 packed/storage/arithmetic 有効 |
| GPU + BF16 storage | 161.7 ms | bf16 storage 有効 |
| GPU + FP16 + pack + light | 160.5 ms | 全オプション有効 |
| **CPU NEON (baseline)** | **6.1 ms** | ARM Cortex-A76 NEON |
| CPU + FP16 | 5.9 ms | NEON + fp16 |

> GPU の全最適化オプションを有効にしても **160ms → 159ms（1% 未満の改善）**。  
> オーバーヘッドではなく V3D の演算速度そのものがボトルネック。

### モード別の実効速度

| モード | GPU (Vulkan V3D) | CPU (ARM NEON) | 改善倍率 |
|--------|-----------------|----------------|---------|
| fast (全体 1 パス) | 165 ms / 6 FPS | **12 ms / 82 FPS** | **14×** |
| count (全体 + 5 タイル) | 971 ms / 1 FPS | **49 ms / 20 FPS** | **20×** |

---

## 調査: command buffer は往復していなかった

### 初期仮説

> 100 層のニューラルネットワークで毎層ごとに CPU↔GPU の submit-wait が走り、  
> その往復オーバーヘッドが支配的になっている。

### NCNN ソースコード調査の結果

NCNN の `command.cpp` を調査したところ、**複数レイヤーの dispatch は 1 本の command buffer に記録されている**。

```
実際の動作:
  [record layer1][record layer2]...[record layer100] → [submit 1回] → [wait 1回]

初期仮説（誤り）:
  [record+submit+wait layer1] → [record+submit+wait layer2] → ...
```

#### submit の分割は threshold ベース

NCNN は `pending_dispatch_total` が閾値を超えると command buffer を途中で submit する。  
閾値はデバイスの `r-score` で決定される。

| デバイススコア | Threshold | 用途 |
|---|---|---|
| > 75 | 8M | 高性能 Discrete GPU |
| 50-75 | 4M | 中性能 GPU |
| 15-50 | 1M | 統合 GPU |
| 10-15 | 256K | 低性能デバイス |
| < 10 | **32K** | **超低性能デバイス** |

V3D の **r-score は 7**（最低カテゴリ）。閾値 32K は小さいが、UltraFace-slim の各層の dispatch サイズも小さいため、実際にはほぼ 1 回の submit でまとまっている可能性が高い。

#### 検証: FP16 等の全オプションを有効にしても改善しない

転送量を半減する FP16 storage、メモリ再利用の lightmode、パッキング最適化をすべて有効にしても **1% 未満の改善**。  
これは往復オーバーヘッドが主因ではないことの証拠。

---

## 真の原因: V3D の演算コア自体が ARM NEON に負けている

### ハードウェア比較

| 項目 | V3D (VideoCore VII) | Cortex-A76 (NEON) |
|------|-------------------|-------------------|
| コア数 | 12 QPU | 4 CPU コア |
| クロック | ~800 MHz | 2.4 GHz (**3 倍**) |
| SIMD 幅 | 16-way float | 128bit = 4×float ×4 コア |
| キャッシュ | なし（VRAM 経由） | **L1 64KB + L2 512KB** |
| 3×3 Conv 最適化 | 汎用 compute shader | **Winograd 変換 + NEON 手書きアセンブリ** |
| メモリ帯域 | 共有メモリ（バス競合） | **CPU に直結** |
| 設計目的 | **グラフィックス描画** | **汎用計算 + モバイル ML** |

### なぜ CPU が速いのか: 3 つの要因

#### 1. モデルが L2 キャッシュに収まる

UltraFace-slim はモデルサイズ **1 MB**。  
Cortex-A76 の L2 キャッシュは **512 KB × 4 コア = 2 MB**。  
モデル全体がキャッシュに載り、メモリアクセスが極めて高速。

GPU にはこのレベルのキャッシュがなく、毎回 VRAM（共有メモリ）へアクセスする。

```
CPU:  [コア] ← 1ns → [L1 64KB] ← 5ns → [L2 512KB] ← モデル全体がここに入る
GPU:  [QPU]  ← 50ns → [共有メモリ]  ← 毎回ここまでアクセス
```

#### 2. NCNN の ARM NEON 最適化が極めて強力

NCNN は ARM NEON 向けに**手書きアセンブリレベルの最適化**を持っている。

- 3×3 Convolution に Winograd 変換を適用（演算量 2.25 倍削減）
- NEON の `vfma` (Fused Multiply-Add) 命令を直接使用
- メモリアクセスパターンを NEON のレジスタ幅に合わせて最適化

対して GPU パスは汎用の Vulkan Compute Shader で、V3D 固有の最適化はない。

#### 3. 各層のデータが小さすぎて GPU の並列性を活かせない

UltraFace-slim の最初の Convolution 層:

```
入力:  320 × 240 × 3 = 230,400 要素
出力:  160 × 120 × 16 = 307,200 要素
```

V3D の 12 QPU で分割すると 1 QPU あたり約 25,600 要素。  
QPU の起動・同期コストに対してデータが少なく、並列効率が低い。

比較: `GPU.add` で 100 万要素の場合は 1 QPU あたり約 83,000 要素で、十分な並列効率が出る。

---

## GPU が活きるケースと活きないケース

| 処理 | dispatch 回数 | データ量/回 | GPU vs CPU |
|------|-------------|------------|------------|
| GPU.add (100 万要素) | **1 回** | **4 MB** | **GPU 5.8ms ≪ CPU ~20ms** |
| YUYV→RGB 変換 (640×480) | **1 回** | **600 KB** | **GPU 向き** |
| 行列積 (大きな行列) | **1 回** | **数 MB** | **GPU 向き** |
| UltraFace 推論 (100 層) | **100 回** | **数 KB/回** | CPU 6ms ≪ GPU 160ms |

### 法則

```
GPU が速い条件:
  ✅ 1 回の dispatch で大量データを並列処理する
  ✅ データがキャッシュに収まらないほど大きい
  ✅ 単純な演算の繰り返し（加算、行列積、画像変換）

GPU が遅い条件:
  ❌ 多段の依存処理（100 層のニューラルネットワーク）
  ❌ 各段のデータが小さい（数 KB〜数十 KB）
  ❌ CPU 側に NEON 等の特化最適化がある
```

---

## 結論: 適材適所の使い分け

### デモのストーリー

> 「mruby から GPU も CPU も使い分けられる。それが mruby-gpu の強み」

1. **GPU.add で 100 万要素の加算** → GPU が圧勝（大量データ × 単純演算）
2. **顔認識は CPU NEON が最速** → 適材適所（小モデル × L2 キャッシュ × NEON 最適化）
3. **YUYV→RGB 変換を GPU Compute Shader で** → 画像処理は GPU 向き（1 dispatch × 大データ）
4. **mruby スクリプト側で `use_gpu: true/false` を切り替えるだけ** → 1 行の変更で最適な実行パスを選択

### demo/face_demo.rb での使い分け

```ruby
# GPU が得意: 画像前処理 (1 dispatch × 大データ)
GPU.init("shader")
rgb = GPU.yuv_to_rgb(frame)    # ← GPU Compute Shader

# CPU が得意: ML 推論 (100 層 × 小データ)
detector = FaceDetector.new("models/ultraface-slim", use_gpu: false)  # ← CPU NEON
faces = detector.detect(frame, 640, 480)
```

---

## NCNN Vulkan 最適化オプション一覧（参考）

NCNN の `ncnn::Option` で設定可能な Vulkan 関連オプション。  
V3D では効果が限定的だったが、他の GPU（Mali, Adreno, Discrete GPU）では有効な場合がある。

| オプション | デフォルト | 効果 |
|---|---|---|
| `use_vulkan_compute` | false | Vulkan Compute 推論を有効化 |
| `use_packing_layout` | true | SIMD-friendly な packed layout |
| `use_fp16_packed` | - | FP16 packed format |
| `use_fp16_storage` | - | FP16 storage buffer（転送量 50% 削減） |
| `use_fp16_arithmetic` | - | FP16 で演算実行 |
| `use_bf16_storage` | - | BF16 storage buffer |
| `use_int8_storage` | - | INT8 storage（メモリ 1/4） |
| `use_int8_arithmetic` | - | INT8 演算（量子化モデル向け） |
| `lightmode` | false | 中間ブロブの即時解放 |
| `use_shader_local_memory` | - | Shared memory 活用 |
| `use_cooperative_matrix` | - | Tensor Core 活用（NVIDIA/ARM） |
| `use_subgroup_ops` | - | Subgroup 最適化 |

### Pipeline Cache

NCNN は SPIR-V コードと specialization constant から hash を生成し、pipeline をキャッシュする。  
初回実行時のみ Vulkan Shader コンパイルが発生（100-500ms）。2 回目以降はキャッシュヒットで < 1ms。

```
初回フレーム:  200ms（Vulkan shader compile + 推論）
2回目以降:     160ms（推論のみ）
```

---

## 測定環境

- **Board**: Raspberry Pi 5 (4GB)
- **GPU**: VideoCore VII (V3D 7.1.10.2) — r-score 7
- **CPU**: ARM Cortex-A76 × 4 @ 2.4 GHz
- **OS**: Ubuntu 24.04 arm64
- **NCNN**: v1.0.20260331 (Vulkan enabled)
- **Model**: UltraFace-slim (slim_320) — 1 MB, 100 layers, 4420 anchors
- **Camera**: Logitech HD Pro Webcam C920 (640×480 YUYV @ 30fps)

---

## ソースコード参照

| 機能 | ファイル | 行番号 |
|---|---|---|
| Command buffer 統合 | ncnn/src/command.cpp | 1555-1577 |
| Submit threshold 判定 | ncnn/src/net.cpp | 252-273 |
| Pipeline Cache | ncnn/src/pipelinecache.cpp | 130-160 |
| V3D デバイススコア | ncnn/src/gpu.cpp | スコア算出ロジック |
| NEON Conv 最適化 | ncnn/src/layer/arm/convolution_arm.cpp | Winograd 実装 |
