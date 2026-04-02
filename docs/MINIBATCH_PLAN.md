# MNIST ミニバッチ化計画

現在の MNIST トレーニング（SGD, バッチサイズ 1）を ミニバッチ対応に改善し、GPU の恩恵を引き出す計画。

---

## 現状の問題

### 1 サンプルずつ処理している

```ruby
# 現在の train.rb（バッチサイズ 1）
N_TRAIN.times do |i|
  x = GPU.load("data/train_images.bin", i * INPUT, INPUT)  # 784 要素
  z1 = GPU.matmul(w1, x, 128, 784, 1)                      # 128×784×1
  ...
end
```

### GPU dispatch が小さすぎる

| 演算 | 行列サイズ | 要素数 | 時間 |
|------|----------|--------|------|
| matmul W1@x | 128 × 784 × **1** | 100,352 | 5.9 ms |
| add z1+b1 | 128 | 128 | 0.2 ms |
| 1 sample 合計 | 17 dispatch | | 9.9 ms |
| **1 epoch** | **17 × 60,000 = 1,020,000 dispatch** | | **≈ 10 分** |

GPU は 100 万要素の加算を 5ms で処理できるのに、128 要素の add に 0.2ms（dispatch 固定コスト）かかる。

---

## ミニバッチ化の効果（予測）

バッチサイズ B で行列を横に束ねる:

```
現在:   W1 (128×784) @ x (784×1)   → z (128×1)     要素数: 100K
ミニバッチ: W1 (128×784) @ X (784×B) → Z (128×B)   要素数: 100K × B
```

| バッチサイズ B | matmul 要素数 | add 要素数 | dispatch/epoch | 予想速度 |
|---------------|-------------|-----------|----------------|---------|
| 1 (現状) | 100K | 128 | 1,020,000 | ≈ 10 分/epoch |
| **32** | 3.2M | 4,096 | 31,875 | **≈ 1〜2 分/epoch** |
| **64** | 6.4M | 8,192 | 15,937 | **≈ 30秒〜1 分/epoch** |
| 128 | 12.8M | 16,384 | 7,968 | ≈ 20〜30 秒/epoch |

### なぜ速くなるか

1. **dispatch 回数が 1/B に減る** → 固定コスト (0.2ms/回) の削減
2. **行列が大きくなり GPU の 12 QPU を活用できる** → 並列効率向上
3. **メモリ I/O がバッチでまとまる** → 帯域効率向上

---

## 実装計画

### Phase 1: データローダーのバッチ化

現在は `GPU.load` で 1 サンプルずつ読み込んでいる。  
バッチ分のデータを一度に GPU バッファに載せる。

```ruby
# 変更前
x = GPU.load("data/train_images.bin", i * INPUT, INPUT)  # 784 floats

# 変更後
# B サンプル分を一度に読み込み (784 × B floats)
x_batch = GPU.load("data/train_images.bin", batch_start * INPUT, INPUT * B)
```

**必要な変更**: なし（既存の `GPU.load` で offset と count を指定可能）

### Phase 2: Forward パスのバッチ化

```ruby
# 変更前: matmul(W1, x, 128, 784, 1) → 128×1
# 変更後: matmul(W1, X, 128, 784, B) → 128×B

B = 64
z1_batch = GPU.matmul(w1, x_batch, HIDDEN, INPUT, B)     # 128 × B
```

**bias 加算の問題**: 現在 `GPU.add(z1, b1)` は同サイズのバッファ同士。  
バッチの場合 z1 は 128×B だが b1 は 128×1 → **ブロードキャスト add が必要**。

**必要な変更**:
- `GPU.add_broadcast(matrix, bias, rows, cols)` を追加
  - matrix: rows × cols の行列バッファ
  - bias: rows × 1 のベクトル
  - 各列に bias を加算する compute shader

### Phase 3: ReLU のバッチ化

```ruby
# 変更前: GPU.relu(h_pre)        → 128 要素
# 変更後: GPU.relu(h_pre_batch)  → 128 × B 要素
```

**必要な変更**: なし（既存の `GPU.relu` は要素数に依存しない汎用実装）

### Phase 4: Softmax + Loss のバッチ化

現在 Ruby 側で 10 要素の softmax を計算。バッチ化すると 10 × B 要素。

```ruby
# 変更前（Ruby 側）
scores = o.head(CLASSES)
probs = softmax(scores)

# 変更後（2つの選択肢）
# A. Ruby 側でループ（B=64 なら 64 回の 10 要素 softmax → 十分高速）
# B. GPU compute shader で softmax を実装
```

**推奨**: 選択肢 A（Ruby 側ループ）。10 要素 × 64 回は Ruby でも < 1ms。

### Phase 5: Backward パスのバッチ化

```ruby
# grad_w2 = grad_o @ h^T → 現在: (10×1) @ (1×128) = 10×128
# バッチ: 各サンプルの勾配を平均する

# 変更後:
# grad_O: 10 × B,  H: 128 × B
# grad_W2 = (1/B) * grad_O @ H^T → (10×B) @ (B×128) = 10×128
grad_w2 = GPU.matmul_nt(grad_o_batch, h_batch, CLASSES, B, HIDDEN)
grad_w2 = GPU.scale(grad_w2, 1.0 / B)
```

**必要な変更**: なし（既存の `matmul_nt` で M=CLASSES, K=B, N=HIDDEN を指定）

### Phase 6: SGD 更新

変更なし。バッチ平均済みの勾配を使って既存コードがそのまま動く。

---

## 必要な新規実装まとめ

| 項目 | 種類 | 難易度 |
|------|------|--------|
| `GPU.add_broadcast(mat, vec, rows, cols)` | C + compute shader | 中 |
| `add_broadcast.comp` シェーダー | GLSL | 低 |
| ラベルのバッチ読み込み | Ruby | 低 |
| train.rb のループ構造変更 | Ruby | 中 |

**新規 C 実装は `add_broadcast` の 1 メソッドのみ。** 他はすべて既存 API の組み合わせで実現可能。

---

## add_broadcast.comp シェーダー設計

```glsl
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly  buffer MatBuf  { float mat[]; };  // rows × cols
layout(binding = 1) readonly  buffer BiasBuf { float bias[]; }; // rows
layout(binding = 2) writeonly buffer OutBuf  { float out_[]; }; // rows × cols

layout(push_constant) uniform Params { uint rows; uint cols; };

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= rows * cols) return;
  uint row = idx % rows;
  out_[idx] = mat[idx] + bias[row];
}
```

---

## 期待されるパフォーマンス

| 構成 | 1 epoch | 5 epochs |
|------|---------|----------|
| 現状 (B=1, GPU) | ≈ 10 分 | ≈ 50 分 |
| **ミニバッチ (B=64, GPU)** | **≈ 30〜60 秒** | **≈ 3〜5 分** |
| ミニバッチ (B=64, CPU NEON) | ≈ 2〜3 分 | ≈ 10〜15 分 |

ミニバッチ化により GPU が CPU を逆転する見込み。  
行列サイズが 128×784×64 = 6.4M 要素となり、GPU の並列性が活きる領域に入る。
