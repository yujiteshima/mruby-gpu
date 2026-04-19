# Architecture

mruby-gpu の内部構造と、mruby スクリプトから GPU ハードウェアまでのデータフロー。

---

## 全体像

mruby コードから Raspberry Pi 5 の GPU に辿り着くまでのレイヤ:

```
mruby script (Ruby)
    │  例: c = GPU.add(a, b)
    ▼
mrbgem バインディング (src/gpu_ops.c)
    │  mrb_value ↔ GpuBuffer 変換
    ▼
Vulkan バックエンド (src/gpu_vulkan.c)
    │  vkCreateBuffer / vkCmdDispatch / vkQueueSubmit
    ▼
Vulkan Driver (Mesa V3D 7.1)
    │
    ▼
GPU ハードウェア (Broadcom VideoCore VII / Raspberry Pi 5)
```

---

## 各レイヤの責務

### 1. mruby スクリプト層

ユーザが書く層。GPU の詳細を意識せず、高レベル API を呼ぶだけ。

```ruby
GPU.init("shader")
a = GPU.array([1.0, 2.0, 3.0])
b = GPU.array([4.0, 5.0, 6.0])
c = GPU.add(a, b)
puts c.head(3).inspect    # => [5.0, 7.0, 9.0]
```

### 2. mrbgem バインディング層

[src/gpu_ops.c](../src/gpu_ops.c) が中心。Ruby の `mrb_value` と C 構造体 `GpuBuffer` の変換を担当。

- `mrb_mruby_gpu_gem_init` で `GPU` モジュール / `GPU::Buffer` クラスを Ruby 側に定義
- 各メソッド呼び出しは 「引数取り出し → GPU バックエンド呼び出し → 戻り値 wrap」 の3段構え

### 3. Vulkan バックエンド層

[src/gpu_vulkan.c](../src/gpu_vulkan.c) が中心。

- Vulkan Instance / Device / Queue / CommandPool を 1 度だけ初期化
- 算術演算ごとに Pipeline / DescriptorSetLayout を作成
- 演算リクエストを受けたら `dispatch_compute` で:
  1. DescriptorSet をバッファに bind
  2. CommandBuffer に `vkCmdBindPipeline` / `vkCmdDispatch` を記録
  3. `vkQueueSubmit` + `vkQueueWaitIdle` で実行完了を待つ

### 4. Compute Shader 層

[shader/](../shader/) 以下の `.comp` ファイル(GLSL)が GPU 上で実行される実体。`shader/Makefile` で SPIR-V (`.spv`) にコンパイルされ、起動時にロードされる。

| ファイル | 演算 |
|---|---|
| `add.comp` | `c[i] = a[i] + b[i]` |
| `sub.comp` | `c[i] = a[i] - b[i]` |
| `mul.comp` | `c[i] = a[i] * b[i]` |
| `scale.comp` | `b[i] = a[i] * scalar` |
| `relu.comp` | `b[i] = max(0, a[i])` |
| `matmul.comp` | 行列乗算(転置オプション付き) |
| `rgb_convert.comp` | YUYV → RGB 変換 |
| `draw_rect.comp` | 矩形描画(検出枠の可視化) |

---

## Vulkan 初期化 — レストランの比喩

Vulkan は OpenGL と違い**キッチンを全部自分で組み立てる**低レベル API。初期化手順は次の 8 ステップ:

| # | Vulkan API | 比喩 |
|---|---|---|
| 1 | `vkCreateInstance` | レストランに入る |
| 2 | `vkEnumeratePhysicalDevices` + pick | シェフを選ぶ |
| 3 | `vkCreateDevice` | 席につく |
| 4 | `vkCreateCommandPool` | 注文カウンターを見つける |
| 5 | `vkAllocateCommandBuffers` | 注文票を準備 |
| 6 | `vkCreateDescriptorSetLayout` | 材料の形式を決める |
| 7 | `vkCreatePipeline` / `ShaderModule` | レシピを渡す |
| 8 | `vkCreateBuffer` / `vkAllocateMemory` | 食材庫を予約 |

実装は [src/gpu_vulkan.c](../src/gpu_vulkan.c) の `ensure_initialized` 周辺。

---

## データフロー: `GPU.add(a, b)` の内部

```
[mruby]  c = GPU.add(a, b)
   │
   ▼
[src/gpu_ops.c]  mrb_gpu_add → mrb_gpu_binop3
   │  mrb_get_args で a, b の GpuBuffer* を取り出す
   │  create_buffer(mrb, a->n) で出力 c を確保
   ▼
[src/gpu_vulkan.c]  dispatch_compute(PIPE_ADD, bufs, ...)
   │  1. pipeline 切り替え (add.spv)
   │  2. descriptor set に a, b, c をバインド
   │  3. vkCmdDispatch((n + 255) / 256, 1, 1)
   │  4. vkQueueSubmit + vkQueueWaitIdle
   ▼
[GPU]  VideoCore VII が add.comp を並列実行
   │
   ▼
[mruby]  wrap_buffer(c) → Ruby の GPU::Buffer として返る
```

重要な特徴: **`GPU.add` の結果は GPU 上に残る**(Ruby 配列ではなく `GPU::Buffer`)。
`c.head(n)` を呼ぶまで CPU ↔ GPU 転送は発生しない。これが
[docs/GPU_VS_CPU_ANALYSIS.md](GPU_VS_CPU_ANALYSIS.md) で示す
"packing" 戦略の基盤になっている。

---

## 検出器のアプローチ比較

3 種類の検出器が、意図的に異なる実装方針を採っている:

| 検出器 | 推論エンジン | GPU 使用 | 依存 |
|---|---|---|---|
| `FaceDetector` | NCNN(Tencent) | Vulkan / NEON 切替可 | libncnn |
| `SkinDetector` | 自作ロジックベース | なし(CPU のみ) | libc のみ |
| MNIST MLP | 自作 mruby + Compute Shader | 常に Vulkan | なし |

詳細は [README の推論モデル・検出器セクション](../README.md) 参照。
