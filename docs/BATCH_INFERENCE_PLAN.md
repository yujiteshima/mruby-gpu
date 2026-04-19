# 録画→バッチ推論パイプライン計画

リアルタイム処理ではなく「先に録画→後でまとめて推論」にすることで、  
GPU の強みを活かす新しいパイプラインの設計。

---

## コンセプト

```
現在（リアルタイム）:
  [capture] → [YUYV→RGB] → [推論] → [表示]  を 1 フレームずつ繰り返す
                              ↑ ここがボトルネック（GPU 160ms / CPU 6ms）

提案（録画→バッチ推論）:
  Phase 1: [capture] → [保存] × 300 フレーム     ← 推論なし、30fps で録画
  Phase 2: [全フレーム YUYV→RGB 一括変換]         ← GPU 1 dispatch で 300 フレーム分
  Phase 3: [推論] × 300 フレーム                  ← CPU NEON で高速処理
  Phase 4: [結果集計]                             ← 最大人数・平均人数を報告
```

---

## なぜこの方式で GPU が活きるか

### 現在の問題（リアルタイム）

リアルタイムでは 1 フレームごとに GPU dispatch が走る。  
NCNN の推論は 100 層 × 小データで GPU が不利（V3D 160ms vs CPU NEON 6ms）。

### 録画→バッチなら GPU の出番がある

| 処理 | データ量 | 1フレームずつ | 300フレーム一括 |
|------|---------|-------------|----------------|
| YUYV→RGB 変換 | 600KB/frame | 2ms × 300 = **600ms** | GPU 1 dispatch = **~10ms** |
| 画像リサイズ | 900KB→230KB | 1ms × 300 = 300ms | GPU 1 dispatch = **~5ms** |
| 推論（NCNN） | 100層/frame | GPU 向きではない | GPU 向きではない |

**前処理（YUYV→RGB + リサイズ）を GPU でバッチ化**し、  
**推論は CPU NEON で高速実行**する。  
GPU と CPU の適材適所を 1 つのパイプラインで両方見せられる。

---

## 想定する使い方

### LT 会場での人数カウント

```ruby
# 1. 10秒間録画（推論なし、30fps）
puts "録画中..."
frames = []
300.times { frames << cam.capture }

# 2. GPU で一括前処理
puts "GPU で前処理中..."
rgb_all = GPU.batch_yuyv_to_rgb(frames, W, H)  # 1 dispatch で 300 フレーム

# 3. CPU で高速推論
puts "CPU で推論中..."
max_people = 0
frames.size.times do |i|
  rgb = GPU.slice(rgb_all, i * W * H * 3, W * H * 3)
  faces = detector.detect_rgb(rgb, W, H, threshold: 0.5)
  max_people = faces.size if faces.size > max_people
end

puts "最大 #{max_people} 人検出"
```

---

## パフォーマンス予測（10 秒 = 300 フレーム）

| Phase | 処理 | 時間 |
|-------|------|------|
| 1. 録画 | V4L2 capture × 300 | 10.0 秒 |
| 2. GPU 前処理 | YUYV→RGB 300 フレーム一括 | **~0.01 秒** |
| 3. CPU 推論 | NCNN NEON × 300 | **1.8 秒** |
| 4. 集計 | Ruby 側 | < 0.01 秒 |
| **合計** | | **11.8 秒** |

### 比較

| 方式 | 10 秒の映像を処理する時間 | 備考 |
|------|-------------------------|------|
| リアルタイム CPU | 10.0 秒 | カメラ 33ms が律速（推論 6ms は間に合う） |
| リアルタイム GPU | 48.5 秒 | GPU 推論 160ms が律速 |
| **録画→バッチ** | **11.8 秒** | 録画 10秒 + GPU 前処理 + CPU 推論 1.8秒 |

リアルタイム CPU と同等の速度だが、メリットは:
- **録画中にフレーム落ちしない**（推論の遅延がない）
- **GPU 前処理のデモができる**（1 dispatch で 300 フレーム = GPU の強みを見せられる）
- **後から閾値やモードを変えて再処理できる**

---

## 実装計画

### Phase 1: フレーム保存

```ruby
# カメラフレームを mruby Array に貯める
frames = []
300.times { frames << cam.capture }
```

**必要な変更**: なし（既存 API で可能）  
**注意**: 300 フレーム × 600KB = 180MB のメモリが必要

### Phase 2: GPU バッチ前処理

300 フレーム分の YUYV データを 1 つの GPU::Buffer に結合して、  
1 回の compute shader dispatch で全フレームを RGB に変換する。

**必要な新規実装**:

| 項目 | 説明 |
|------|------|
| `GPU.batch_yuyv_to_rgb(frames, w, h)` | 複数フレームの YUYV→RGB 一括変換 |
| `batch_rgb_convert.comp` | バッチ対応の YUYV→RGB シェーダー |
| `GPU.slice(buffer, offset, count)` | 大きなバッファから部分取り出し |

#### batch_rgb_convert.comp の設計

```glsl
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly  buffer YUYVBuf { uint yuyv[]; };
layout(binding = 1) writeonly buffer RGBBuf  { uint rgb[];  };

layout(push_constant) uniform Params {
  uint width;
  uint height;
  uint n_frames;   // ← 追加: フレーム数
};

void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint pixels_per_frame = width * height;
  uint macropixels_per_frame = pixels_per_frame / 2;
  uint total_macropixels = macropixels_per_frame * n_frames;
  if (gid >= total_macropixels) return;

  // フレーム内のオフセットを計算
  uint frame_idx = gid / macropixels_per_frame;
  uint local_idx = gid % macropixels_per_frame;

  // 以降は既存の rgb_convert.comp と同じ YUV→RGB 変換
  // ただしバッファオフセットに frame_idx を加算
  ...
}
```

1 dispatch で **300 × 640 × 480 / 2 = 46,080,000 macropixel** を並列処理。  
GPU.add 100 万要素が 5ms なので、4,600 万要素でも **~200ms 程度** と推定。

### Phase 3: CPU 推論

```ruby
300.times do |i|
  rgb = GPU.slice(rgb_all, i * frame_size, frame_size)
  faces = detector.detect_rgb(rgb, W, H, threshold: 0.5)
  # 結果を集計
end
```

**必要な変更**: `GPU.slice` または `GPU::Buffer#slice` メソッドの追加

### Phase 4: 結果表示

```ruby
puts "=== 会場人数カウント結果 ==="
puts "  録画時間     : #{duration} 秒 (#{n_frames} フレーム)"
puts "  最大同時検出 : #{max_people} 人"
puts "  平均検出数   : #{avg_people.round(1)} 人"
puts "  処理時間     : #{total_sec.round(1)} 秒"
```

---

## 必要な新規実装まとめ

| 項目 | 種類 | 難易度 | 優先度 |
|------|------|--------|--------|
| `GPU.batch_yuyv_to_rgb(frames, w, h)` | C メソッド | 中 | 高 |
| `batch_rgb_convert.comp` シェーダー | GLSL | 中 | 高 |
| `GPU.slice(buffer, offset, count)` | C メソッド | 低 | 高 |
| `examples/batch_count.rb` デモスクリプト | Ruby | 低 | 高 |
| メモリ管理（180MB 確保） | C | 中 | 中 |

---

## LT でのストーリー

```
1. 「まず GPU で 100 万要素を 5ms で計算します」      → demo/vector_add.rb
2. 「次にリアルタイム顔認識」                         → demo/face_demo.rb fast cpu (30fps)
3. 「では会場の皆さんを数えます」                     → batch_count.rb
   - カメラを会場に向けて 10 秒録画
   - 「録画完了。GPU で 300 フレームを一括前処理します」
   - 「CPU で高速推論中...完了！」
   - 「この会場には最大 XX 人います！」
4. 「GPU は大量データの並列処理、CPU は ML 推論が得意。
    mruby-gpu なら 1 行の変更で使い分けられます」
```
