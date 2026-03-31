# mruby-gpu

mruby から Raspberry Pi 5 の GPU（VideoCore VII / Vulkan 1.2）を利用するための mrbgem です。  
カメラキャプチャ・GPU顔認識・リアルタイムプレビューをたった数十行の mruby スクリプトで実現します。

## ドキュメント

| ファイル | 内容 |
|---------|------|
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | ボトルネック分析と高速化アイデア |

---

## 機能

| クラス | 概要 |
|--------|------|
| `GPU` | Vulkan Compute Shader によるベクトル演算・行列演算 |
| `Camera` | V4L2 経由のカメラキャプチャ（USB / CSI カメラ対応） |
| `FaceDetector` | NCNN + Vulkan による GPU 顔検出（UltraFace-slim モデル） |
| `Display` | SDL2 ウィンドウへのリアルタイム表示・枠描画 |

---

## 動作環境

- **ボード**: Raspberry Pi 5
- **OS**: Ubuntu 24.04 (arm64)
- **GPU**: VideoCore VII（V3D 7.1）— Vulkan 1.2 対応
- **カメラ**: USB Webカメラ（UVC準拠、例: Logitech C920）

---

## 依存関係のインストール

```bash
# Vulkan / SDL2 / ビルドツール
sudo apt install -y libvulkan-dev vulkan-tools libsdl2-dev cmake build-essential git

# NCNN (GPU 推論フレームワーク) — ソースからビルド
git clone --depth=1 https://github.com/Tencent/ncnn.git ~/ncnn
cd ~/ncnn
git submodule update --init --depth=1
mkdir build && cd build
cmake \
  -DNCNN_VULKAN=ON \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  ..
make -j1          # メモリ不足防止のため -j1 推奨（RAM 2GB 環境）
sudo make install
```

> **注意**: Raspberry Pi 5 (RAM 2GB) では `-j4` でビルドするとフリーズします。  
> 事前にスワップを追加してください：
> ```bash
> sudo fallocate -l 4G /swapfile
> sudo chmod 600 /swapfile
> sudo mkswap /swapfile
> sudo swapon /swapfile
> ```

---

## セットアップ

### 1. モデルのダウンロード

```bash
cd /home/ubuntu/work/mruby-gpu
mkdir -p models
wget "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/ncnn/data/version-slim/slim_320.param" \
     -O models/ultraface-slim.param
wget "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/ncnn/data/version-slim/slim_320.bin" \
     -O models/ultraface-slim.bin
```

### 2. シェーダーのコンパイル

```bash
cd shader
make
cd ..
```

### 3. mruby のビルド

```bash
cd /home/ubuntu/work/mruby
rake MRUBY_CONFIG=gpu
```

ビルドが成功すると `build/host/bin/mruby` が生成されます。

---

## 顔認識デモの実行

```bash
cd /home/ubuntu/work/mruby-gpu
MRUBY=/home/ubuntu/work/mruby/build/host/bin/mruby

# fast モード: ~6 FPS、約3m以内の顔を検出（デモ・近距離向け）
DISPLAY=:0 $MRUBY face_demo.rb fast

# count モード: ~1 FPS、約7m以内の顔を検出（会場の人数カウント向け）
DISPLAY=:0 $MRUBY face_demo.rb count

# モード省略時は count モード
DISPLAY=:0 $MRUBY face_demo.rb
```

### モードの違い

| モード | 処理方式 | 速度 | 検出距離 | 用途 |
|--------|----------|------|---------|------|
| `fast` | 全体画像1回のみ | **6.1 FPS** | ~3m | デモ・プレゼン |
| `count` | 全体+5タイル分割 | **1.0 FPS** | **~7m** | LT会場の人数カウント |

**タイル処理の仕組み**（count モード）:  
640×480 の画像を 320×240 の5タイルに分割して各タイルを個別に推論することで、  
実効解像度を2倍にして遠距離の小さな顔（17px → 34px）を検出可能にしています。

```
[全体] + [左上] + [右上] + [左下] + [右下] + [中央] = 6回推論 / フレーム
       ↑ 各タイルは元画像の1/4領域 = モデル入力と同サイズで2倍ズーム相当
```

### 操作方法

| 操作 | 動作 |
|------|------|
| `ESC` キー | デモを終了してサマリーを表示 |
| ウィンドウの × ボタン | デモを終了してサマリーを表示 |
| `Ctrl+C`（ターミナル） | 強制終了（サマリーは表示されません） |

### 終了時のサマリー表示例

```
====================================================
  終了サマリー
  総フレーム数 : 300
  最大同時検出 : 5 人
  平均処理時間 : 165.0 ms/frame  (6.1 FPS)
====================================================
```

- **最大同時検出**: セッション中にカメラ映像内に映っていた最大人数

### ターミナル出力（30フレームごと）

```
Frame #30  |  2 人検出  |  163.6ms (6.1 FPS)  |  最大検出: 2 人
  [0] (120, 80) 140x200  score=0.99
  [1] (380, 60) 130x190  score=0.97
```

---

## パフォーマンス（Raspberry Pi 5 実測値）

| 処理 | fast モード | count モード |
|------|------------|-------------|
| V4L2 キャプチャ | ~2ms | ~2ms |
| YUYV → RGB 変換 (C) | ~2ms | ~2ms |
| UltraFace 推論 (NCNN + Vulkan) | ~160ms × 1 | ~160ms × 6 |
| NMS・枠描画・表示 | <2ms | <5ms |
| **合計** | **~165ms (6.1 FPS)** | **~970ms (1.0 FPS)** |
| **検出距離** | **~3m** | **~7m** |

> 初回フレームのみ Vulkan シェーダーコンパイルで約 200ms かかります。

---

## mruby API リファレンス

### `GPU` モジュール

```ruby
GPU.init("shader")        # シェーダーディレクトリを指定して初期化
GPU.backend               # => "Vulkan"
GPU.device_name           # => "V3D 7.1.10.2"

# ベクトル演算
a = GPU.array([1.0, 2.0, 3.0])
b = GPU.fill(3, 1.0)
c = GPU.add(a, b)         # => GPU::Buffer
c.head(3)                 # => [2.0, 3.0, 4.0]

# 行列演算
c = GPU.matmul(a, b, M, K, N)
```

### `Camera` クラス

```ruby
cam = Camera.open("/dev/video0", 640, 480)  # デバイス, 幅, 高さ
cam.width                                   # => 640
cam.height                                  # => 480

yuyv = cam.capture                          # YUYV raw bytes (String)
rgb  = cam.capture_rgb                      # RGB888 bytes (String)

rgb  = Camera.yuyv_to_rgb(yuyv, 640, 480)  # YUYV→RGB 変換のみ

cam.close
```

### `FaceDetector` クラス

```ruby
detector = FaceDetector.new("models/ultraface-slim", use_gpu: true)

faces = detector.detect(yuyv, width, height, threshold: 0.6)
# faces => Array of Hash
# 各要素: { x: Float, y: Float, w: Float, h: Float, score: Float }

faces.each do |f|
  puts "x=#{f[:x].round} y=#{f[:y].round} w=#{f[:w].round} h=#{f[:h].round}"
  puts "score=#{f[:score].round(2)}"
end
```

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `use_gpu:` | Vulkan GPU 推論を使用 | `true` |
| `threshold:` | 検出スコアの閾値（0.0〜1.0） | `0.7` |

- 座標 `(x, y)` は検出枠の左上角（ピクセル単位、元画像サイズ基準）
- `score` が 1.0 に近いほど確信度が高い

### `Display` クラス

```ruby
disp = Display.open(640, 480, "ウィンドウタイトル")

# RGB888 文字列をウィンドウに表示
disp.show(rgb, width, height)

# 検出枠を描画した新しい RGB 文字列を返す（元のデータは変更しない）
rgb = disp.draw_rect(rgb, width, height,
                     x, y, w, h,      # 枠の位置とサイズ（Integer）
                     r, g, b)         # 枠の色 (0〜255)

# ウィンドウ閉じる / ESC イベントを確認
break if Display.poll_quit

disp.close
```

---

## ファイル構成

```
mruby-gpu/
├── src/
│   ├── gpu.c               # GPU クラス (Vulkan Compute)
│   ├── mrb_camera.c        # Camera クラス (V4L2)
│   ├── mrb_face.cpp        # FaceDetector クラス (NCNN + Vulkan)
│   ├── mrb_display.c       # Display クラス (SDL2)
│   ├── v4l2_capture.c      # V4L2 カメラ低レベル実装
│   └── v4l2_capture.h
├── shader/
│   ├── add.comp / .spv     # ベクトル加算
│   ├── matmul.comp / .spv  # 行列積
│   ├── rgb_convert.comp    # YUYV→RGBA GPU 変換
│   ├── draw_rect.comp      # 枠描画 GPU シェーダー
│   └── Makefile
├── models/
│   ├── ultraface-slim.param
│   └── ultraface-slim.bin
├── face_demo.rb            # 顔認識デモ（メイン）
├── demo.rb                 # GPU ベクトル演算デモ
├── mrbgem.rake             # gem ビルド設定
└── README.md               # このファイル
```

---

## トラブルシューティング

**`SDL_Init failed` エラー**  
環境変数 `DISPLAY` が設定されていません。

```bash
DISPLAY=:0 mruby face_demo.rb
```

**カメラが開けない (`Camera.open failed`)**  
デバイスを確認してください。

```bash
v4l2-ctl --list-devices   # 接続中のカメラ一覧
# /dev/video0 以外の場合は face_demo.rb の Camera.open 行を修正
```

**顔が検出されない**  
- カメラに顔を向けてください
- `threshold:` を下げてみてください（例: `threshold: 0.4`）
- 照明が暗すぎる場合は明るい場所で試してください

**ビルド時に `undefined reference to ncnn` エラー**  
NCNN がインストールされていません。「依存関係のインストール」を参照してください。

**Raspberry Pi がフリーズする（NCNN ビルド時）**  
メモリ不足です。スワップを追加して `-j1` でビルドしてください（前述の手順を参照）。
