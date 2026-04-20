# LT 本番用カンペ

RubyKaigi 2026 LT のデモで使うコマンド一覧。プロジェクタ脇の iPad / スマホに
開いておくと安心。

---

## 0. 事前準備(LT 前日〜当日朝)

### SSH 接続(有線直結)

```bash
# Mac en7 = 10.0.0.1, Pi eth0 = 10.0.0.2
ssh ubuntu@10.0.0.2
```

### 作業ディレクトリ移動 + 変数設定

```bash
cd /home/ubuntu/work/mruby-gpu
git pull
MRUBY=/home/ubuntu/work/mruby/build/host/bin/mruby
```

### swap 有効化(prepare_mnist 用)

```bash
sudo swapon /swapfile
free -h       # Swap: 4.0Gi が見えていれば OK
```

### シェーダーコンパイル(初回 or 更新時のみ)

```bash
cd shader && make && cd ..
ls shader/*.spv   # add.spv matmul.spv 等が揃っているか
```

### カメラ確認

```bash
ls /dev/video0    # 存在すること
v4l2-ctl --device=/dev/video0 --all | head -5
```

### MNIST データ準備(bench/packing_inference 用、任意)

```bash
bash tools/download_mnist.sh
ruby tools/prepare_mnist.rb
```

---

## 1. LT 前の計測(スライドに埋める数値の取得)

### p.10 "It worked" — 初回成功ログ

```bash
$MRUBY examples/gpu_add.rb
```

期待出力:
```
Backend: Vulkan
Device:  V3D 7.1.x.x
Result:  [5.0, 7.0, 9.0]
```

### p.11 "The wall" — Transfer / Compute / Readback 内訳

```bash
$MRUBY bench/basic.rb
```

期待出力(値は実機依存):
```
Elements:  10000
Transfer:  X.X ms
Compute:   X.X ms
Readback:  X.X ms
Result:    [...]
```

### p.12 "Packing" — Before / After の数値

```bash
$MRUBY bench/packing_inference.rb
```

期待出力(Nx faster がスライドの "7x faster" に対応):
```
[Before] running 200 inferences (batch=1)...  X.XXs
[After]  running 192 inferences (batch=64)... X.XXs

           per sample      estimated total (N=10000)
[Before]     X.XX ms         X.X s
[After]      X.XX ms         X.X s

Nx faster
```

### p.14 "GPU < CPU" — 各モードの FPS

```bash
# GPU モード(重い、~6 FPS)
DISPLAY=:0 $MRUBY demo/face_demo.rb fast gpu

# CPU モード(軽い、~30 FPS)
DISPLAY=:0 $MRUBY demo/face_demo.rb fast cpu
```

画面左上の `FPS:` 値を読み取ってスライドに記載。

---

## 2. LT 本番(p.15 "Demo" ライブ実行)

スライド p.15 の順番に実行。ESC で各デモを終了して次へ。

### Demo 1: 1M 要素加算 — GPU の並列計算力

```bash
$MRUBY demo/vector_add.rb
```

- 時間: 30 秒程度
- 見どころ: 100 万要素を数 ms で処理

### Demo 2: 顔検出 — GPU モード(期待通り動くが遅い)

```bash
DISPLAY=:0 $MRUBY demo/face_demo.rb fast gpu
```

- 時間: 20 秒程度
- 見どころ: ~6 FPS でカクカク動作、PEOPLE: N が表示
- **ESC で終了してから次へ**

### Demo 3: 顔検出 — CPU モード(1 行変えるだけで速い)

```bash
DISPLAY=:0 $MRUBY demo/face_demo.rb fast cpu
```

- 時間: 20 秒程度
- 見どころ: ~30 FPS でヌルヌル動作

---

## 3. トラブルシューティング

### `/dev/dri` 権限エラー(`video` / `render` グループ未反映時)

```bash
sg video -c "sg render -c 'DISPLAY=:0 $MRUBY demo/face_demo.rb fast cpu'"
```

### `DISPLAY` が未設定と言われる

```bash
export DISPLAY=:0
# Pi のコンソール/VNC セッションを起動中か確認
```

### カメラが見えない(`/dev/video0` 無し)

```bash
# ドライバ確認
v4l2-ctl --list-devices

# 他デバイスを指定(例: /dev/video1)
# demo/face_demo.rb の Camera.open 引数を書き換える
```

### mruby バイナリが無い

```bash
cd /home/ubuntu/work/mruby
rake MRUBY_CONFIG=gpu
ls build/host/bin/mruby
```

### 人数表示が `?` に化ける

→ 過去のバグ。[PR #6](../../pull/6) で修正済み。`git pull` で最新にすればOK。

---

## 4. 緊急時フォールバック

ライブデモが動かなかった時のための**事前録画**を用意しておく:

- [ ] GPU モード顔検出の動画(10 秒)
- [ ] CPU モード顔検出の動画(10 秒)
- [ ] 1M ベクトル加算の出力スクショ

iPad に入れておいて、プロジェクタに映す緊急フォールバック用。
