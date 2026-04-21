# demo/ — LT / ステージ向けデモ

RubyKaigi 2026 LT で使う、**見せて動かす**ためのデモスクリプト。

## スクリプト一覧

| ファイル | 内容 | 実行例 |
|---|---|---|
| [vector_add.rb](vector_add.rb) | 100 万要素ベクトル加算の GPU vs CPU 比較 | `mruby demo/vector_add.rb` |
| [face_demo.rb](face_demo.rb) | カメラ顔検出(GPU / CPU 切り替え) | `mruby demo/face_demo.rb fast cpu` |
| [counter_demo.rb](counter_demo.rb) | 人数カウント(肌色検出 / NCNN / 両方) | `mruby demo/counter_demo.rb skin` |

## 前提

- ビルド済みの mruby 実行ファイル
- シェーダがコンパイル済み(`shader/*.spv` が存在する)
- `models/ultraface-slim.{param,bin}`(`face_demo` / `counter_demo` で使用)
- `/dev/video0` に UVC カメラが接続されていること(`face_demo` / `counter_demo`)

## 主な引数

### face_demo.rb

```
mruby demo/face_demo.rb [mode] [backend]
  mode:    fast | count    (default: count)
  backend: gpu  | cpu      (default: cpu)

```

### counter_demo.rb

```
mruby demo/counter_demo.rb [method] [mode] [backend] [resolution]
  method:     skin | ncnn | both    (default: skin)
  mode:       snap | pan             (default: snap)
  backend:    cpu  | gpu             (NCNN only, default: cpu)
  resolution: 480  | 720             (default: 720)


```

/home/ubuntu/work/mruby/build/host/bin/mruby demo/vector_add.rb
DISPLAY=:0 /home/ubuntu/work/mruby/build/host/bin/mruby demo/face_demo.rb gpu
DISPLAY=:0 /home/ubuntu/work/mruby/build/host/bin/mruby demo/face_demo.rb cpu
DISPLAY=:0 /home/ubuntu/work/mruby/build/host/bin/mruby demo/counter_demo2.rb


## 注意

- `counter_demo.rb` の `ncnn` モードは **1 フレームにつき 7 パス**(全画像 + 3×2 タイル)
  を実行するため重い。LT で「GPU が CPU に負ける」対比を見せたい場合は
  `face_demo.rb fast` の方が明確。
- 権限エラー(`/dev/dri`)が出る場合は [../README.md](../README.md) のトラブル
  シューティング節を参照。
