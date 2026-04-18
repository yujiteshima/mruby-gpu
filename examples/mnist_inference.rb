# MNIST Batched Inference on GPU via mruby + Vulkan Compute
# 10,000 枚のテスト画像を BATCH 単位で推論し、精度と所要時間を出力する。
#
# 前提:
#   tools/prepare_mnist.rb を実行して以下を生成しておく
#     data/test_images.bin   (10000 x 784 float32)
#     data/test_labels.bin   (10000 float32, 値は 0..9)
#     weights/fc1_weight.bin, fc1_bias.bin, fc2_weight.bin, fc2_bias.bin
#
# 使い方:
#   mruby examples/mnist_inference.rb         # デフォルト BATCH=64
#   mruby examples/mnist_inference.rb 128     # BATCH=128

GPU.init("shader")

BATCH   = (ARGV[0] || "64").to_i
HIDDEN  = 128
CLASSES = 10
INPUT   = 784
N_TEST  = 10_000

puts "=== MNIST Batched Inference on #{GPU.device_name} ==="
puts "    Backend: #{GPU.backend}"
puts "    Batch size: #{BATCH}"

# --- Load pre-trained weights ---
fc1_w = GPU.load("weights/fc1_weight.bin")   # HIDDEN x INPUT  = 128 x 784
fc1_b = GPU.load("weights/fc1_bias.bin")     # HIDDEN         = 128
fc2_w = GPU.load("weights/fc2_weight.bin")   # CLASSES x HIDDEN = 10 x 128
fc2_b = GPU.load("weights/fc2_bias.bin")     # CLASSES         = 10
puts "    Weights: fc1=#{fc1_w.size} fc1_b=#{fc1_b.size} fc2=#{fc2_w.size} fc2_b=#{fc2_b.size}"

# --- Load labels (all N_TEST) ---
labels_buf = GPU.load("data/test_labels.bin")
labels_all = labels_buf.head(N_TEST).map { |v| v.to_i }

# --- bias のバッチタイル化 (add_broadcast 未実装のため) ---
# matmul 出力は row-major: C[row * BATCH + col]
# bias[row] を各列に加算したいので tiled[row*BATCH+col] = bias[row]
def tile_bias(bias_gpu, size, batch)
  vals = bias_gpu.head(size)
  tiled = Array.new(size * batch)
  size.times { |r| batch.times { |c| tiled[r * batch + c] = vals[r] } }
  GPU.array(tiled)
end

# --- Inference loop ---
correct = 0
n_iters = N_TEST / BATCH
samples_done = n_iters * BATCH

t0 = Time.now
n_iters.times do |iter|
  base = iter * BATCH

  # Batched upload: BATCH 枚分を一発で GPU に送る
  # ファイルは sample-major (BATCH x INPUT): [sample0_784px, sample1_784px, ...]
  # matmul は (INPUT x BATCH) を期待するので転置
  raw = GPU.load("data/test_images.bin", base * INPUT, INPUT * BATCH)
  x_batch = GPU.transpose(raw, BATCH, INPUT)

  # Forward: Z1 = W1 @ X (128 x BATCH), H = ReLU(Z1 + b1)
  z1 = GPU.matmul(fc1_w, x_batch, HIDDEN, INPUT, BATCH)
  h  = GPU.relu(GPU.add(z1, tile_bias(fc1_b, HIDDEN, BATCH)))

  # Layer 2: O = W2 @ H (10 x BATCH) + b2
  o_batch = GPU.add(GPU.matmul(fc2_w, h, CLASSES, HIDDEN, BATCH),
                    tile_bias(fc2_b, CLASSES, BATCH))

  # 読み戻しは batch 単位で1回
  o_vals = o_batch.head(CLASSES * BATCH)

  # Softmax + argmax (Ruby) — サンプルごとに予測
  BATCH.times do |b|
    scores = Array.new(CLASSES) { |c| o_vals[c * BATCH + b] }
    predicted = scores.each_with_index.max_by { |v, _| v }[1]
    correct += 1 if predicted == labels_all[base + b]
  end

  # 進捗表示
  if (iter + 1) % (n_iters / 5) == 0 || iter == n_iters - 1
    done = (iter + 1) * BATCH
    acc = correct.to_f / done * 100
    puts "  [#{done}/#{samples_done}] acc=#{acc.round(2)}%"
  end
end
elapsed = Time.now - t0

# --- 残り端数(BATCH で割り切れない分)を batch=1 で処理 ---
remainder = N_TEST - samples_done
if remainder > 0
  remainder.times do |i|
    idx = samples_done + i
    x = GPU.load("data/test_images.bin", idx * INPUT, INPUT)
    h = GPU.relu(GPU.add(GPU.matmul(fc1_w, x, HIDDEN, INPUT, 1), fc1_b))
    o = GPU.add(GPU.matmul(fc2_w, h, CLASSES, HIDDEN, 1), fc2_b)
    scores = o.head(CLASSES)
    predicted = scores.each_with_index.max_by { |v, _| v }[1]
    correct += 1 if predicted == labels_all[idx]
  end
end

# --- Results ---
total_elapsed = Time.now - t0
accuracy = correct.to_f / N_TEST * 100
per_sample_ms = total_elapsed * 1000.0 / N_TEST

puts ""
puts "=== Inference complete ==="
puts "  Total:       #{N_TEST} images"
puts "  Accuracy:    #{correct}/#{N_TEST} (#{accuracy.round(2)}%)"
puts "  Elapsed:     #{total_elapsed.round(2)} s"
puts "  Per sample:  #{per_sample_ms.round(3)} ms"
puts "  Throughput:  #{(N_TEST / total_elapsed).round(1)} images/s"
