# MNIST Minibatch Training on GPU via mruby + Vulkan Compute
# 2-layer MLP: 784 -> 128 (ReLU) -> 10 (Softmax + Cross-Entropy)
# Minibatch SGD: configurable batch size via ARGV

GPU.init("shader")

BATCH   = (ARGV[0] || "16").to_i
HIDDEN  = 128
CLASSES = 10
INPUT   = 784
LR      = 0.01
EPOCHS  = 1        # ベンチマーク用に 1 epoch
N_TRAIN = 60000

puts "=== MNIST Minibatch Training on #{GPU.device_name} ==="
puts "    Batch size: #{BATCH}"
TRAIN_START = Time.now

# --- Weight Initialization (Xavier) ---
srand(42)
def xavier(fan_in, fan_out)
  scale = Math.sqrt(2.0 / (fan_in + fan_out))
  Array.new(fan_out * fan_in) { (rand - 0.5) * 2 * scale }
end

w1 = GPU.array(xavier(INPUT, HIDDEN))     # 128 x 784
b1 = GPU.array(Array.new(HIDDEN, 0.0))
w2 = GPU.array(xavier(HIDDEN, CLASSES))   # 10 x 128
b2 = GPU.array(Array.new(CLASSES, 0.0))

# --- bias をバッチサイズ分タイル化するヘルパー ---
# add_broadcast 未実装のため、bias を行方向に B 回繰り返して同サイズにする
# matmul 出力は row-major: C[row][col] = index row*N+col
# bias[row] を各 col に加算したいので: tiled[row*B+col] = bias[row]
def tile_bias(bias_gpu, size, batch)
  vals = bias_gpu.head(size)
  tiled = Array.new(size * batch)
  size.times do |r|
    batch.times do |c|
      tiled[r * batch + c] = vals[r]
    end
  end
  GPU.array(tiled)
end

# --- Load labels ---
labels_buf = GPU.load("data/train_labels.bin")
labels_all = labels_buf.head(N_TRAIN).map { |v| v.to_i }

# --- Training Loop ---
n_iters = N_TRAIN / BATCH
correct = 0
loss_sum = 0.0

n_iters.times do |iter|
  base = iter * BATCH

  # === Load batch ===
  # ファイルは sample-major (BATCH×INPUT): [sample0の784px, sample1の784px, ...]
  # matmul は column-major (INPUT×BATCH) を期待するので転置
  raw = GPU.load("data/train_images.bin", base * INPUT, INPUT * BATCH)
  x_batch = GPU.transpose(raw, BATCH, INPUT)  # BATCH×INPUT → INPUT×BATCH

  # === Forward ===
  # Layer 1: Z1 = W1 @ X_batch (128 x B), H = ReLU(Z1 + b1)
  z1 = GPU.matmul(w1, x_batch, HIDDEN, INPUT, BATCH)
  b1_tile = tile_bias(b1, HIDDEN, BATCH)
  h = GPU.relu(GPU.add(z1, b1_tile))

  # Layer 2: O = W2 @ H (10 x B) + b2
  o_batch = GPU.add(GPU.matmul(w2, h, CLASSES, HIDDEN, BATCH),
                    tile_bias(b2, CLASSES, BATCH))

  # === Softmax + Loss (Ruby, per-sample) ===
  # matmul 出力は row-major: O[class_c * BATCH + sample_b]
  o_vals = o_batch.head(CLASSES * BATCH)
  grad_o_data = Array.new(CLASSES * BATCH, 0.0)

  BATCH.times do |b|
    # sample b のスコアを stride アクセスで取得
    scores = Array.new(CLASSES) { |c| o_vals[c * BATCH + b] }
    max_s = scores.max
    exps = scores.map { |s| Math.exp(s - max_s) }
    sum_e = exps.sum
    probs = exps.map { |e| e / sum_e }

    label = labels_all[base + b]
    predicted = probs.each_with_index.max_by { |v, _| v }[1]
    correct += 1 if predicted == label
    loss_sum += -Math.log(probs[label] + 1e-8)

    # grad_o = probs - one_hot (row-major に scatter)
    CLASSES.times { |c| grad_o_data[c * BATCH + b] = probs[c] }
    grad_o_data[label * BATCH + b] -= 1.0
  end
  grad_o = GPU.array(grad_o_data)

  # === Backward ===
  # dW2 = (1/B) * grad_O @ H^T  (10×B) @ (B×128) = 10×128
  grad_w2 = GPU.scale(GPU.matmul_nt(grad_o, h, CLASSES, BATCH, HIDDEN), 1.0 / BATCH)

  # dH = W2^T @ grad_O  (128×10) @ (10×B) = 128×B
  grad_h = GPU.matmul_tn(w2, grad_o, HIDDEN, CLASSES, BATCH)

  # ReLU backward
  h_vals = h.head(HIDDEN * BATCH)
  mask_data = h_vals.map { |v| v > 0 ? 1.0 : 0.0 }
  mask = GPU.array(mask_data)
  grad_h_pre = GPU.mul(grad_h, mask)

  # dW1 = (1/B) * grad_h_pre @ X^T  (128×B) @ (B×784) = 128×784
  grad_w1 = GPU.scale(GPU.matmul_nt(grad_h_pre, x_batch, HIDDEN, BATCH, INPUT), 1.0 / BATCH)

  # === Bias gradient: バッチ方向に平均 (row-major: [neuron][sample]) ===
  ghp_vals = grad_h_pre.head(HIDDEN * BATCH)
  gb1_data = Array.new(HIDDEN) { |i|
    s = 0.0; BATCH.times { |j| s += ghp_vals[i * BATCH + j] }; s / BATCH
  }
  go_vals = grad_o.head(CLASSES * BATCH)  # reuse from earlier
  gb2_data = Array.new(CLASSES) { |c|
    s = 0.0; BATCH.times { |j| s += go_vals[c * BATCH + j] }; s / BATCH
  }

  # === SGD Update ===
  w1 = GPU.sub(w1, GPU.scale(grad_w1, LR))
  w2 = GPU.sub(w2, GPU.scale(grad_w2, LR))
  b1 = GPU.sub(b1, GPU.scale(GPU.array(gb1_data), LR))
  b2 = GPU.sub(b2, GPU.scale(GPU.array(gb2_data), LR))

  # Progress
  if (iter + 1) % (n_iters / 6) == 0 || iter == n_iters - 1
    done = (iter + 1) * BATCH
    acc = (correct.to_f / done * 100).round(1)
    avg_loss = (loss_sum / done).round(4)
    elapsed = ((Time.now - TRAIN_START)).round(1)
    puts "  [#{done}/#{N_TRAIN}] loss=#{avg_loss} acc=#{acc}% (#{elapsed}s)"
  end
end

total_sec = (Time.now - TRAIN_START).round(1)
acc = (correct.to_f / (n_iters * BATCH) * 100).round(1)
avg_loss = (loss_sum / (n_iters * BATCH)).round(4)
puts "Epoch 1: loss=#{avg_loss} accuracy=#{acc}%  time=#{total_sec}s (B=#{BATCH})"
