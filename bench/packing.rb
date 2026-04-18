# Packing benchmark:
#   Before: batch=1   (train.rb pattern)
#   After:  batch=64  (train_minibatch.rb pattern)
#
# Runs SAMPLES iterations then extrapolates to N_TRAIN samples/epoch.

GPU.init("shader")
puts "=== Packing benchmark on #{GPU.device_name} ==="

HIDDEN  = 128
CLASSES = 10
INPUT   = 784
LR      = 0.01
N_TRAIN = 60_000
SAMPLES = 200
BATCH   = 64

srand(42)
def xavier(fan_in, fan_out)
  scale = Math.sqrt(2.0 / (fan_in + fan_out))
  Array.new(fan_out * fan_in) { (rand - 0.5) * 2 * scale }
end

def tile_bias(bias_gpu, size, batch)
  vals = bias_gpu.head(size)
  tiled = Array.new(size * batch)
  size.times { |r| batch.times { |c| tiled[r * batch + c] = vals[r] } }
  GPU.array(tiled)
end

# --- Synthetic inputs (MNIST 代替) ---
images = Array.new(SAMPLES * INPUT) { rand }
labels = Array.new(SAMPLES) { rand(CLASSES) }

# =====================================================================
# BEFORE: batch=1 pattern (train.rb)
# =====================================================================
w1 = GPU.array(xavier(INPUT, HIDDEN))
b1 = GPU.array(Array.new(HIDDEN, 0.0))
w2 = GPU.array(xavier(HIDDEN, CLASSES))
b2 = GPU.array(Array.new(CLASSES, 0.0))

print "[Before] running #{SAMPLES} samples (batch=1)... "
t0 = Time.now
SAMPLES.times do |i|
  x = GPU.array(images[i * INPUT, INPUT])                 # CPU -> GPU

  # Forward
  z1 = GPU.matmul(w1, x, HIDDEN, INPUT, 1)
  h  = GPU.relu(GPU.add(z1, b1))
  o  = GPU.add(GPU.matmul(w2, h, CLASSES, HIDDEN, 1), b2)

  scores = o.head(CLASSES)                                # GPU -> CPU
  max_s = scores.max
  exps = scores.map { |s| Math.exp(s - max_s) }
  sum_e = exps.inject(0.0, :+)
  probs = exps.map { |e| e / sum_e }
  grad_o_data = probs.dup
  grad_o_data[labels[i]] -= 1.0
  grad_o = GPU.array(grad_o_data)                         # CPU -> GPU

  # Backward
  grad_w2 = GPU.matmul_nt(grad_o, h, CLASSES, 1, HIDDEN)
  grad_h  = GPU.matmul_tn(w2, grad_o, HIDDEN, CLASSES, 1)
  h_vals = h.head(HIDDEN)                                 # GPU -> CPU
  mask = GPU.array(h_vals.map { |v| v > 0 ? 1.0 : 0.0 })  # CPU -> GPU
  grad_h_pre = GPU.mul(grad_h, mask)
  grad_w1 = GPU.matmul_nt(grad_h_pre, x, HIDDEN, 1, INPUT)

  # SGD
  w1 = GPU.sub(w1, GPU.scale(grad_w1, LR))
  b1 = GPU.sub(b1, GPU.scale(grad_h_pre, LR))
  w2 = GPU.sub(w2, GPU.scale(grad_w2, LR))
  b2 = GPU.sub(b2, GPU.scale(grad_o, LR))
end
before_elapsed = Time.now - t0
before_per_sample_ms = before_elapsed * 1000.0 / SAMPLES
before_epoch_s = before_per_sample_ms * N_TRAIN / 1000.0
puts "#{before_elapsed.round(2)}s"

# =====================================================================
# AFTER: batch=64 pattern (train_minibatch.rb)
# =====================================================================
w1 = GPU.array(xavier(INPUT, HIDDEN))
b1 = GPU.array(Array.new(HIDDEN, 0.0))
w2 = GPU.array(xavier(HIDDEN, CLASSES))
b2 = GPU.array(Array.new(CLASSES, 0.0))

n_iters = SAMPLES / BATCH
samples_done = n_iters * BATCH

print "[After]  running #{samples_done} samples (batch=#{BATCH})... "
t0 = Time.now
n_iters.times do |iter|
  base = iter * BATCH
  raw = GPU.array(images[base * INPUT, INPUT * BATCH])    # CPU -> GPU (packed)
  x_batch = GPU.transpose(raw, BATCH, INPUT)

  # Forward (batched)
  z1 = GPU.matmul(w1, x_batch, HIDDEN, INPUT, BATCH)
  h  = GPU.relu(GPU.add(z1, tile_bias(b1, HIDDEN, BATCH)))
  o_batch = GPU.add(GPU.matmul(w2, h, CLASSES, HIDDEN, BATCH),
                    tile_bias(b2, CLASSES, BATCH))

  o_vals = o_batch.head(CLASSES * BATCH)                  # GPU -> CPU (packed)
  grad_o_data = Array.new(CLASSES * BATCH, 0.0)
  BATCH.times do |b|
    scores = Array.new(CLASSES) { |c| o_vals[c * BATCH + b] }
    max_s = scores.max
    exps = scores.map { |s| Math.exp(s - max_s) }
    sum_e = exps.inject(0.0, :+)
    probs = exps.map { |e| e / sum_e }
    label = labels[base + b]
    CLASSES.times { |c| grad_o_data[c * BATCH + b] = probs[c] }
    grad_o_data[label * BATCH + b] -= 1.0
  end
  grad_o = GPU.array(grad_o_data)                         # CPU -> GPU (packed)

  # Backward (batched)
  grad_w2 = GPU.scale(GPU.matmul_nt(grad_o, h, CLASSES, BATCH, HIDDEN), 1.0 / BATCH)
  grad_h  = GPU.matmul_tn(w2, grad_o, HIDDEN, CLASSES, BATCH)
  h_vals = h.head(HIDDEN * BATCH)
  mask = GPU.array(h_vals.map { |v| v > 0 ? 1.0 : 0.0 })
  grad_h_pre = GPU.mul(grad_h, mask)
  grad_w1 = GPU.scale(GPU.matmul_nt(grad_h_pre, x_batch, HIDDEN, BATCH, INPUT), 1.0 / BATCH)

  ghp_vals = grad_h_pre.head(HIDDEN * BATCH)
  gb1_data = Array.new(HIDDEN) { |i|
    s = 0.0; BATCH.times { |j| s += ghp_vals[i * BATCH + j] }; s / BATCH
  }
  go_vals = grad_o.head(CLASSES * BATCH)
  gb2_data = Array.new(CLASSES) { |c|
    s = 0.0; BATCH.times { |j| s += go_vals[c * BATCH + j] }; s / BATCH
  }

  # SGD
  w1 = GPU.sub(w1, GPU.scale(grad_w1, LR))
  w2 = GPU.sub(w2, GPU.scale(grad_w2, LR))
  b1 = GPU.sub(b1, GPU.scale(GPU.array(gb1_data), LR))
  b2 = GPU.sub(b2, GPU.scale(GPU.array(gb2_data), LR))
end
after_elapsed = Time.now - t0
after_per_sample_ms = after_elapsed * 1000.0 / samples_done
after_epoch_s = after_per_sample_ms * N_TRAIN / 1000.0
puts "#{after_elapsed.round(2)}s"

# =====================================================================
# Results
# =====================================================================
speedup = before_epoch_s / after_epoch_s
puts ""
puts "           per sample      estimated epoch (N=#{N_TRAIN})"
puts "[Before]   #{before_per_sample_ms.round(2).to_s.rjust(7)} ms     #{before_epoch_s.round(1).to_s.rjust(7)} s"
puts "[After]    #{after_per_sample_ms.round(2).to_s.rjust(7)} ms     #{after_epoch_s.round(1).to_s.rjust(7)} s"
puts ""
puts "#{speedup.round(1)}x faster"
