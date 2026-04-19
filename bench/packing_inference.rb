# Packing inference benchmark:
#   Before: batch=1   (mnist.rb pattern — per-sample forward + softmax)
#   After:  batch=64  (batched forward, single readback per batch)
#
# Runs SAMPLES inferences then extrapolates to N_TEST (MNIST test set).

GPU.init("shader")
puts "=== Packing inference benchmark on #{GPU.device_name} ==="

HIDDEN  = 128
CLASSES = 10
INPUT   = 784
N_TEST  = 10_000
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

# --- Synthetic weights (pre-trained MNIST 代替) ---
fc1_w = GPU.array(xavier(INPUT, HIDDEN))
fc1_b = GPU.array(Array.new(HIDDEN, 0.0))
fc2_w = GPU.array(xavier(HIDDEN, CLASSES))
fc2_b = GPU.array(Array.new(CLASSES, 0.0))

# --- Synthetic test images ---
images = Array.new(SAMPLES) { Array.new(INPUT) { rand } }

# =====================================================================
# BEFORE: batch=1 inference (mnist.rb pattern)
# =====================================================================
print "[Before] running #{SAMPLES} inferences (batch=1)... "
t0 = Time.now
SAMPLES.times do |i|
  x = GPU.array(images[i])                                # CPU -> GPU (784 floats)

  # Forward
  h = GPU.relu(GPU.add(GPU.matmul(fc1_w, x, HIDDEN, INPUT, 1), fc1_b))
  o = GPU.add(GPU.matmul(fc2_w, h, CLASSES, HIDDEN, 1), fc2_b)

  # Softmax + argmax (Ruby) — forces GPU -> CPU readback per sample
  scores = o.head(CLASSES)                                # GPU -> CPU (10 floats)
  max_s = scores.max
  exps = scores.map { |s| Math.exp(s - max_s) }
  sum_e = exps.inject(0.0, :+)
  probs = exps.map { |e| e / sum_e }
  _predicted = probs.each_with_index.max_by { |v, _| v }[1]
end
before_elapsed = Time.now - t0
before_per_sample_ms = before_elapsed * 1000.0 / SAMPLES
before_total_s = before_per_sample_ms * N_TEST / 1000.0
puts "#{before_elapsed.round(2)}s"

# =====================================================================
# AFTER: batch=64 inference
# =====================================================================
n_iters = SAMPLES / BATCH
samples_done = n_iters * BATCH

print "[After]  running #{samples_done} inferences (batch=#{BATCH})... "
t0 = Time.now
n_iters.times do |iter|
  base = iter * BATCH

  # Pack BATCH samples into one upload
  batch_flat = []
  BATCH.times { |b| batch_flat.concat(images[base + b]) }
  raw = GPU.array(batch_flat)                             # CPU -> GPU (packed)
  x_batch = GPU.transpose(raw, BATCH, INPUT)

  # Forward (batched) — matmul dispatches get BATCH× bigger
  h = GPU.relu(GPU.add(GPU.matmul(fc1_w, x_batch, HIDDEN, INPUT, BATCH),
                       tile_bias(fc1_b, HIDDEN, BATCH)))
  o_batch = GPU.add(GPU.matmul(fc2_w, h, CLASSES, HIDDEN, BATCH),
                    tile_bias(fc2_b, CLASSES, BATCH))

  # Single readback for the whole batch
  o_vals = o_batch.head(CLASSES * BATCH)                  # GPU -> CPU (packed)
  BATCH.times do |b|
    scores = Array.new(CLASSES) { |c| o_vals[c * BATCH + b] }
    max_s = scores.max
    exps = scores.map { |s| Math.exp(s - max_s) }
    sum_e = exps.inject(0.0, :+)
    probs = exps.map { |e| e / sum_e }
    _predicted = probs.each_with_index.max_by { |v, _| v }[1]
  end
end
after_elapsed = Time.now - t0
after_per_sample_ms = after_elapsed * 1000.0 / samples_done
after_total_s = after_per_sample_ms * N_TEST / 1000.0
puts "#{after_elapsed.round(2)}s"

# =====================================================================
# Results
# =====================================================================
speedup = before_total_s / after_total_s
puts ""
puts "           per sample      estimated total (N=#{N_TEST})"
puts "[Before]   #{before_per_sample_ms.round(2).to_s.rjust(7)} ms     #{before_total_s.round(1).to_s.rjust(7)} s"
puts "[After]    #{after_per_sample_ms.round(2).to_s.rjust(7)} ms     #{after_total_s.round(1).to_s.rjust(7)} s"
puts ""
puts "#{speedup.round(1)}x faster"
