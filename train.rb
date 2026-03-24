# MNIST Training on GPU via mruby + Vulkan Compute
# 2-layer MLP: 784 -> 128 (ReLU) -> 10 (Softmax + Cross-Entropy)
# All matrix operations run on GPU. Softmax/loss computed in Ruby (10 elements).

GPU.init("shader")
puts "=== MNIST Training on #{GPU.device_name} ==="

# --- Hyperparameters ---
HIDDEN  = 128
CLASSES = 10
INPUT   = 784
LR      = 0.01
EPOCHS  = 5
N_TRAIN = 60000

# --- Weight Initialization (Xavier) ---
srand(42)

def xavier(fan_in, fan_out)
  scale = Math.sqrt(2.0 / (fan_in + fan_out))
  Array.new(fan_out * fan_in) { (rand - 0.5) * 2 * scale }
end

puts "DEBUG: xavier w1..."
w1 = GPU.array(xavier(INPUT, HIDDEN))     # 128 x 784
puts "DEBUG: w1 done"
b1 = GPU.array(Array.new(HIDDEN, 0.0))    # 128
puts "DEBUG: b1 done"
w2 = GPU.array(xavier(HIDDEN, CLASSES))   # 10 x 128
puts "DEBUG: w2 done"
b2 = GPU.array(Array.new(CLASSES, 0.0))   # 10
puts "DEBUG: b2 done"

# --- Load labels ---
puts "DEBUG: loading labels..."
labels_buf = GPU.load("data/train_labels.bin")
puts "DEBUG: labels loaded"
labels_all = labels_buf.head(N_TRAIN).map { |v| v.to_i }
puts "DEBUG: labels parsed"

# --- Training Loop ---
EPOCHS.times do |epoch|
  correct = 0
  loss_sum = 0.0

  N_TRAIN.times do |i|
    if i == 0
      puts "DEBUG: loading image 0..."
    end
    # Load one image (784 floats) from concatenated file
    x = GPU.load("data/train_images.bin", i * INPUT, INPUT)
    if i == 0
      puts "DEBUG: image loaded, matmul w1*x..."
    end

    # === Forward ===
    # Layer 1: h = ReLU(W1 @ x + b1)
    z1 = GPU.matmul(w1, x, HIDDEN, INPUT, 1)
    if i == 0
      puts "DEBUG: matmul done, add b1..."
    end
    h_pre = GPU.add(z1, b1)
    if i == 0
      puts "DEBUG: add done, relu..."
    end
    h = GPU.relu(h_pre)

    if i == 0
      puts "DEBUG: relu done, layer2..."
    end
    # Layer 2: o = W2 @ h + b2
    o = GPU.add(GPU.matmul(w2, h, CLASSES, HIDDEN, 1), b2)
    if i == 0
      puts "DEBUG: layer2 done, softmax..."
    end

    # Softmax (Ruby side - only 10 elements)
    scores = o.head(CLASSES)
    max_s = scores.max
    exps = scores.map { |s| Math.exp(s - max_s) }
    sum_e = exps.sum
    probs = exps.map { |e| e / sum_e }

    label = labels_all[i]
    predicted = probs.each_with_index.max_by { |v, _| v }[1]
    correct += 1 if predicted == label

    # Cross-entropy loss
    loss_sum += -Math.log(probs[label] + 1e-8)

    if i == 0
      puts "DEBUG: softmax done, backward..."
    end
    # === Backward ===
    # dL/do = probs - one_hot(label)
    grad_o_data = probs.dup
    grad_o_data[label] -= 1.0
    grad_o = GPU.array(grad_o_data)

    # dL/dW2 = grad_o (10x1) @ h^T (1x128) => 10x128
    grad_w2 = GPU.matmul_nt(grad_o, h, CLASSES, 1, HIDDEN)

    # dL/db2 = grad_o
    grad_b2 = grad_o

    # dL/dh = W2^T (128x10) @ grad_o (10x1) => 128x1
    grad_h = GPU.matmul_tn(w2, grad_o, HIDDEN, CLASSES, 1)

    # ReLU backward: mask where h > 0
    h_vals = h.head(HIDDEN)
    mask_data = h_vals.map { |v| v > 0 ? 1.0 : 0.0 }
    mask = GPU.array(mask_data)
    grad_h_pre = GPU.mul(grad_h, mask)

    # dL/dW1 = grad_h_pre (128x1) @ x^T (1x784) => 128x784
    grad_w1 = GPU.matmul_nt(grad_h_pre, x, HIDDEN, 1, INPUT)

    # dL/db1 = grad_h_pre
    grad_b1 = grad_h_pre

    if i == 0
      puts "DEBUG: backward done, SGD update..."
    end
    # === SGD Update ===
    w1 = GPU.sub(w1, GPU.scale(grad_w1, LR))
    b1 = GPU.sub(b1, GPU.scale(grad_b1, LR))
    w2 = GPU.sub(w2, GPU.scale(grad_w2, LR))
    b2 = GPU.sub(b2, GPU.scale(grad_b2, LR))

    # Progress
    if (i + 1) % 10000 == 0
      acc = (correct.to_f / (i + 1) * 100).round(1)
      avg_loss = (loss_sum / (i + 1)).round(4)
      puts "  [#{i + 1}/#{N_TRAIN}] loss=#{avg_loss} acc=#{acc}%"
    end
  end

  acc = (correct.to_f / N_TRAIN * 100).round(1)
  avg_loss = (loss_sum / N_TRAIN).round(4)
  puts "Epoch #{epoch + 1}/#{EPOCHS}: loss=#{avg_loss} accuracy=#{acc}%"
end

# --- Save weights ---
puts "Saving weights..."
w1.save("weights/fc1_weight.bin")
b1.save("weights/fc1_bias.bin")
w2.save("weights/fc2_weight.bin")
b2.save("weights/fc2_bias.bin")
puts "Done."
