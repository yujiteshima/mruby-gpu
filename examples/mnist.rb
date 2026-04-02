# MNIST Inference on GPU via mruby + Vulkan Compute
# Loads pre-trained weights and classifies test images.

GPU.init("shader")
puts "=== MNIST Inference on #{GPU.device_name} ==="
puts "Backend: #{GPU.backend}"

# Load trained weights
fc1_w = GPU.load("weights/fc1_weight.bin")  # 128 x 784
fc1_b = GPU.load("weights/fc1_bias.bin")    # 128
fc2_w = GPU.load("weights/fc2_weight.bin")  # 10 x 128
fc2_b = GPU.load("weights/fc2_bias.bin")    # 10

puts "Weights loaded. fc1=#{fc1_w.size} fc2=#{fc2_w.size}"

# Load ground truth labels
labels = File.read("data/test_labels.txt").split("\n").map(&:to_i) rescue []

correct = 0
total = 10

total.times do |i|
  # Load test image
  x = GPU.load("data/test_img_#{i}.bin")  # 784 floats

  # Forward pass
  h = GPU.relu(GPU.add(GPU.matmul(fc1_w, x, 128, 784, 1), fc1_b))
  o = GPU.add(GPU.matmul(fc2_w, h, 10, 128, 1), fc2_b)

  # Softmax (Ruby side)
  scores = o.head(10)
  max_s = scores.max
  exps = scores.map { |s| Math.exp(s - max_s) }
  sum_e = exps.sum
  probs = exps.map { |e| e / sum_e }

  predicted = probs.each_with_index.max_by { |v, _| v }[1]
  confidence = (probs[predicted] * 100).round(1)

  label_str = labels[i] ? " (label=#{labels[i]}#{predicted == labels[i] ? ' OK' : ' NG'})" : ""
  correct += 1 if labels[i] && predicted == labels[i]

  puts "Image #{i}: predicted=#{predicted} confidence=#{confidence}%#{label_str}"
end

if labels.size >= total
  puts "Accuracy: #{correct}/#{total} (#{(correct.to_f / total * 100).round(1)}%)"
end
