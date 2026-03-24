#!/usr/bin/env ruby
# Converts MNIST IDX binary files to raw float32 (.bin) for mruby-gpu.
# No external gems required.

SCRIPT_DIR = File.dirname(__FILE__)
DATA_DIR = File.join(File.dirname(SCRIPT_DIR), "data")

def read_idx_images(path)
  data = File.binread(path)
  _magic, count, rows, cols = data[0, 16].unpack("N4")
  pixels = data[16..].unpack("C*")
  puts "  #{count} images (#{rows}x#{cols})"
  pixels.map { |p| p / 255.0 }
end

def read_idx_labels(path)
  data = File.binread(path)
  _magic, count = data[0, 8].unpack("N2")
  labels = data[8..].unpack("C*")
  puts "  #{count} labels"
  labels
end

# --- Training data ---
puts "Reading training images..."
train_images = read_idx_images(File.join(DATA_DIR, "train-images-idx3-ubyte"))
puts "Reading training labels..."
train_labels = read_idx_labels(File.join(DATA_DIR, "train-labels-idx1-ubyte"))

# All images concatenated: 60000 x 784 floats
puts "Writing train_images.bin..."
File.binwrite(File.join(DATA_DIR, "train_images.bin"), train_images.pack("e*"))

# Labels as float32 (integer values 0-9 stored as floats)
puts "Writing train_labels.bin..."
File.binwrite(File.join(DATA_DIR, "train_labels.bin"), train_labels.map(&:to_f).pack("e*"))

# --- Test data ---
puts "Reading test images..."
test_images = read_idx_images(File.join(DATA_DIR, "t10k-images-idx3-ubyte"))
puts "Reading test labels..."
test_labels = read_idx_labels(File.join(DATA_DIR, "t10k-labels-idx1-ubyte"))

# Individual test images for inference demo
puts "Writing individual test images..."
10.times do |i|
  slice = test_images[i * 784, 784]
  File.binwrite(File.join(DATA_DIR, "test_img_#{i}.bin"), slice.pack("e*"))
end
File.write(File.join(DATA_DIR, "test_labels.txt"), test_labels[0, 10].join("\n") + "\n")

puts "Done. Output files in #{DATA_DIR}:"
Dir.glob(File.join(DATA_DIR, "*.bin")).sort.each { |f| puts "  #{File.basename(f)} (#{File.size(f)} bytes)" }
