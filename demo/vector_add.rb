puts "=" * 50
puts " mruby + Vulkan Compute Shader Demo"
puts " Raspberry Pi 5 - GPU vs CPU Benchmark"
puts "=" * 50
puts

GPU.init("shader")
puts "Backend: #{GPU.backend}"
puts "Device:  #{GPU.device_name}"
puts

# 1. Basic: mruby array -> GPU -> result
puts "--- Basic Vector Add ---"
a = GPU.array([1.0, 2.0, 3.0, 4.0])
b = GPU.array([10.0, 20.0, 30.0, 40.0])
c = GPU.add(a, b)
puts "a = [1, 2, 3, 4]"
puts "b = [10, 20, 30, 40]"
puts "GPU.add(a, b) => #{c.head(4).inspect}"
puts

# 2. GPU: 1M elements
puts "--- 1,000,000 Elements: GPU (Vulkan) ---"
a_gpu = GPU.fill(1_000_000, 1.0)
b_gpu = GPU.fill(1_000_000, 2.0)

gpu_times = []
5.times do
  t0 = Time.now
  GPU.add(a_gpu, b_gpu)
  gpu_times << (Time.now - t0) * 1000
end
gpu_avg = gpu_times.inject(0.0, :+) / gpu_times.size

puts "  Times: #{gpu_times.map{|t| "#{t.round(2)}ms"}.join(", ")}"
puts "  Avg:   #{gpu_avg.round(2)} ms"
puts

# 3. CPU: 100K elements (pure mruby) — 同じ演算をCPUで
puts "--- 100,000 Elements: CPU (pure mruby) ---"
n_cpu = 100_000
a_cpu = Array.new(n_cpu, 1.0)
b_cpu = Array.new(n_cpu, 2.0)

cpu_times = []
3.times do
  t0 = Time.now
  c_cpu = Array.new(n_cpu) { |i| a_cpu[i] + b_cpu[i] }
  cpu_times << (Time.now - t0) * 1000
end
cpu_avg = cpu_times.inject(0.0, :+) / cpu_times.size

puts "  Times: #{cpu_times.map{|t| "#{t.round(0).to_i}ms"}.join(", ")}"
puts "  Avg:   #{cpu_avg.round(0).to_i} ms  (#{n_cpu / 1000}K elements)"
puts

# 4. 比較
# CPU は 100K で計測 → 1M 相当に換算
cpu_1m = cpu_avg * 10
speedup = cpu_1m / gpu_avg

puts "--- 比較 (1,000,000 要素換算) ---"
puts "  GPU : #{gpu_avg.round(1)} ms"
puts "  CPU : #{cpu_1m.round(0).to_i} ms  (100K × 10 で推定)"
puts
puts "  => GPU is #{speedup.round(0).to_i}x faster!"
puts
puts "Done! mruby -> GPU/CPU, choose the right tool for the job."
