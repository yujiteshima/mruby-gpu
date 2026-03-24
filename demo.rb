puts "=" * 50
puts " mruby + Vulkan Compute Shader Demo"
puts " Raspberry Pi 5 - GPU General Computing"
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

# 2. Scale: 1M elements on GPU
puts "--- 1M Elements on GPU ---"
a = GPU.fill(1_000_000, 1.0)
b = GPU.fill(1_000_000, 2.0)

times = []
5.times do
  t0 = Time.now
  c = GPU.add(a, b)
  t1 = Time.now
  times << (t1 - t0) * 1000
end

avg = times.inject(0.0){|s,t| s + t} / times.size
puts "1,000,000 elements x 5 runs"
puts "Times: #{times.map{|t| "#{t.round(2)}ms"}.join(", ")}"
puts "Avg:   #{avg.round(2)} ms"
puts "Result: #{c.head(4).inspect}"
puts

puts "Done! mruby -> C(mrbgem) -> Vulkan -> GPU"
