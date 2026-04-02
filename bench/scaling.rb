GPU.init("shader")
puts "=== mruby + Vulkan Compute on #{GPU.device_name} ==="

[1_000, 10_000, 100_000, 1_000_000].each do |n|
  a = GPU.fill(n, 1.0)
  b = GPU.fill(n, 2.0)

  t0 = Time.now
  c = GPU.add(a, b)
  t1 = Time.now

  puts "N=#{n.to_s.rjust(10)}: #{((t1 - t0) * 1000).round(2)} ms  head=#{c.head(3).inspect}"
end
