GPU.init("shader")

puts "=== mruby + Vulkan Compute on #{GPU.device_name} ==="

n = 10_000
data_a = []
data_b = []
n.times do |i|
  data_a << (i % 100).to_f
  data_b << ((i % 50) * 2).to_f
end

t0 = Time.now
a = GPU.array(data_a)
b = GPU.array(data_b)
t1 = Time.now

c = GPU.add(a, b)
t2 = Time.now

result = c.head(4)
t3 = Time.now

puts "Elements:  #{c.size}"
puts "Transfer:  #{((t1 - t0) * 1000).round(1)} ms"
puts "Compute:   #{((t2 - t1) * 1000).round(1)} ms"
puts "Readback:  #{((t3 - t2) * 1000).round(1)} ms"
puts "Result:    #{result.inspect}"
