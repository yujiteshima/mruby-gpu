GPU.init("/home/ubuntu/work/mruby-gpu/shader/add.spv")

puts "Backend: #{GPU.backend}"
puts "Device:  #{GPU.device_name}"

a = GPU.array([1.0, 2.0, 3.0, 4.0])
b = GPU.array([10.0, 20.0, 30.0, 40.0])
c = GPU.add(a, b)

puts "Result: #{c.head(4).inspect}"
puts "Size:   #{c.size}"
