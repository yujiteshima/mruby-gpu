GPU.init("shader")
puts "Backend: #{GPU.backend}"
puts "Device:  #{GPU.device_name}"

a = GPU.array([1.0, 2.0, 3.0])
b = GPU.array([4.0, 5.0, 6.0])
c = GPU.add(a, b)

puts "Result:  #{c.head(3).inspect}"
