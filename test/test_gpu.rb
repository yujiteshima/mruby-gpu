# test/test_gpu.rb -- CPU/GPU backend parity tests
GPU.init("shader")

$pass = 0
$fail = 0

def assert_close(label, expected, actual, eps = 0.001)
  if expected.size != actual.size
    $fail += 1
    puts "FAIL #{label}: size mismatch (expected #{expected.size}, got #{actual.size})"
    return
  end
  expected.size.times do |i|
    diff = (expected[i] - actual[i]).abs
    if diff > eps
      $fail += 1
      puts "FAIL #{label}: index #{i} expected #{expected[i]} got #{actual[i]} (diff=#{diff})"
      return
    end
  end
  $pass += 1
  puts "PASS #{label}"
end

# --- Test data ---
a = GPU.array([1.0, 2.0, 3.0, 4.0])
b = GPU.array([10.0, 20.0, 30.0, 40.0])

# --- backend getter/setter ---
GPU.backend = :vulkan
if GPU.backend == "Vulkan"
  $pass += 1; puts "PASS backend=:vulkan returns 'Vulkan'"
else
  $fail += 1; puts "FAIL backend=:vulkan returns '#{GPU.backend}'"
end

GPU.backend = :cpu
if GPU.backend == "CPU"
  $pass += 1; puts "PASS backend=:cpu returns 'CPU'"
else
  $fail += 1; puts "FAIL backend=:cpu returns '#{GPU.backend}'"
end

# --- add ---
GPU.backend = :vulkan
gpu_r = GPU.add(a, b).head(4)
GPU.backend = :cpu
cpu_r = GPU.add(a, b).head(4)
assert_close("add parity", gpu_r, cpu_r)
assert_close("add values", [11.0, 22.0, 33.0, 44.0], cpu_r)

# --- sub ---
GPU.backend = :vulkan
gpu_r = GPU.sub(a, b).head(4)
GPU.backend = :cpu
cpu_r = GPU.sub(a, b).head(4)
assert_close("sub parity", gpu_r, cpu_r)
assert_close("sub values", [-9.0, -18.0, -27.0, -36.0], cpu_r)

# --- mul ---
GPU.backend = :vulkan
gpu_r = GPU.mul(a, b).head(4)
GPU.backend = :cpu
cpu_r = GPU.mul(a, b).head(4)
assert_close("mul parity", gpu_r, cpu_r)
assert_close("mul values", [10.0, 40.0, 90.0, 160.0], cpu_r)

# --- scale ---
GPU.backend = :vulkan
gpu_r = GPU.scale(a, 2.5).head(4)
GPU.backend = :cpu
cpu_r = GPU.scale(a, 2.5).head(4)
assert_close("scale parity", gpu_r, cpu_r)
assert_close("scale values", [2.5, 5.0, 7.5, 10.0], cpu_r)

# --- relu ---
neg = GPU.array([-1.0, 2.0, -3.0, 4.0])
GPU.backend = :vulkan
gpu_r = GPU.relu(neg).head(4)
GPU.backend = :cpu
cpu_r = GPU.relu(neg).head(4)
assert_close("relu parity", gpu_r, cpu_r)
assert_close("relu values", [0.0, 2.0, 0.0, 4.0], cpu_r)

# --- matmul (2x3 * 3x2 = 2x2) ---
# A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
# C = [[58,64],[139,154]]
ma = GPU.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
mb = GPU.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
GPU.backend = :vulkan
gpu_r = GPU.matmul(ma, mb, 2, 3, 2).head(4)
GPU.backend = :cpu
cpu_r = GPU.matmul(ma, mb, 2, 3, 2).head(4)
assert_close("matmul parity", gpu_r, cpu_r)
assert_close("matmul values", [58.0, 64.0, 139.0, 154.0], cpu_r)

# --- matmul_tn (transpose A) ---
# A stored as K*M (3x2): columns of original A become rows
ma_t = GPU.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
GPU.backend = :vulkan
gpu_r = GPU.matmul_tn(ma_t, mb, 2, 3, 2).head(4)
GPU.backend = :cpu
cpu_r = GPU.matmul_tn(ma_t, mb, 2, 3, 2).head(4)
assert_close("matmul_tn parity", gpu_r, cpu_r)
assert_close("matmul_tn values", [58.0, 64.0, 139.0, 154.0], cpu_r)

# --- matmul_nt (transpose B) ---
# B stored as N*K (2x3): columns of original B become rows
mb_t = GPU.array([7.0, 9.0, 11.0, 8.0, 10.0, 12.0])
GPU.backend = :vulkan
gpu_r = GPU.matmul_nt(ma, mb_t, 2, 3, 2).head(4)
GPU.backend = :cpu
cpu_r = GPU.matmul_nt(ma, mb_t, 2, 3, 2).head(4)
assert_close("matmul_nt parity", gpu_r, cpu_r)
assert_close("matmul_nt values", [58.0, 64.0, 139.0, 154.0], cpu_r)

# --- summary ---
GPU.backend = :vulkan  # restore default
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
if $fail > 0
  puts "SOME TESTS FAILED"
else
  puts "ALL TESTS PASSED"
end
