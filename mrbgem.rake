MRuby::Gem::Specification.new('mruby-gpu') do |spec|
  spec.license = 'MIT'
  spec.authors = 'Yuji Teshima'
  spec.summary = 'GPU compute via Vulkan for mruby (+ Camera / FaceDetector / Display)'

  # Vulkan
  spec.linker.libraries << 'vulkan'

  # NCNN (face detection)
  spec.linker.libraries << 'ncnn'
  spec.linker.libraries << 'glslang'
  spec.linker.libraries << 'SPIRV'
  spec.linker.libraries << 'MachineIndependent'
  spec.linker.libraries << 'GenericCodeGen'
  spec.linker.libraries << 'glslang-default-resource-limits'
  spec.linker.library_paths << '/usr/local/lib'

  # OpenMP (required by NCNN)
  spec.linker.flags_after_libraries << '-fopenmp'

  # SDL2 (Display class)
  spec.linker.libraries << 'SDL2'

  # Headers
  spec.cc.include_paths  << '/usr/local/include/ncnn'
  spec.cxx.include_paths << '/usr/local/include/ncnn'

  # C++ standard required by NCNN
  spec.cxx.flags << '-std=c++17'
end
