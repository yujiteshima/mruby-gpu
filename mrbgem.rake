MRuby::Gem::Specification.new('mruby-gpu') do |spec|
  spec.license = 'MIT'
  spec.authors = 'Yuji Teshima'
  spec.summary = 'GPU compute via Vulkan for mruby'
  spec.linker.libraries << 'vulkan'
end
