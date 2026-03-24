GLSLC ?= glslangValidator

SHADERS = add.comp mul.comp sub.comp relu.comp scale.comp matmul.comp
SPIRV = $(SHADERS:.comp=.spv)

all: $(SPIRV)

%.spv: %.comp
	$(GLSLC) -V $< -o $@

clean:
	rm -f $(SPIRV)

.PHONY: all clean
