/* gpu_ops.c -- mruby method definitions for GPU module operations + gem init/final */
#include "gpu_internal.h"

/* Buffer methods (defined in gpu_buffer.c) */
mrb_value mrb_gpu_buffer_head(mrb_state *mrb, mrb_value self);
mrb_value mrb_gpu_buffer_size(mrb_state *mrb, mrb_value self);
mrb_value mrb_gpu_buffer_save(mrb_state *mrb, mrb_value self);

/* ---- mruby: GPU.init(shader_dir) ---- */
static mrb_value mrb_gpu_init(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_get_args(mrb, "z", &path);
  gpu_init(path);
  return mrb_nil_value();
}

/* ---- mruby: GPU.array(ary) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_array(mrb_state *mrb, mrb_value self) {
  mrb_value ary;
  mrb_get_args(mrb, "A", &ary);

  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "GPU not initialized. Call GPU.init first.");
  }

  uint32_t n = (uint32_t)RARRAY_LEN(ary);
  GpuBuffer *buf = create_buffer(mrb, n);

  float *mapped;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);
  for (uint32_t i = 0; i < n; i++) {
    mapped[i] = (float)mrb_float(mrb_ary_ref(mrb, ary, i));
  }
  vkUnmapMemory(g_ctx.device, buf->memory);

  return wrap_buffer(mrb, buf);
}

/* ---- mruby: GPU.fill(n, value) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_fill(mrb_state *mrb, mrb_value self) {
  mrb_int n;
  mrb_float val;
  mrb_get_args(mrb, "if", &n, &val);

  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "GPU not initialized. Call GPU.init first.");
  }

  GpuBuffer *buf = create_buffer(mrb, (uint32_t)n);
  float *mapped;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);
  for (uint32_t i = 0; i < (uint32_t)n; i++) {
    mapped[i] = (float)val;
  }
  vkUnmapMemory(g_ctx.device, buf->memory);

  return wrap_buffer(mrb, buf);
}

/* ---- mruby: GPU.load(path [, offset, count]) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_load(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_int offset = 0, count = -1;
  mrb_get_args(mrb, "z|ii", &path, &offset, &count);

  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "GPU not initialized. Call GPU.init first.");
  }

  FILE *f = fopen(path, "rb");
  if (!f) {
    mrb_raisef(mrb, E_RUNTIME_ERROR, "Cannot open file: %s", path);
  }

  if (count < 0) {
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    count = (file_size / (long)sizeof(float)) - offset;
  }

  GpuBuffer *buf = create_buffer(mrb, (uint32_t)count);

  float *mapped;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);
  fseek(f, (long)(offset * sizeof(float)), SEEK_SET);
  fread(mapped, sizeof(float), (size_t)count, f);
  vkUnmapMemory(g_ctx.device, buf->memory);
  fclose(f);

  return wrap_buffer(mrb, buf);
}

/* ---- 3-buffer ops: add, sub, mul ---- */
static mrb_value mrb_gpu_binop3(mrb_state *mrb, mrb_value self, PipeId pipe) {
  mrb_value va, vb;
  mrb_get_args(mrb, "oo", &va, &vb);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  GpuBuffer *b = DATA_GET_PTR(mrb, vb, &gpu_buffer_type, GpuBuffer);

  if (a->n != b->n) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "Buffer size mismatch");
  }

  GpuBuffer *c = create_buffer(mrb, a->n);

  VkBuffer bufs[3] = {a->buffer, b->buffer, c->buffer};
  VkDeviceSize sizes[3] = {a->bytes, b->bytes, c->bytes};
  uint32_t push = a->n;

  dispatch_compute(pipe, bufs, sizes, 3, &push, sizeof(uint32_t),
    (a->n + 255) / 256, 1, 1);

  return wrap_buffer(mrb, c);
}

static mrb_value mrb_gpu_add(mrb_state *mrb, mrb_value self) {
  return mrb_gpu_binop3(mrb, self, PIPE_ADD);
}

static mrb_value mrb_gpu_sub(mrb_state *mrb, mrb_value self) {
  return mrb_gpu_binop3(mrb, self, PIPE_SUB);
}

static mrb_value mrb_gpu_mul(mrb_state *mrb, mrb_value self) {
  return mrb_gpu_binop3(mrb, self, PIPE_MUL);
}

/* ---- mruby: GPU.scale(a, scalar) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_scale(mrb_state *mrb, mrb_value self) {
  mrb_value va;
  mrb_float scalar;
  mrb_get_args(mrb, "of", &va, &scalar);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  GpuBuffer *b = create_buffer(mrb, a->n);

  VkBuffer bufs[2] = {a->buffer, b->buffer};
  VkDeviceSize sizes[2] = {a->bytes, b->bytes};

  struct { uint32_t n; float scalar; } push = {a->n, (float)scalar};

  dispatch_compute(PIPE_SCALE, bufs, sizes, 2, &push, sizeof(push),
    (a->n + 255) / 256, 1, 1);

  return wrap_buffer(mrb, b);
}

/* ---- mruby: GPU.relu(a) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_relu(mrb_state *mrb, mrb_value self) {
  mrb_value va;
  mrb_get_args(mrb, "o", &va);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  GpuBuffer *b = create_buffer(mrb, a->n);

  VkBuffer bufs[2] = {a->buffer, b->buffer};
  VkDeviceSize sizes[2] = {a->bytes, b->bytes};
  uint32_t push = a->n;

  dispatch_compute(PIPE_RELU, bufs, sizes, 2, &push, sizeof(uint32_t),
    (a->n + 255) / 256, 1, 1);

  return wrap_buffer(mrb, b);
}

/* ---- matmul helper (flags: 0=normal, 1=transpose_A, 2=transpose_B) ---- */
static mrb_value mrb_gpu_matmul_impl(mrb_state *mrb, mrb_value self, uint32_t flags) {
  mrb_value va, vb;
  mrb_int m, k, n;
  mrb_get_args(mrb, "ooiii", &va, &vb, &m, &k, &n);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  GpuBuffer *b = DATA_GET_PTR(mrb, vb, &gpu_buffer_type, GpuBuffer);

  /* Validate sizes based on transpose flags */
  uint32_t expected_a = (flags & 1) ? (uint32_t)(k * m) : (uint32_t)(m * k);
  uint32_t expected_b = (flags & 2) ? (uint32_t)(n * k) : (uint32_t)(k * n);

  if (a->n != expected_a) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "Buffer A size mismatch: expected %d, got %d",
      (int)expected_a, (int)a->n);
  }
  if (b->n != expected_b) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "Buffer B size mismatch: expected %d, got %d",
      (int)expected_b, (int)b->n);
  }

  GpuBuffer *c = create_buffer(mrb, (uint32_t)(m * n));

  VkBuffer bufs[3] = {a->buffer, b->buffer, c->buffer};
  VkDeviceSize sizes[3] = {a->bytes, b->bytes, c->bytes};

  uint32_t push[4] = {(uint32_t)m, (uint32_t)k, (uint32_t)n, flags};

  dispatch_compute(PIPE_MATMUL, bufs, sizes, 3, push, sizeof(push),
    ((uint32_t)n + 15) / 16, ((uint32_t)m + 15) / 16, 1);

  return wrap_buffer(mrb, c);
}

static mrb_value mrb_gpu_matmul(mrb_state *mrb, mrb_value self) {
  return mrb_gpu_matmul_impl(mrb, self, 0);
}

static mrb_value mrb_gpu_matmul_tn(mrb_state *mrb, mrb_value self) {
  return mrb_gpu_matmul_impl(mrb, self, 1);
}

static mrb_value mrb_gpu_matmul_nt(mrb_state *mrb, mrb_value self) {
  return mrb_gpu_matmul_impl(mrb, self, 2);
}

/* ---- mruby: GPU.backend ---- */
static mrb_value mrb_gpu_backend(mrb_state *mrb, mrb_value self) {
  return mrb_str_new_cstr(mrb, "Vulkan");
}

/* ---- mruby: GPU.device_name ---- */
static mrb_value mrb_gpu_device_name(mrb_state *mrb, mrb_value self) {
  if (!g_ctx.initialized) return mrb_str_new_cstr(mrb, "(not initialized)");
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);
  return mrb_str_new_cstr(mrb, props.deviceName);
}

/* ---- forward declarations for sub-gem inits ---- */
void mrb_camera_gem_init(mrb_state *mrb);
void mrb_camera_gem_final(mrb_state *mrb);
void mrb_face_gem_init(mrb_state *mrb);
void mrb_face_gem_final(mrb_state *mrb);
void mrb_display_gem_init(mrb_state *mrb);
void mrb_display_gem_final(mrb_state *mrb);

/* ---- gem init ---- */
void mrb_mruby_gpu_gem_init(mrb_state *mrb) {
  struct RClass *gpu = mrb_define_module(mrb, "GPU");
  mrb_define_module_function(mrb, gpu, "init", mrb_gpu_init, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "array", mrb_gpu_array, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "fill", mrb_gpu_fill, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, gpu, "load", mrb_gpu_load, MRB_ARGS_REQ(1) | MRB_ARGS_OPT(2));
  mrb_define_module_function(mrb, gpu, "add", mrb_gpu_add, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, gpu, "sub", mrb_gpu_sub, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, gpu, "mul", mrb_gpu_mul, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, gpu, "scale", mrb_gpu_scale, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, gpu, "relu", mrb_gpu_relu, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "matmul", mrb_gpu_matmul, MRB_ARGS_REQ(5));
  mrb_define_module_function(mrb, gpu, "matmul_tn", mrb_gpu_matmul_tn, MRB_ARGS_REQ(5));
  mrb_define_module_function(mrb, gpu, "matmul_nt", mrb_gpu_matmul_nt, MRB_ARGS_REQ(5));
  mrb_define_module_function(mrb, gpu, "backend", mrb_gpu_backend, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "device_name", mrb_gpu_device_name, MRB_ARGS_NONE());

  struct RClass *buf_class = mrb_define_class_under(mrb, gpu, "Buffer", mrb->object_class);
  MRB_SET_INSTANCE_TT(buf_class, MRB_TT_CDATA);
  mrb_define_method(mrb, buf_class, "head", mrb_gpu_buffer_head, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, buf_class, "size", mrb_gpu_buffer_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, buf_class, "save", mrb_gpu_buffer_save, MRB_ARGS_REQ(1));

  mrb_camera_gem_init(mrb);
  mrb_face_gem_init(mrb);
  mrb_display_gem_init(mrb);
}

void mrb_mruby_gpu_gem_final(mrb_state *mrb) {
  mrb_display_gem_final(mrb);
  mrb_face_gem_final(mrb);
  mrb_camera_gem_final(mrb);
  if (g_ctx.initialized) {
    vkDestroyDescriptorPool(g_ctx.device, g_ctx.desc_pool, NULL);
    for (int p = 0; p < PIPE_COUNT; p++) {
      if (g_ctx.pipelines[p] != VK_NULL_HANDLE) {
        vkDestroyPipeline(g_ctx.device, g_ctx.pipelines[p], NULL);
      }
    }
    for (int l = 0; l < LAYOUT_COUNT; l++) {
      vkDestroyPipelineLayout(g_ctx.device, g_ctx.pipe_layouts[l], NULL);
      vkDestroyDescriptorSetLayout(g_ctx.device, g_ctx.desc_layouts[l], NULL);
    }
    vkDestroyCommandPool(g_ctx.device, g_ctx.cmd_pool, NULL);
    vkDestroyDevice(g_ctx.device, NULL);
    vkDestroyInstance(g_ctx.instance, NULL);
    g_ctx.initialized = 0;
  }
}
