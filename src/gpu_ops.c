/* gpu_ops.c -- mruby method definitions for GPU module operations + gem init/final */
#include "gpu_internal.h"
#include <string.h>
#include <time.h>
#include <stdio.h>

/* ---- Backend selection: 0 = Vulkan (default), 1 = CPU ---- */
static int g_use_cpu = 0;

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

/* ---- CPU backend: element-wise binary op ---- */
static void cpu_binop3(GpuBuffer *a, GpuBuffer *b, GpuBuffer *c, PipeId pipe) {
  float *pa, *pb, *pc;
  vkMapMemory(g_ctx.device, a->memory, 0, a->bytes, 0, (void **)&pa);
  vkMapMemory(g_ctx.device, b->memory, 0, b->bytes, 0, (void **)&pb);
  vkMapMemory(g_ctx.device, c->memory, 0, c->bytes, 0, (void **)&pc);
  for (uint32_t i = 0; i < a->n; i++) {
    switch (pipe) {
      case PIPE_ADD: pc[i] = pa[i] + pb[i]; break;
      case PIPE_SUB: pc[i] = pa[i] - pb[i]; break;
      case PIPE_MUL: pc[i] = pa[i] * pb[i]; break;
      default: break;
    }
  }
  vkUnmapMemory(g_ctx.device, a->memory);
  vkUnmapMemory(g_ctx.device, b->memory);
  vkUnmapMemory(g_ctx.device, c->memory);
}

/* ---- CPU backend: scale ---- */
static void cpu_scale(GpuBuffer *a, GpuBuffer *b, float scalar) {
  float *pa, *pb;
  vkMapMemory(g_ctx.device, a->memory, 0, a->bytes, 0, (void **)&pa);
  vkMapMemory(g_ctx.device, b->memory, 0, b->bytes, 0, (void **)&pb);
  for (uint32_t i = 0; i < a->n; i++) pb[i] = pa[i] * scalar;
  vkUnmapMemory(g_ctx.device, a->memory);
  vkUnmapMemory(g_ctx.device, b->memory);
}

/* ---- CPU backend: relu ---- */
static void cpu_relu(GpuBuffer *a, GpuBuffer *b) {
  float *pa, *pb;
  vkMapMemory(g_ctx.device, a->memory, 0, a->bytes, 0, (void **)&pa);
  vkMapMemory(g_ctx.device, b->memory, 0, b->bytes, 0, (void **)&pb);
  for (uint32_t i = 0; i < a->n; i++) pb[i] = pa[i] > 0.0f ? pa[i] : 0.0f;
  vkUnmapMemory(g_ctx.device, a->memory);
  vkUnmapMemory(g_ctx.device, b->memory);
}

/* ---- CPU backend: matmul (transpose flags match shader/matmul.comp) ---- */
static void cpu_matmul(GpuBuffer *a, GpuBuffer *b, GpuBuffer *c,
                       uint32_t M, uint32_t K, uint32_t N, uint32_t flags) {
  float *pa, *pb, *pc;
  vkMapMemory(g_ctx.device, a->memory, 0, a->bytes, 0, (void **)&pa);
  vkMapMemory(g_ctx.device, b->memory, 0, b->bytes, 0, (void **)&pb);
  vkMapMemory(g_ctx.device, c->memory, 0, c->bytes, 0, (void **)&pc);
  for (uint32_t row = 0; row < M; row++) {
    for (uint32_t col = 0; col < N; col++) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; k++) {
        float va = (flags & 1u) ? pa[k * M + row] : pa[row * K + k];
        float vb = (flags & 2u) ? pb[col * K + k] : pb[k * N + col];
        sum += va * vb;
      }
      pc[row * N + col] = sum;
    }
  }
  vkUnmapMemory(g_ctx.device, a->memory);
  vkUnmapMemory(g_ctx.device, b->memory);
  vkUnmapMemory(g_ctx.device, c->memory);
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

  if (g_use_cpu) {
    cpu_binop3(a, b, c, pipe);
  } else {
    VkBuffer bufs[3] = {a->buffer, b->buffer, c->buffer};
    VkDeviceSize sizes[3] = {a->bytes, b->bytes, c->bytes};
    uint32_t push = a->n;
    dispatch_compute(pipe, bufs, sizes, 3, &push, sizeof(uint32_t),
      (a->n + 255) / 256, 1, 1);
  }

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

  if (g_use_cpu) {
    cpu_scale(a, b, (float)scalar);
  } else {
    VkBuffer bufs[2] = {a->buffer, b->buffer};
    VkDeviceSize sizes[2] = {a->bytes, b->bytes};
    struct { uint32_t n; float scalar; } push = {a->n, (float)scalar};
    dispatch_compute(PIPE_SCALE, bufs, sizes, 2, &push, sizeof(push),
      (a->n + 255) / 256, 1, 1);
  }

  return wrap_buffer(mrb, b);
}

/* ---- mruby: GPU.relu(a) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_relu(mrb_state *mrb, mrb_value self) {
  mrb_value va;
  mrb_get_args(mrb, "o", &va);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  GpuBuffer *b = create_buffer(mrb, a->n);

  if (g_use_cpu) {
    cpu_relu(a, b);
  } else {
    VkBuffer bufs[2] = {a->buffer, b->buffer};
    VkDeviceSize sizes[2] = {a->bytes, b->bytes};
    uint32_t push = a->n;
    dispatch_compute(PIPE_RELU, bufs, sizes, 2, &push, sizeof(uint32_t),
      (a->n + 255) / 256, 1, 1);
  }

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

  if (g_use_cpu) {
    cpu_matmul(a, b, c, (uint32_t)m, (uint32_t)k, (uint32_t)n, flags);
  } else {
    VkBuffer bufs[3] = {a->buffer, b->buffer, c->buffer};
    VkDeviceSize sizes[3] = {a->bytes, b->bytes, c->bytes};
    uint32_t push[4] = {(uint32_t)m, (uint32_t)k, (uint32_t)n, flags};
    dispatch_compute(PIPE_MATMUL, bufs, sizes, 3, push, sizeof(push),
      ((uint32_t)n + 15) / 16, ((uint32_t)m + 15) / 16, 1);
  }

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
  return mrb_str_new_cstr(mrb, g_use_cpu ? "CPU" : "Vulkan");
}

/* ---- mruby: GPU.backend=(sym) ---- */
static mrb_value mrb_gpu_set_backend(mrb_state *mrb, mrb_value self) {
  mrb_value val;
  mrb_get_args(mrb, "o", &val);
  if (mrb_symbol_p(val)) {
    mrb_sym sym = mrb_symbol(val);
    if (sym == mrb_intern_cstr(mrb, "cpu"))         g_use_cpu = 1;
    else if (sym == mrb_intern_cstr(mrb, "vulkan")) g_use_cpu = 0;
    else mrb_raise(mrb, E_ARGUMENT_ERROR, "backend must be :vulkan or :cpu");
  } else {
    mrb_raise(mrb, E_TYPE_ERROR, "backend= expects Symbol (:vulkan or :cpu)");
  }
  return val;
}

/* ---- mruby: GPU.device_name ---- */
static mrb_value mrb_gpu_device_name(mrb_state *mrb, mrb_value self) {
  if (!g_ctx.initialized) return mrb_str_new_cstr(mrb, "(not initialized)");
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);
  return mrb_str_new_cstr(mrb, props.deviceName);
}

/* ---- mruby: GPU.transpose(buf, rows, cols) -> GPU::Buffer ---- */
/* rows x cols (row-major) を cols x rows (row-major) に転置 */
static mrb_value mrb_gpu_transpose(mrb_state *mrb, mrb_value self) {
  mrb_value va;
  mrb_int rows, cols;
  mrb_get_args(mrb, "oii", &va, &rows, &cols);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  if (a->n != (uint32_t)(rows * cols)) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "Buffer size != rows * cols");
  }

  GpuBuffer *b = create_buffer(mrb, a->n);

  float *src, *dst;
  vkMapMemory(g_ctx.device, a->memory, 0, a->bytes, 0, (void **)&src);
  vkMapMemory(g_ctx.device, b->memory, 0, b->bytes, 0, (void **)&dst);

  for (mrb_int r = 0; r < rows; r++) {
    for (mrb_int c = 0; c < cols; c++) {
      dst[c * rows + r] = src[r * cols + c];
    }
  }

  vkUnmapMemory(g_ctx.device, a->memory);
  vkUnmapMemory(g_ctx.device, b->memory);

  return wrap_buffer(mrb, b);
}

/* ---- mruby: GPU.split_rgb(rgb_str, w, h) -> [R_buf, G_buf, B_buf] ---- */
static mrb_value mrb_gpu_split_rgb(mrb_state *mrb, mrb_value self) {
  mrb_value rgb_str;
  mrb_int w, h;
  mrb_get_args(mrb, "Sii", &rgb_str, &w, &h);

  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "GPU not initialized. Call GPU.init first.");
  }

  uint32_t n = (uint32_t)(w * h);
  const uint8_t *src = (const uint8_t *)RSTRING_PTR(rgb_str);

  GpuBuffer *r = create_buffer(mrb, n);
  GpuBuffer *g = create_buffer(mrb, n);
  GpuBuffer *b = create_buffer(mrb, n);

  float *pr, *pg, *pb;
  vkMapMemory(g_ctx.device, r->memory, 0, r->bytes, 0, (void **)&pr);
  vkMapMemory(g_ctx.device, g->memory, 0, g->bytes, 0, (void **)&pg);
  vkMapMemory(g_ctx.device, b->memory, 0, b->bytes, 0, (void **)&pb);

  for (uint32_t i = 0; i < n; i++) {
    pr[i] = (float)src[i * 3];
    pg[i] = (float)src[i * 3 + 1];
    pb[i] = (float)src[i * 3 + 2];
  }

  vkUnmapMemory(g_ctx.device, r->memory);
  vkUnmapMemory(g_ctx.device, g->memory);
  vkUnmapMemory(g_ctx.device, b->memory);

  mrb_value ary = mrb_ary_new_capa(mrb, 3);
  mrb_ary_push(mrb, ary, wrap_buffer(mrb, r));
  mrb_ary_push(mrb, ary, wrap_buffer(mrb, g));
  mrb_ary_push(mrb, ary, wrap_buffer(mrb, b));
  return ary;
}

/* ---- mruby: GPU.benchmark(n) -> Hash {avg:, min:, max:, runs:} ---- */
static mrb_value mrb_gpu_benchmark(mrb_state *mrb, mrb_value self) {
  mrb_int n;
  mrb_get_args(mrb, "i", &n);

  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "GPU not initialized. Call GPU.init first.");
  }

  GpuBuffer *a = create_buffer(mrb, (uint32_t)n);
  GpuBuffer *b_buf = create_buffer(mrb, (uint32_t)n);
  float *pa, *pb;
  vkMapMemory(g_ctx.device, a->memory, 0, a->bytes, 0, (void **)&pa);
  vkMapMemory(g_ctx.device, b_buf->memory, 0, b_buf->bytes, 0, (void **)&pb);
  for (uint32_t i = 0; i < (uint32_t)n; i++) { pa[i] = 1.0f; pb[i] = 2.0f; }
  vkUnmapMemory(g_ctx.device, a->memory);
  vkUnmapMemory(g_ctx.device, b_buf->memory);

  int runs = 5;
  double total_time = 0.0, mn = 1e9, mx = 0.0;
  for (int r = 0; r < runs; r++) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    GpuBuffer *c = create_buffer(mrb, (uint32_t)n);
    if (g_use_cpu) {
      cpu_binop3(a, b_buf, c, PIPE_ADD);
    } else {
      VkBuffer bufs[3] = {a->buffer, b_buf->buffer, c->buffer};
      VkDeviceSize sizes[3] = {a->bytes, b_buf->bytes, c->bytes};
      uint32_t push = (uint32_t)n;
      dispatch_compute(PIPE_ADD, bufs, sizes, 3, &push, sizeof(uint32_t),
        ((uint32_t)n + 255) / 256, 1, 1);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    total_time += ms;
    if (ms < mn) mn = ms;
    if (ms > mx) mx = ms;
    /* c will be GC'd */
  }

  mrb_value h = mrb_hash_new(mrb);
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "avg")),
               mrb_float_value(mrb, total_time / runs));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "min")),
               mrb_float_value(mrb, mn));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "max")),
               mrb_float_value(mrb, mx));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "runs")),
               mrb_fixnum_value(runs));
  return h;
}

/* ---- mruby: GPU.info -> Hash ---- */
static mrb_value mrb_gpu_info(mrb_state *mrb, mrb_value self) {
  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "GPU not initialized. Call GPU.init first.");
  }

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_ctx.physical_device, &props);

  char api_ver[32];
  snprintf(api_ver, sizeof(api_ver), "%d.%d.%d",
    VK_VERSION_MAJOR(props.apiVersion),
    VK_VERSION_MINOR(props.apiVersion),
    VK_VERSION_PATCH(props.apiVersion));

  mrb_value h = mrb_hash_new(mrb);
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "device")),
               mrb_str_new_cstr(mrb, props.deviceName));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "api_version")),
               mrb_str_new_cstr(mrb, api_ver));
  mrb_hash_set(mrb, h, mrb_symbol_value(mrb_intern_cstr(mrb, "backend")),
               mrb_str_new_cstr(mrb, g_use_cpu ? "CPU" : "Vulkan"));
  return h;
}

/* ---- forward declarations for sub-gem inits ---- */
void mrb_camera_gem_init(mrb_state *mrb);
void mrb_camera_gem_final(mrb_state *mrb);
void mrb_face_gem_init(mrb_state *mrb);
void mrb_face_gem_final(mrb_state *mrb);
void mrb_display_gem_init(mrb_state *mrb);
void mrb_display_gem_final(mrb_state *mrb);
void mrb_skin_gem_init(mrb_state *mrb);
void mrb_skin_gem_final(mrb_state *mrb);

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
  mrb_define_module_function(mrb, gpu, "backend=", mrb_gpu_set_backend, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "device_name", mrb_gpu_device_name, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "transpose", mrb_gpu_transpose, MRB_ARGS_REQ(3));
  mrb_define_module_function(mrb, gpu, "split_rgb", mrb_gpu_split_rgb, MRB_ARGS_REQ(3));
  mrb_define_module_function(mrb, gpu, "benchmark", mrb_gpu_benchmark, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, gpu, "info",      mrb_gpu_info,      MRB_ARGS_NONE());

  struct RClass *buf_class = mrb_define_class_under(mrb, gpu, "Buffer", mrb->object_class);
  MRB_SET_INSTANCE_TT(buf_class, MRB_TT_CDATA);
  mrb_define_method(mrb, buf_class, "head", mrb_gpu_buffer_head, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, buf_class, "size", mrb_gpu_buffer_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, buf_class, "save", mrb_gpu_buffer_save, MRB_ARGS_REQ(1));

  mrb_camera_gem_init(mrb);
  mrb_face_gem_init(mrb);
  mrb_display_gem_init(mrb);
  mrb_skin_gem_init(mrb);
}

void mrb_mruby_gpu_gem_final(mrb_state *mrb) {
  mrb_skin_gem_final(mrb);
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
