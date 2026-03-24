#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>
#include <mruby/variable.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Pipeline / Layout enums ---- */
typedef enum {
  PIPE_ADD = 0, PIPE_SUB, PIPE_MUL, PIPE_SCALE, PIPE_RELU, PIPE_MATMUL,
  PIPE_COUNT
} PipeId;

typedef enum { LAYOUT_3BUF = 0, LAYOUT_2BUF, LAYOUT_COUNT } LayoutId;

static const LayoutId pipe_to_layout[PIPE_COUNT] = {
  LAYOUT_3BUF, /* ADD    */
  LAYOUT_3BUF, /* SUB    */
  LAYOUT_3BUF, /* MUL    */
  LAYOUT_2BUF, /* SCALE  */
  LAYOUT_2BUF, /* RELU   */
  LAYOUT_3BUF, /* MATMUL */
};

static const char *pipe_names[PIPE_COUNT] = {
  "add", "sub", "mul", "scale", "relu", "matmul"
};

/* ---- GPU Context (singleton) ---- */
typedef struct {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorSetLayout desc_layouts[LAYOUT_COUNT];
  VkPipelineLayout pipe_layouts[LAYOUT_COUNT];
  VkPipeline pipelines[PIPE_COUNT];
  VkDescriptorPool desc_pool;
  int initialized;
} GpuCtx;

static GpuCtx g_ctx = {0};

/* ---- GPU Buffer ---- */
typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  uint32_t n;
  VkDeviceSize bytes;
} GpuBuffer;

static void gpu_buffer_free(mrb_state *mrb, void *p) {
  GpuBuffer *buf = (GpuBuffer *)p;
  if (buf) {
    if (g_ctx.initialized) {
      vkDestroyBuffer(g_ctx.device, buf->buffer, NULL);
      vkFreeMemory(g_ctx.device, buf->memory, NULL);
    }
    mrb_free(mrb, buf);
  }
}

static const struct mrb_data_type gpu_buffer_type = {"GpuBuffer", gpu_buffer_free};

/* ---- Helpers ---- */
static uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(g_ctx.physical_device, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  return 0;
}

static uint8_t *load_spv(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *size = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc(*size);
  fread(buf, 1, *size, f);
  fclose(f);
  return buf;
}

static mrb_value wrap_buffer(mrb_state *mrb, GpuBuffer *buf) {
  struct RClass *buf_class = mrb_class_get_under(mrb, mrb_module_get(mrb, "GPU"), "Buffer");
  struct RData *data = mrb_data_object_alloc(mrb, buf_class, buf, &gpu_buffer_type);
  return mrb_obj_value(data);
}

/* ---- Create Buffer (host-visible) ---- */
static GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n) {
  GpuBuffer *buf = mrb_malloc(mrb, sizeof(GpuBuffer));
  buf->n = n;
  buf->bytes = sizeof(float) * n;

  VkBufferCreateInfo bi = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = buf->bytes,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
  };
  vkCreateBuffer(g_ctx.device, &bi, NULL, &buf->buffer);

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_ctx.device, buf->buffer, &req);

  VkMemoryAllocateInfo ai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size,
    .memoryTypeIndex = find_memory_type(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
  };
  vkAllocateMemory(g_ctx.device, &ai, NULL, &buf->memory);
  vkBindBufferMemory(g_ctx.device, buf->buffer, buf->memory, 0);

  return buf;
}

/* ---- Generic Compute Dispatch ---- */
static void dispatch_compute(
    PipeId pipe_id,
    VkBuffer *buffers, VkDeviceSize *sizes, int num_buffers,
    const void *push_data, uint32_t push_size,
    uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
  LayoutId lid = pipe_to_layout[pipe_id];

  /* Allocate descriptor set */
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_ctx.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_ctx.desc_layouts[lid]
  };
  VkDescriptorSet desc_set;
  vkAllocateDescriptorSets(g_ctx.device, &dsai, &desc_set);

  /* Update descriptor set */
  VkDescriptorBufferInfo buf_infos[3];
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < num_buffers; i++) {
    buf_infos[i] = (VkDescriptorBufferInfo){buffers[i], 0, sizes[i]};
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = i,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buf_infos[i]
    };
  }
  vkUpdateDescriptorSets(g_ctx.device, num_buffers, writes, 0, NULL);

  /* Command buffer */
  VkCommandBufferAllocateInfo cbai = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_ctx.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(g_ctx.device, &cbai, &cmd);

  VkCommandBufferBeginInfo begin = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };
  vkBeginCommandBuffer(cmd, &begin);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.pipelines[pipe_id]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    g_ctx.pipe_layouts[lid], 0, 1, &desc_set, 0, NULL);
  if (push_size > 0) {
    vkCmdPushConstants(cmd, g_ctx.pipe_layouts[lid], VK_SHADER_STAGE_COMPUTE_BIT,
      0, push_size, push_data);
  }
  vkCmdDispatch(cmd, group_x, group_y, group_z);
  vkEndCommandBuffer(cmd);

  /* Submit and wait */
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(g_ctx.device, &fi, NULL, &fence);

  VkSubmitInfo si = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd
  };
  vkQueueSubmit(g_ctx.queue, 1, &si, fence);
  vkWaitForFences(g_ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(g_ctx.device, fence, NULL);
  vkFreeCommandBuffers(g_ctx.device, g_ctx.cmd_pool, 1, &cmd);
}

/* ---- Init ---- */
static void gpu_init(const char *shader_dir) {
  if (g_ctx.initialized) return;

  /* Instance */
  VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "mruby-gpu",
    .apiVersion = VK_API_VERSION_1_1
  };
  VkInstanceCreateInfo inst_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info
  };
  vkCreateInstance(&inst_info, NULL, &g_ctx.instance);

  /* Physical Device (pick first) */
  uint32_t dev_count = 1;
  vkEnumeratePhysicalDevices(g_ctx.instance, &dev_count, &g_ctx.physical_device);

  /* Queue Family (compute) */
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, NULL);
  VkQueueFamilyProperties *qf_props = malloc(sizeof(VkQueueFamilyProperties) * qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &qf_count, qf_props);
  g_ctx.queue_family = 0;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      g_ctx.queue_family = i;
      break;
    }
  }
  free(qf_props);

  /* Device + Queue */
  float priority = 1.0f;
  VkDeviceQueueCreateInfo q_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_ctx.queue_family,
    .queueCount = 1,
    .pQueuePriorities = &priority
  };
  VkDeviceCreateInfo dev_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &q_info
  };
  vkCreateDevice(g_ctx.physical_device, &dev_info, NULL, &g_ctx.device);
  vkGetDeviceQueue(g_ctx.device, g_ctx.queue_family, 0, &g_ctx.queue);

  /* Command Pool */
  VkCommandPoolCreateInfo pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = g_ctx.queue_family
  };
  vkCreateCommandPool(g_ctx.device, &pool_info, NULL, &g_ctx.cmd_pool);

  /* Descriptor Set Layouts */
  int buf_counts[LAYOUT_COUNT] = {3, 2};
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < buf_counts[l]; i++) {
      bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      };
    }
    VkDescriptorSetLayoutCreateInfo dl_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = buf_counts[l],
      .pBindings = bindings
    };
    vkCreateDescriptorSetLayout(g_ctx.device, &dl_info, NULL, &g_ctx.desc_layouts[l]);
  }

  /* Pipeline Layouts (one per descriptor layout, shared push constant range) */
  VkPushConstantRange push_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = 16  /* 4 x uint32_t = enough for matmul {M,K,N,flags} and scale {n,scalar} */
  };
  for (int l = 0; l < LAYOUT_COUNT; l++) {
    VkPipelineLayoutCreateInfo pl_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &g_ctx.desc_layouts[l],
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_range
    };
    vkCreatePipelineLayout(g_ctx.device, &pl_info, NULL, &g_ctx.pipe_layouts[l]);
  }

  /* Load shaders and create pipelines */
  for (int p = 0; p < PIPE_COUNT; p++) {
    char spv_path[512];
    snprintf(spv_path, sizeof(spv_path), "%s/%s.spv", shader_dir, pipe_names[p]);

    size_t spv_size;
    uint8_t *spv_code = load_spv(spv_path, &spv_size);
    if (!spv_code) {
      fprintf(stderr, "Warning: could not load %s\n", spv_path);
      g_ctx.pipelines[p] = VK_NULL_HANDLE;
      continue;
    }

    VkShaderModuleCreateInfo sm_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = (uint32_t *)spv_code
    };
    VkShaderModule shader;
    vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader);
    free(spv_code);

    LayoutId lid = pipe_to_layout[p];
    VkComputePipelineCreateInfo cp_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main"
      },
      .layout = g_ctx.pipe_layouts[lid]
    };
    vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info, NULL, &g_ctx.pipelines[p]);
    vkDestroyShaderModule(g_ctx.device, shader, NULL);
  }

  /* Descriptor Pool */
  VkDescriptorPoolSize pool_size = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 768
  };
  VkDescriptorPoolCreateInfo dp_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 256,
    .poolSizeCount = 1,
    .pPoolSizes = &pool_size
  };
  vkCreateDescriptorPool(g_ctx.device, &dp_info, NULL, &g_ctx.desc_pool);

  g_ctx.initialized = 1;
}

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

/* ---- mruby: GPU::Buffer#head(n) ---- */
static mrb_value mrb_gpu_buffer_head(mrb_state *mrb, mrb_value self) {
  mrb_int count;
  mrb_get_args(mrb, "i", &count);

  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  if ((uint32_t)count > buf->n) count = buf->n;

  float *mapped;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);

  mrb_value ary = mrb_ary_new_capa(mrb, count);
  for (mrb_int i = 0; i < count; i++) {
    mrb_ary_push(mrb, ary, mrb_float_value(mrb, (mrb_float)mapped[i]));
  }
  vkUnmapMemory(g_ctx.device, buf->memory);

  return ary;
}

/* ---- mruby: GPU::Buffer#size ---- */
static mrb_value mrb_gpu_buffer_size(mrb_state *mrb, mrb_value self) {
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  return mrb_fixnum_value(buf->n);
}

/* ---- mruby: GPU::Buffer#save(path) ---- */
static mrb_value mrb_gpu_buffer_save(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_get_args(mrb, "z", &path);

  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);

  float *mapped;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);
  FILE *f = fopen(path, "wb");
  if (!f) {
    vkUnmapMemory(g_ctx.device, buf->memory);
    mrb_raisef(mrb, E_RUNTIME_ERROR, "Cannot open file for writing: %s", path);
  }
  fwrite(mapped, sizeof(float), buf->n, f);
  fclose(f);
  vkUnmapMemory(g_ctx.device, buf->memory);

  return mrb_nil_value();
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
}

void mrb_mruby_gpu_gem_final(mrb_state *mrb) {
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
