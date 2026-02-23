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

/* ---- GPU Context (singleton) ---- */
typedef struct {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
  VkPipeline pipeline;
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

/* ---- Init ---- */
static void gpu_init(const char *spv_path) {
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

  /* Descriptor Set Layout (3 storage buffers) */
  VkDescriptorSetLayoutBinding bindings[3];
  for (int i = 0; i < 3; i++) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
      .binding = i,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
  }
  VkDescriptorSetLayoutCreateInfo dl_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 3,
    .pBindings = bindings
  };
  vkCreateDescriptorSetLayout(g_ctx.device, &dl_info, NULL, &g_ctx.desc_layout);

  /* Push Constant Range */
  VkPushConstantRange push_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = sizeof(uint32_t)
  };

  /* Pipeline Layout */
  VkPipelineLayoutCreateInfo pl_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &g_ctx.desc_layout,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_range
  };
  vkCreatePipelineLayout(g_ctx.device, &pl_info, NULL, &g_ctx.pipe_layout);

  /* Shader Module */
  size_t spv_size;
  uint8_t *spv_code = load_spv(spv_path, &spv_size);
  VkShaderModuleCreateInfo sm_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spv_size,
    .pCode = (uint32_t *)spv_code
  };
  VkShaderModule shader;
  vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader);
  free(spv_code);

  /* Compute Pipeline */
  VkComputePipelineCreateInfo cp_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main"
    },
    .layout = g_ctx.pipe_layout
  };
  vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info, NULL, &g_ctx.pipeline);
  vkDestroyShaderModule(g_ctx.device, shader, NULL);

  /* Descriptor Pool */
  VkDescriptorPoolSize pool_size = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 30
  };
  VkDescriptorPoolCreateInfo dp_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 10,
    .poolSizeCount = 1,
    .pPoolSizes = &pool_size
  };
  vkCreateDescriptorPool(g_ctx.device, &dp_info, NULL, &g_ctx.desc_pool);

  g_ctx.initialized = 1;
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

/* ---- mruby: GPU.init(spv_path) ---- */
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

  /* Copy data to GPU buffer */
  float *mapped;
  vkMapMemory(g_ctx.device, buf->memory, 0, buf->bytes, 0, (void **)&mapped);
  for (uint32_t i = 0; i < n; i++) {
    mapped[i] = (float)mrb_float(mrb_ary_ref(mrb, ary, i));
  }
  vkUnmapMemory(g_ctx.device, buf->memory);

  struct RClass *buf_class = mrb_class_get_under(mrb, mrb_module_get(mrb, "GPU"), "Buffer");
  struct RData *data = mrb_data_object_alloc(mrb, buf_class, buf, &gpu_buffer_type);
  return mrb_obj_value(data);
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

  struct RClass *buf_class = mrb_class_get_under(mrb, mrb_module_get(mrb, "GPU"), "Buffer");
  struct RData *data = mrb_data_object_alloc(mrb, buf_class, buf, &gpu_buffer_type);
  return mrb_obj_value(data);
}

/* ---- mruby: GPU.add(a, b) -> GPU::Buffer ---- */
static mrb_value mrb_gpu_add(mrb_state *mrb, mrb_value self) {
  mrb_value va, vb;
  mrb_get_args(mrb, "oo", &va, &vb);

  GpuBuffer *a = DATA_GET_PTR(mrb, va, &gpu_buffer_type, GpuBuffer);
  GpuBuffer *b = DATA_GET_PTR(mrb, vb, &gpu_buffer_type, GpuBuffer);

  if (a->n != b->n) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "Buffer size mismatch");
  }

  GpuBuffer *c = create_buffer(mrb, a->n);

  /* Allocate descriptor set */
  VkDescriptorSetAllocateInfo dsai = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_ctx.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &g_ctx.desc_layout
  };
  VkDescriptorSet desc_set;
  vkAllocateDescriptorSets(g_ctx.device, &dsai, &desc_set);

  /* Update descriptor set */
  VkDescriptorBufferInfo buf_infos[3] = {
    {a->buffer, 0, a->bytes},
    {b->buffer, 0, b->bytes},
    {c->buffer, 0, c->bytes}
  };
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = i,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buf_infos[i]
    };
  }
  vkUpdateDescriptorSets(g_ctx.device, 3, writes, 0, NULL);

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
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    g_ctx.pipe_layout, 0, 1, &desc_set, 0, NULL);
  vkCmdPushConstants(cmd, g_ctx.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
    0, sizeof(uint32_t), &a->n);
  vkCmdDispatch(cmd, (a->n + 255) / 256, 1, 1);
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

  struct RClass *buf_class = mrb_class_get_under(mrb, mrb_module_get(mrb, "GPU"), "Buffer");
  struct RData *data = mrb_data_object_alloc(mrb, buf_class, c, &gpu_buffer_type);
  return mrb_obj_value(data);
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
  mrb_define_module_function(mrb, gpu, "add", mrb_gpu_add, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, gpu, "backend", mrb_gpu_backend, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "device_name", mrb_gpu_device_name, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, gpu, "fill", mrb_gpu_fill, MRB_ARGS_REQ(2));

  struct RClass *buf_class = mrb_define_class_under(mrb, gpu, "Buffer", mrb->object_class);
  MRB_SET_INSTANCE_TT(buf_class, MRB_TT_CDATA);
  mrb_define_method(mrb, buf_class, "head", mrb_gpu_buffer_head, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, buf_class, "size", mrb_gpu_buffer_size, MRB_ARGS_NONE());
}

void mrb_mruby_gpu_gem_final(mrb_state *mrb) {
  if (g_ctx.initialized) {
    vkDestroyDescriptorPool(g_ctx.device, g_ctx.desc_pool, NULL);
    vkDestroyPipeline(g_ctx.device, g_ctx.pipeline, NULL);
    vkDestroyPipelineLayout(g_ctx.device, g_ctx.pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(g_ctx.device, g_ctx.desc_layout, NULL);
    vkDestroyCommandPool(g_ctx.device, g_ctx.cmd_pool, NULL);
    vkDestroyDevice(g_ctx.device, NULL);
    vkDestroyInstance(g_ctx.instance, NULL);
    g_ctx.initialized = 0;
  }
}
