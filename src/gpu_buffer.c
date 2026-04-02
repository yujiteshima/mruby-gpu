/* gpu_buffer.c -- GpuBuffer creation, access, and I/O */
#include "gpu_internal.h"

/* ---- Buffer free ---- */
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

const struct mrb_data_type gpu_buffer_type = {"GpuBuffer", gpu_buffer_free};

/* ---- Create Buffer (host-visible) ---- */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n) {
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

  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(g_ctx.physical_device, &mem_props);
  uint32_t mem_idx = 0;
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((req.memoryTypeBits & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags &
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      mem_idx = i;
      break;
    }
  }

  VkMemoryAllocateInfo ai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = req.size,
    .memoryTypeIndex = mem_idx
  };
  vkAllocateMemory(g_ctx.device, &ai, NULL, &buf->memory);
  vkBindBufferMemory(g_ctx.device, buf->buffer, buf->memory, 0);

  return buf;
}

mrb_value wrap_buffer(mrb_state *mrb, GpuBuffer *buf) {
  struct RClass *buf_class = mrb_class_get_under(mrb, mrb_module_get(mrb, "GPU"), "Buffer");
  struct RData *data = mrb_data_object_alloc(mrb, buf_class, buf, &gpu_buffer_type);
  return mrb_obj_value(data);
}

/* ---- mruby: GPU::Buffer#head(n) ---- */
mrb_value mrb_gpu_buffer_head(mrb_state *mrb, mrb_value self) {
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
mrb_value mrb_gpu_buffer_size(mrb_state *mrb, mrb_value self) {
  GpuBuffer *buf = DATA_GET_PTR(mrb, self, &gpu_buffer_type, GpuBuffer);
  return mrb_fixnum_value(buf->n);
}

/* ---- mruby: GPU::Buffer#save(path) ---- */
mrb_value mrb_gpu_buffer_save(mrb_state *mrb, mrb_value self) {
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
