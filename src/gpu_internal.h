/* gpu_internal.h -- shared declarations for gpu_vulkan.c, gpu_buffer.c, gpu_ops.c */
#ifndef GPU_INTERNAL_H
#define GPU_INTERNAL_H

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

extern GpuCtx g_ctx;

/* ---- GPU Buffer ---- */
typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  uint32_t n;
  VkDeviceSize bytes;
} GpuBuffer;

extern const struct mrb_data_type gpu_buffer_type;

/* ---- gpu_vulkan.c ---- */
void gpu_init(const char *shader_dir);
void dispatch_compute(PipeId pipe_id,
                      VkBuffer *buffers, VkDeviceSize *sizes, int num_buffers,
                      const void *push_data, uint32_t push_size,
                      uint32_t group_x, uint32_t group_y, uint32_t group_z);

/* ---- gpu_buffer.c ---- */
GpuBuffer *create_buffer(mrb_state *mrb, uint32_t n);
mrb_value wrap_buffer(mrb_state *mrb, GpuBuffer *buf);

#endif
