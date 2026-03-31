#ifndef V4L2_CAPTURE_H
#define V4L2_CAPTURE_H

#include <stdint.h>
#include <stddef.h>

#define V4L2_MAX_BUFFERS 4

typedef struct {
  int fd;
  uint8_t *buffers[V4L2_MAX_BUFFERS];
  size_t buffer_lengths[V4L2_MAX_BUFFERS];
  int n_buffers;
  int width;
  int height;
} v4l2_camera_t;

/* Opens device, sets format to YUYV at (w x h), allocates mmap buffers, starts streaming.
   Returns 0 on success, -1 on error (errno set). */
int v4l2_camera_open(v4l2_camera_t *cam, const char *dev, int w, int h);

/* Dequeues one frame. Sets *data to the mmap pointer, *len to byte count.
   Caller must call v4l2_camera_release() after consuming the frame.
   Returns 0 on success, -1 on error. */
int v4l2_camera_capture(v4l2_camera_t *cam, uint8_t **data, size_t *len);

/* Re-enqueues the last captured buffer so the driver can reuse it. */
int v4l2_camera_release(v4l2_camera_t *cam);

/* Stops streaming, unmaps buffers, closes fd. */
void v4l2_camera_close(v4l2_camera_t *cam);

#endif /* V4L2_CAPTURE_H */
