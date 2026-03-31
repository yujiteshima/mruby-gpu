#include "v4l2_capture.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <unistd.h>

static int xioctl(int fd, unsigned long request, void *arg) {
  int r;
  do {
    r = ioctl(fd, request, arg);
  } while (r == -1 && errno == EINTR);
  return r;
}

int v4l2_camera_open(v4l2_camera_t *cam, const char *dev, int w, int h) {
  memset(cam, 0, sizeof(*cam));
  cam->fd = -1;

  cam->fd = open(dev, O_RDWR | O_NONBLOCK);
  if (cam->fd < 0) {
    perror("v4l2: open");
    return -1;
  }

  /* Set format: YUYV */
  struct v4l2_format fmt = {0};
  fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width       = (uint32_t)w;
  fmt.fmt.pix.height      = (uint32_t)h;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
  if (xioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) {
    perror("v4l2: VIDIOC_S_FMT");
    close(cam->fd);
    cam->fd = -1;
    return -1;
  }
  cam->width  = (int)fmt.fmt.pix.width;
  cam->height = (int)fmt.fmt.pix.height;

  /* Request mmap buffers */
  struct v4l2_requestbuffers req = {0};
  req.count  = V4L2_MAX_BUFFERS;
  req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(cam->fd, VIDIOC_REQBUFS, &req) < 0) {
    perror("v4l2: VIDIOC_REQBUFS");
    close(cam->fd);
    cam->fd = -1;
    return -1;
  }
  cam->n_buffers = (int)req.count;

  /* mmap each buffer */
  for (int i = 0; i < cam->n_buffers; i++) {
    struct v4l2_buffer buf = {0};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = (uint32_t)i;
    if (xioctl(cam->fd, VIDIOC_QUERYBUF, &buf) < 0) {
      perror("v4l2: VIDIOC_QUERYBUF");
      v4l2_camera_close(cam);
      return -1;
    }
    cam->buffer_lengths[i] = buf.length;
    cam->buffers[i] = (uint8_t *)mmap(NULL, buf.length,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED,
                           cam->fd, buf.m.offset);
    if (cam->buffers[i] == MAP_FAILED) {
      perror("v4l2: mmap");
      cam->buffers[i] = NULL;
      v4l2_camera_close(cam);
      return -1;
    }
  }

  /* Enqueue all buffers */
  for (int i = 0; i < cam->n_buffers; i++) {
    struct v4l2_buffer buf = {0};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = (uint32_t)i;
    if (xioctl(cam->fd, VIDIOC_QBUF, &buf) < 0) {
      perror("v4l2: VIDIOC_QBUF (init)");
      v4l2_camera_close(cam);
      return -1;
    }
  }

  /* Start streaming */
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(cam->fd, VIDIOC_STREAMON, &type) < 0) {
    perror("v4l2: VIDIOC_STREAMON");
    v4l2_camera_close(cam);
    return -1;
  }

  return 0;
}

/* We track which buffer index was last dequeued so we can re-enqueue it. */
static int g_last_buf_index = -1;

int v4l2_camera_capture(v4l2_camera_t *cam, uint8_t **data, size_t *len) {
  /* Wait for frame with select (timeout 2s) */
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(cam->fd, &fds);
  struct timeval tv = {2, 0};
  int r = select(cam->fd + 1, &fds, NULL, NULL, &tv);
  if (r <= 0) {
    fprintf(stderr, "v4l2: select timeout or error\n");
    return -1;
  }

  struct v4l2_buffer buf = {0};
  buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if (xioctl(cam->fd, VIDIOC_DQBUF, &buf) < 0) {
    perror("v4l2: VIDIOC_DQBUF");
    return -1;
  }

  g_last_buf_index = (int)buf.index;
  *data = cam->buffers[buf.index];
  *len  = buf.bytesused;
  return 0;
}

int v4l2_camera_release(v4l2_camera_t *cam) {
  if (g_last_buf_index < 0) return 0;
  struct v4l2_buffer buf = {0};
  buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index  = (uint32_t)g_last_buf_index;
  g_last_buf_index = -1;
  if (xioctl(cam->fd, VIDIOC_QBUF, &buf) < 0) {
    perror("v4l2: VIDIOC_QBUF (release)");
    return -1;
  }
  return 0;
}

void v4l2_camera_close(v4l2_camera_t *cam) {
  if (cam->fd >= 0) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(cam->fd, VIDIOC_STREAMOFF, &type);
  }
  for (int i = 0; i < cam->n_buffers; i++) {
    if (cam->buffers[i] && cam->buffers[i] != MAP_FAILED) {
      munmap(cam->buffers[i], cam->buffer_lengths[i]);
      cam->buffers[i] = NULL;
    }
  }
  if (cam->fd >= 0) {
    close(cam->fd);
    cam->fd = -1;
  }
}
