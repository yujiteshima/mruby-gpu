/* Quick sanity test for v4l2_capture — captures 1 frame and prints info */
#include <stdio.h>
#include <stdint.h>
#include "../src/v4l2_capture.h"

int main(void) {
  v4l2_camera_t cam;
  printf("Opening /dev/video0 at 640x480...\n");
  if (v4l2_camera_open(&cam, "/dev/video0", 640, 480) < 0) {
    fprintf(stderr, "Failed to open camera\n");
    return 1;
  }
  printf("Opened: %dx%d, %d buffers\n", cam.width, cam.height, cam.n_buffers);

  uint8_t *data = NULL;
  size_t len = 0;
  printf("Capturing frame...\n");
  if (v4l2_camera_capture(&cam, &data, &len) < 0) {
    fprintf(stderr, "Capture failed\n");
    v4l2_camera_close(&cam);
    return 1;
  }
  printf("Got frame: %zu bytes (expected %d)\n", len, cam.width * cam.height * 2);
  /* Print first 8 bytes as YUYV values */
  printf("First macropixel: Y0=%d U=%d Y1=%d V=%d\n",
    data[0], data[1], data[2], data[3]);

  v4l2_camera_release(&cam);
  v4l2_camera_close(&cam);
  printf("Done.\n");
  return 0;
}
