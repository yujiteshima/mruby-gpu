/*
 * mrb_camera.c — mruby Camera class backed by V4L2
 *
 * Camera.open(dev, width, height) -> Camera
 * cam.capture                     -> String (raw YUYV bytes, width*height*2 bytes)
 * cam.width  -> Integer
 * cam.height -> Integer
 * cam.close  -> nil
 */

#include <mruby.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>
#include <mruby/variable.h>
#include <string.h>
#include <stdlib.h>

#include "v4l2_capture.h"

/* ---- mruby data type ---- */

static void camera_free(mrb_state *mrb, void *p) {
  if (p) {
    v4l2_camera_t *cam = (v4l2_camera_t *)p;
    v4l2_camera_close(cam);
    mrb_free(mrb, cam);
  }
}

static const struct mrb_data_type camera_type = {"Camera", camera_free};

/* ---- Camera.open(dev, width, height) ---- */
static mrb_value mrb_camera_open(mrb_state *mrb, mrb_value klass) {
  const char *dev;
  mrb_int w, h;
  mrb_get_args(mrb, "zii", &dev, &w, &h);

  v4l2_camera_t *cam = (v4l2_camera_t *)mrb_malloc(mrb, sizeof(v4l2_camera_t));
  if (v4l2_camera_open(cam, dev, (int)w, (int)h) < 0) {
    mrb_free(mrb, cam);
    mrb_raisef(mrb, E_RUNTIME_ERROR, "Camera.open failed: %s", dev);
  }

  struct RClass *cls = mrb_class_ptr(klass);
  struct RData *data = mrb_data_object_alloc(mrb, cls, cam, &camera_type);
  return mrb_obj_value(data);
}

/* ---- cam.capture -> String (YUYV raw bytes) ---- */
static mrb_value mrb_camera_capture(mrb_state *mrb, mrb_value self) {
  v4l2_camera_t *cam = DATA_GET_PTR(mrb, self, &camera_type, v4l2_camera_t);

  uint8_t *data = NULL;
  size_t len = 0;
  if (v4l2_camera_capture(cam, &data, &len) < 0) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "Camera capture failed");
  }

  /* Copy frame into a mruby String so the driver buffer can be released */
  mrb_value frame = mrb_str_new(mrb, (const char *)data, (mrb_int)len);
  v4l2_camera_release(cam);

  return frame;
}

/* shared YUYV->RGB conversion helper */
static void yuyv_to_rgb_buf(const uint8_t *yuyv, uint8_t *rgb, int w, int h) {
#define CLAMP(x) ((uint8_t)((x) < 0 ? 0 : (x) > 255 ? 255 : (x)))
  for (int i = 0; i < w * h / 2; i++) {
    int y0 = yuyv[i * 4 + 0];
    int u  = yuyv[i * 4 + 1] - 128;
    int y1 = yuyv[i * 4 + 2];
    int v  = yuyv[i * 4 + 3] - 128;
    rgb[(i*2)  *3+0] = CLAMP(y0 + (int)(1.402f*v));
    rgb[(i*2)  *3+1] = CLAMP(y0 - (int)(0.344f*u) - (int)(0.714f*v));
    rgb[(i*2)  *3+2] = CLAMP(y0 + (int)(1.772f*u));
    rgb[(i*2+1)*3+0] = CLAMP(y1 + (int)(1.402f*v));
    rgb[(i*2+1)*3+1] = CLAMP(y1 - (int)(0.344f*u) - (int)(0.714f*v));
    rgb[(i*2+1)*3+2] = CLAMP(y1 + (int)(1.772f*u));
  }
#undef CLAMP
}

/* cam.capture_rgb -> String (RGB888, width*height*3 bytes) */
static mrb_value mrb_camera_capture_rgb(mrb_state *mrb, mrb_value self) {
  v4l2_camera_t *cam = DATA_GET_PTR(mrb, self, &camera_type, v4l2_camera_t);
  uint8_t *yuyv = NULL;
  size_t len = 0;
  if (v4l2_camera_capture(cam, &yuyv, &len) < 0)
    mrb_raise(mrb, E_RUNTIME_ERROR, "Camera capture failed");

  int w = cam->width, h = cam->height;
  mrb_int rgb_len = (mrb_int)(w * h * 3);
  mrb_value out = mrb_str_buf_new(mrb, rgb_len);
  yuyv_to_rgb_buf(yuyv, (uint8_t *)RSTRING_PTR(out), w, h);
  RSTR_SET_LEN(mrb_str_ptr(out), rgb_len);
  v4l2_camera_release(cam);
  return out;
}

/* Camera.yuyv_to_rgb(yuyv_str, width, height) -> String (RGB888)
 * YUYV 文字列を渡して RGB に変換する (キャプチャなし) */
static mrb_value mrb_camera_yuyv_to_rgb(mrb_state *mrb, mrb_value klass) {
  mrb_value yuyv_str;
  mrb_int w, h;
  mrb_get_args(mrb, "Sii", &yuyv_str, &w, &h);

  mrb_int rgb_len = w * h * 3;
  mrb_value out = mrb_str_buf_new(mrb, rgb_len);
  yuyv_to_rgb_buf((const uint8_t *)RSTRING_PTR(yuyv_str),
                  (uint8_t *)RSTRING_PTR(out), (int)w, (int)h);
  RSTR_SET_LEN(mrb_str_ptr(out), rgb_len);
  return out;
}

/* Camera.crop_rgb(rgb_str, src_w, src_h, x, y, crop_w, crop_h) -> String
 * RGB888 画像から矩形領域を切り出して返す（タイル処理用）*/
static mrb_value mrb_camera_crop_rgb(mrb_state *mrb, mrb_value klass) {
  mrb_value rgb_str;
  mrb_int sw, sh, cx, cy, cw, ch;
  mrb_get_args(mrb, "Siiiiii", &rgb_str, &sw, &sh, &cx, &cy, &cw, &ch);

  /* clamp to source bounds */
  if (cx < 0) cx = 0;
  if (cy < 0) cy = 0;
  if (cx + cw > sw) cw = sw - cx;
  if (cy + ch > sh) ch = sh - cy;

  mrb_int out_len = cw * ch * 3;
  mrb_value out = mrb_str_buf_new(mrb, out_len);
  const uint8_t *src = (const uint8_t *)RSTRING_PTR(rgb_str);
  uint8_t *dst = (uint8_t *)RSTRING_PTR(out);

  for (mrb_int row = 0; row < ch; row++) {
    const uint8_t *s = src + ((cy + row) * sw + cx) * 3;
    uint8_t       *d = dst + row * cw * 3;
    memcpy(d, s, (size_t)(cw * 3));
  }
  RSTR_SET_LEN(mrb_str_ptr(out), out_len);
  return out;
}

/* ---- cam.width / cam.height ---- */
static mrb_value mrb_camera_width(mrb_state *mrb, mrb_value self) {
  v4l2_camera_t *cam = DATA_GET_PTR(mrb, self, &camera_type, v4l2_camera_t);
  return mrb_fixnum_value(cam->width);
}

static mrb_value mrb_camera_height(mrb_state *mrb, mrb_value self) {
  v4l2_camera_t *cam = DATA_GET_PTR(mrb, self, &camera_type, v4l2_camera_t);
  return mrb_fixnum_value(cam->height);
}

/* ---- cam.close ---- */
static mrb_value mrb_camera_close(mrb_state *mrb, mrb_value self) {
  v4l2_camera_t *cam = DATA_GET_PTR(mrb, self, &camera_type, v4l2_camera_t);
  v4l2_camera_close(cam);
  return mrb_nil_value();
}

/* ---- gem registration ---- */
void mrb_camera_gem_init(mrb_state *mrb) {
  struct RClass *cls = mrb_define_class(mrb, "Camera", mrb->object_class);
  MRB_SET_INSTANCE_TT(cls, MRB_TT_CDATA);

  mrb_define_class_method(mrb, cls, "open",         mrb_camera_open,         MRB_ARGS_REQ(3));
  mrb_define_class_method(mrb, cls, "yuyv_to_rgb",  mrb_camera_yuyv_to_rgb,  MRB_ARGS_REQ(3));
  mrb_define_class_method(mrb, cls, "crop_rgb",     mrb_camera_crop_rgb,     MRB_ARGS_REQ(7));
  mrb_define_method(mrb, cls, "capture",      mrb_camera_capture,      MRB_ARGS_NONE());
  mrb_define_method(mrb, cls, "capture_rgb",  mrb_camera_capture_rgb,  MRB_ARGS_NONE());
  mrb_define_method(mrb, cls, "width",        mrb_camera_width,        MRB_ARGS_NONE());
  mrb_define_method(mrb, cls, "height",       mrb_camera_height,       MRB_ARGS_NONE());
  mrb_define_method(mrb, cls, "close",        mrb_camera_close,        MRB_ARGS_NONE());
}

void mrb_camera_gem_final(mrb_state *mrb) {
  (void)mrb;
}
