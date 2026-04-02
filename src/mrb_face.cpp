/*
 * mrb_face.cpp — mruby FaceDetector class backed by NCNN + Vulkan
 *
 * FaceDetector.new(model_path, use_gpu: true) -> FaceDetector
 * detector.detect_rgb(rgb_str, width, height, threshold: 0.7) -> Array of Hash
 *   Each Hash has keys: :x, :y, :w, :h, :score (all Float)
 *
 * Input is RGB888. Use Camera.yuyv_to_rgb for YUYV→RGB conversion.
 */

extern "C" {
#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/hash.h>
#include <mruby/string.h>
#include <mruby/variable.h>
}

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#include <ncnn/net.h>

/* ---- Face box ---- */
struct FaceBox {
  float x, y, w, h, score;
};

/* ---- Detector state ---- */
struct FaceDetector {
  ncnn::Net net;
  bool use_gpu;
  int input_w;   /* model input width  (320 for UltraFace-slim) */
  int input_h;   /* model input height (240 for UltraFace-slim) */
};

/* ---- Simple NMS ---- */
static float iou(const FaceBox &a, const FaceBox &b) {
  float ax2 = a.x + a.w, ay2 = a.y + a.h;
  float bx2 = b.x + b.w, by2 = b.y + b.h;
  float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
  float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
  float ua = a.w * a.h + b.w * b.h - inter;
  return ua > 0 ? inter / ua : 0.0f;
}

static std::vector<FaceBox> nms(std::vector<FaceBox> boxes, float iou_thresh) {
  std::sort(boxes.begin(), boxes.end(),
            [](const FaceBox &a, const FaceBox &b){ return a.score > b.score; });
  std::vector<bool> suppressed(boxes.size(), false);
  std::vector<FaceBox> result;
  for (size_t i = 0; i < boxes.size(); i++) {
    if (suppressed[i]) continue;
    result.push_back(boxes[i]);
    for (size_t j = i + 1; j < boxes.size(); j++) {
      if (!suppressed[j] && iou(boxes[i], boxes[j]) > iou_thresh)
        suppressed[j] = true;
    }
  }
  return result;
}

/* ---- Generate UltraFace-slim prior anchors for 320x240 input ----
 * Config: strides=[8,16,32,64], min_sizes=[[10,16,24],[32,48],[64,96],[128,192,256]]
 * Total anchors: 3600+600+160+60 = 4420
 */
struct Anchor { float cx, cy, w, h; };

static std::vector<Anchor> generate_priors(int iw, int ih) {
  static const int strides[]        = {8, 16, 32, 64};
  static const int min_sizes[4][3]  = {{10,16,24},{32,48,0},{64,96,0},{128,192,256}};
  static const int n_sizes[]        = {3, 2, 2, 3};

  std::vector<Anchor> priors;
  priors.reserve(4420);
  for (int s = 0; s < 4; s++) {
    int stride = strides[s];
    int fw = (iw + stride - 1) / stride;
    int fh = (ih + stride - 1) / stride;
    for (int y = 0; y < fh; y++) {
      for (int x = 0; x < fw; x++) {
        float cx = (x + 0.5f) / fw;
        float cy = (y + 0.5f) / fh;
        for (int k = 0; k < n_sizes[s]; k++) {
          float aw = (float)min_sizes[s][k] / iw;
          float ah = (float)min_sizes[s][k] / ih;
          priors.push_back({cx, cy, aw, ah});
        }
      }
    }
  }
  return priors;
}

/* ---- Detect faces (UltraFace-slim, SSD anchor decode) ---- */
static std::vector<FaceBox> detect(FaceDetector *det,
                                    const uint8_t *rgb,
                                    int src_w, int src_h,
                                    float threshold) {
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      rgb, ncnn::Mat::PIXEL_RGB,
      src_w, src_h, det->input_w, det->input_h);

  const float mean[3] = {127.0f, 127.0f, 127.0f};
  const float norm[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
  in.substract_mean_normalize(mean, norm);

  ncnn::Extractor ex = det->net.create_extractor();
  ex.set_light_mode(true);
  ex.input("input", in);

  ncnn::Mat scores_mat, boxes_mat;
  ex.extract("scores", scores_mat);
  ex.extract("boxes",  boxes_mat);

  static std::vector<Anchor> priors = generate_priors(det->input_w, det->input_h);

  /* SSD variance-encoded decode:
   * cx = loc[0]*0.1*prior_w + prior_cx
   * cy = loc[1]*0.1*prior_h + prior_cy
   * w  = exp(loc[2]*0.2) * prior_w
   * h  = exp(loc[3]*0.2) * prior_h  */
  int num_anchors = scores_mat.h;
  std::vector<FaceBox> candidates;

  for (int i = 0; i < num_anchors; i++) {
    float face_score = scores_mat.row(i)[1];
    if (face_score < threshold) continue;

    const float *b = boxes_mat.row(i);
    const Anchor &p = priors[i];

    float cx = b[0] * 0.1f * p.w + p.cx;
    float cy = b[1] * 0.1f * p.h + p.cy;
    float pw = std::exp(b[2] * 0.2f) * p.w;
    float ph = std::exp(b[3] * 0.2f) * p.h;

    /* Normalised centre → pixel xywh in original image space */
    float x = (cx - pw * 0.5f) * src_w;
    float y = (cy - ph * 0.5f) * src_h;
    float w = pw * src_w;
    float h = ph * src_h;
    candidates.push_back({x, y, w, h, face_score});
  }

  return nms(candidates, 0.3f);
}

/* ================================================================
 * mruby wrappers
 * ================================================================ */

static void face_detector_free(mrb_state *mrb, void *p) {
  if (p) {
    FaceDetector *det = (FaceDetector *)p;
    det->net.clear();
    delete det;
  }
}

static const mrb_data_type face_detector_type = {"FaceDetector", face_detector_free};

/* FaceDetector.new(model_path [, use_gpu: true]) */
static mrb_value mrb_face_detector_new(mrb_state *mrb, mrb_value klass) {
  const char *model_path;
  mrb_value opts = mrb_nil_value();
  mrb_get_args(mrb, "z|H", &model_path, &opts);

  bool use_gpu = true;
  if (!mrb_nil_p(opts)) {
    mrb_value gpu_val = mrb_hash_get(mrb, opts,
                          mrb_symbol_value(mrb_intern_cstr(mrb, "use_gpu")));
    if (!mrb_nil_p(gpu_val)) {
      use_gpu = mrb_bool(gpu_val);
    }
  }

  FaceDetector *det = new FaceDetector();
  det->use_gpu = use_gpu;
  det->input_w = 320;
  det->input_h = 240;

  if (use_gpu) {
    det->net.opt.use_vulkan_compute = true;
  }

  char param_path[512], bin_path[512];
  snprintf(param_path, sizeof(param_path), "%s.param", model_path);
  snprintf(bin_path,   sizeof(bin_path),   "%s.bin",   model_path);

  if (det->net.load_param(param_path) != 0) {
    delete det;
    mrb_raisef(mrb, E_RUNTIME_ERROR, "FaceDetector: cannot load param: %s", param_path);
  }
  if (det->net.load_model(bin_path) != 0) {
    delete det;
    mrb_raisef(mrb, E_RUNTIME_ERROR, "FaceDetector: cannot load model: %s", bin_path);
  }

  struct RClass *cls = mrb_class_ptr(klass);
  struct RData *data = mrb_data_object_alloc(mrb, cls, det, &face_detector_type);
  return mrb_obj_value(data);
}

/* ---- Convert FaceBox vector to mruby Array of Hash ---- */
static mrb_value boxes_to_mrb_ary(mrb_state *mrb, const std::vector<FaceBox> &boxes) {
  mrb_value result = mrb_ary_new_capa(mrb, (mrb_int)boxes.size());
  mrb_sym sym_x     = mrb_intern_cstr(mrb, "x");
  mrb_sym sym_y     = mrb_intern_cstr(mrb, "y");
  mrb_sym sym_w     = mrb_intern_cstr(mrb, "w");
  mrb_sym sym_h     = mrb_intern_cstr(mrb, "h");
  mrb_sym sym_score = mrb_intern_cstr(mrb, "score");

  for (const auto &b : boxes) {
    mrb_value h = mrb_hash_new(mrb);
    mrb_hash_set(mrb, h, mrb_symbol_value(sym_x),     mrb_float_value(mrb, b.x));
    mrb_hash_set(mrb, h, mrb_symbol_value(sym_y),     mrb_float_value(mrb, b.y));
    mrb_hash_set(mrb, h, mrb_symbol_value(sym_w),     mrb_float_value(mrb, b.w));
    mrb_hash_set(mrb, h, mrb_symbol_value(sym_h),     mrb_float_value(mrb, b.h));
    mrb_hash_set(mrb, h, mrb_symbol_value(sym_score), mrb_float_value(mrb, b.score));
    mrb_ary_push(mrb, result, h);
  }
  return result;
}

/*
 * detector.detect_rgb(rgb_string, width, height [, threshold: 0.7])
 * RGB888 文字列を受け取って顔検出する。
 * YUYV→RGB変換は Camera.yuyv_to_rgb で事前に行うこと。
 */
static mrb_value mrb_face_detector_detect_rgb(mrb_state *mrb, mrb_value self) {
  FaceDetector *det = DATA_GET_PTR(mrb, self, &face_detector_type, FaceDetector);

  mrb_value frame;
  mrb_int src_w, src_h;
  mrb_value opts = mrb_nil_value();
  mrb_get_args(mrb, "Sii|H", &frame, &src_w, &src_h, &opts);

  float threshold = 0.7f;
  if (!mrb_nil_p(opts)) {
    mrb_value tv = mrb_hash_get(mrb, opts,
                     mrb_symbol_value(mrb_intern_cstr(mrb, "threshold")));
    if (!mrb_nil_p(tv)) threshold = (float)mrb_float(tv);
  }

  const uint8_t *rgb = (const uint8_t *)RSTRING_PTR(frame);
  size_t expected = (size_t)(src_w * src_h * 3);
  if ((size_t)RSTRING_LEN(frame) < expected)
    mrb_raise(mrb, E_ARGUMENT_ERROR, "RGB string too short for given width/height");

  std::vector<FaceBox> boxes = detect(det, rgb, (int)src_w, (int)src_h, threshold);
  return boxes_to_mrb_ary(mrb, boxes);
}

extern "C" void mrb_face_gem_init(mrb_state *mrb) {
  struct RClass *cls = mrb_define_class(mrb, "FaceDetector", mrb->object_class);
  MRB_SET_INSTANCE_TT(cls, MRB_TT_CDATA);

  mrb_define_class_method(mrb, cls, "new",        mrb_face_detector_new,        MRB_ARGS_REQ(1) | MRB_ARGS_OPT(1));
  mrb_define_method(mrb, cls, "detect_rgb", mrb_face_detector_detect_rgb, MRB_ARGS_REQ(3) | MRB_ARGS_OPT(1));
}

extern "C" void mrb_face_gem_final(mrb_state *mrb) {
  (void)mrb;
}
