/*
 * test_face.cpp — カメラ1フレーム取得→UltraFace推論→検出結果表示
 * Build:
 *   gcc -c ../src/v4l2_capture.c -I ../src -o /tmp/v4l2_capture.o
 *   g++ -std=c++17 -O2 -I/usr/local/include/ncnn -I../src \
 *       test_face.cpp /tmp/v4l2_capture.o \
 *       -L/usr/local/lib -lncnn -lvulkan \
 *       -lglslang -lSPIRV -lMachineIndependent -lGenericCodeGen -lglslang-default-resource-limits \
 *       -fopenmp -o /tmp/test_face
 */
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#include <ncnn/net.h>

extern "C" {
#include "v4l2_capture.h"
}

/* ---- YUYV -> RGB ---- */
static void yuyv_to_rgb(const uint8_t *yuyv, uint8_t *rgb, int w, int h) {
  for (int i = 0; i < w * h / 2; i++) {
    int y0=yuyv[i*4+0], u=yuyv[i*4+1]-128, y1=yuyv[i*4+2], v=yuyv[i*4+3]-128;
    auto cl=[](int x)->uint8_t{return x<0?0:x>255?255:(uint8_t)x;};
    rgb[(i*2)  *3+0]=cl(y0+(int)(1.402f*v)); rgb[(i*2)  *3+1]=cl(y0-(int)(0.344f*u)-(int)(0.714f*v)); rgb[(i*2)  *3+2]=cl(y0+(int)(1.772f*u));
    rgb[(i*2+1)*3+0]=cl(y1+(int)(1.402f*v)); rgb[(i*2+1)*3+1]=cl(y1-(int)(0.344f*u)-(int)(0.714f*v)); rgb[(i*2+1)*3+2]=cl(y1+(int)(1.772f*u));
  }
}

/* ---- Prior anchors (UltraFace-slim 320x240) ---- */
struct Anchor { float cx, cy, w, h; };
static std::vector<Anchor> generate_priors(int iw, int ih) {
  static const int strides[]       = {8, 16, 32, 64};
  static const int min_sizes[4][3] = {{10,16,24},{32,48,0},{64,96,0},{128,192,256}};
  static const int n_sizes[]       = {3, 2, 2, 3};
  std::vector<Anchor> p; p.reserve(4420);
  for (int s = 0; s < 4; s++) {
    int st=strides[s], fw=(iw+st-1)/st, fh=(ih+st-1)/st;
    for (int y=0;y<fh;y++) for (int x=0;x<fw;x++) {
      float cx=(x+0.5f)/fw, cy=(y+0.5f)/fh;
      for (int k=0;k<n_sizes[s];k++) {
        p.push_back({cx, cy, (float)min_sizes[s][k]/iw, (float)min_sizes[s][k]/ih});
      }
    }
  }
  return p;
}

/* ---- NMS ---- */
struct Box { float x, y, w, h, score; };
static float iou(const Box &a, const Box &b) {
  float ix1=std::max(a.x,b.x), iy1=std::max(a.y,b.y);
  float ix2=std::min(a.x+a.w,b.x+b.w), iy2=std::min(a.y+a.h,b.y+b.h);
  float inter=std::max(0.f,ix2-ix1)*std::max(0.f,iy2-iy1);
  float ua=a.w*a.h+b.w*b.h-inter;
  return ua>0?inter/ua:0.f;
}
static std::vector<Box> nms(std::vector<Box> boxes, float thr) {
  std::sort(boxes.begin(),boxes.end(),[](const Box &a,const Box &b){return a.score>b.score;});
  std::vector<bool> sup(boxes.size(),false);
  std::vector<Box> out;
  for (size_t i=0;i<boxes.size();i++) {
    if (sup[i]) continue;
    out.push_back(boxes[i]);
    for (size_t j=i+1;j<boxes.size();j++) if (!sup[j]&&iou(boxes[i],boxes[j])>thr) sup[j]=true;
  }
  return out;
}

int main(int argc, char *argv[]) {
  const char *model = argc > 1 ? argv[1] : "models/ultraface-slim";
  const char *dev   = argc > 2 ? argv[2] : "/dev/video0";
  float threshold   = argc > 3 ? atof(argv[3]) : 0.6f;

  /* Load model */
  ncnn::Net net;
  net.opt.use_vulkan_compute = true;
  char pp[512], bp[512];
  snprintf(pp,sizeof(pp),"%s.param",model); snprintf(bp,sizeof(bp),"%s.bin",model);
  if (net.load_param(pp)!=0||net.load_model(bp)!=0) { fprintf(stderr,"load model failed\n"); return 1; }
  printf("Model:  %s\n", model);

  auto priors = generate_priors(320, 240);
  printf("Priors: %zu anchors\n", priors.size());

  /* Capture */
  v4l2_camera_t cam;
  if (v4l2_camera_open(&cam, dev, 640, 480) < 0) { fprintf(stderr,"camera open failed\n"); return 1; }
  printf("Camera: %dx%d\n", cam.width, cam.height);

  /* Warm-up: skip first frame */
  { uint8_t *d; size_t l; v4l2_camera_capture(&cam,&d,&l); v4l2_camera_release(&cam); }

  uint8_t *yuyv; size_t yuyv_len;
  if (v4l2_camera_capture(&cam, &yuyv, &yuyv_len) < 0) { fprintf(stderr,"capture failed\n"); return 1; }

  /* YUYV -> RGB */
  auto t0 = std::chrono::steady_clock::now();
  std::vector<uint8_t> rgb(640*480*3);
  yuyv_to_rgb(yuyv, rgb.data(), 640, 480);
  v4l2_camera_release(&cam);
  v4l2_camera_close(&cam);

  /* Inference */
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data(), ncnn::Mat::PIXEL_RGB, 640, 480, 320, 240);
  const float mean[3]={127.f,127.f,127.f}, norm[3]={1.f/128.f,1.f/128.f,1.f/128.f};
  in.substract_mean_normalize(mean,norm);

  ncnn::Extractor ex = net.create_extractor();
  ex.input("input", in);
  ncnn::Mat scores_mat, boxes_mat;
  ex.extract("scores", scores_mat);
  ex.extract("boxes",  boxes_mat);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

  /* Decode + NMS */
  std::vector<Box> candidates;
  for (int i = 0; i < scores_mat.h; i++) {
    float score = scores_mat.row(i)[1];
    if (score < threshold) continue;
    const float *b = boxes_mat.row(i);
    const Anchor &p = priors[i];
    float cx = b[0]*0.1f*p.w + p.cx;
    float cy = b[1]*0.1f*p.h + p.cy;
    float pw = std::exp(b[2]*0.2f)*p.w;
    float ph = std::exp(b[3]*0.2f)*p.h;
    float x = (cx-pw*0.5f)*640, y = (cy-ph*0.5f)*480;
    float w = pw*640,            h = ph*480;
    candidates.push_back({x,y,w,h,score});
  }
  auto results = nms(candidates, 0.3f);

  printf("Time:   %.1f ms (YUYV->RGB + resize + inference)\n", ms);
  printf("Faces:  %zu detected (threshold=%.2f)\n", results.size(), threshold);
  for (size_t i = 0; i < results.size(); i++) {
    auto &f = results[i];
    printf("  [%zu] x=%.0f y=%.0f w=%.0f h=%.0f  score=%.3f\n",
           i, f.x, f.y, f.w, f.h, f.score);
  }
  return 0;
}
