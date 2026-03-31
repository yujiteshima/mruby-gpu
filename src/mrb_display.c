/*
 * mrb_display.c — mruby Display class via SDL2 (framebuffer backend も可)
 *
 * Display.open(width, height, title="mruby-gpu") -> Display
 * disp.show(rgb_str, width, height)  # RGB888 バイト列を表示
 * disp.draw_rect(rgb_str, x, y, w, h, r, g, b) -> String  # 矩形を描画した新しい文字列
 * disp.close -> nil
 * Display.poll_quit -> bool   # ウィンドウ閉じるイベント確認
 */

#include <mruby.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>
#include <mruby/variable.h>

#include <SDL2/SDL.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
  SDL_Window   *window;
  SDL_Renderer *renderer;
  SDL_Texture  *texture;
  int width;
  int height;
} DisplayCtx;

static void display_free(mrb_state *mrb, void *p) {
  if (!p) return;
  DisplayCtx *ctx = (DisplayCtx *)p;
  if (ctx->texture)  SDL_DestroyTexture(ctx->texture);
  if (ctx->renderer) SDL_DestroyRenderer(ctx->renderer);
  if (ctx->window)   SDL_DestroyWindow(ctx->window);
  mrb_free(mrb, ctx);
  /* SDL_Quit は複数 Display を想定しないシンプル実装のため呼ばない */
}

static const struct mrb_data_type display_type = {"Display", display_free};

/* Display.open(width, height [, title]) */
static mrb_value mrb_display_open(mrb_state *mrb, mrb_value klass) {
  mrb_int w, h;
  const char *title = "mruby-gpu face demo";
  mrb_get_args(mrb, "ii|z", &w, &h, &title);

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    mrb_raisef(mrb, E_RUNTIME_ERROR, "SDL_Init failed: %s", SDL_GetError());
  }

  DisplayCtx *ctx = (DisplayCtx *)mrb_malloc(mrb, sizeof(DisplayCtx));
  ctx->width  = (int)w;
  ctx->height = (int)h;

  ctx->window = SDL_CreateWindow(title,
    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    (int)w, (int)h, SDL_WINDOW_SHOWN);
  if (!ctx->window) {
    mrb_free(mrb, ctx);
    mrb_raisef(mrb, E_RUNTIME_ERROR, "SDL_CreateWindow failed: %s", SDL_GetError());
  }

  ctx->renderer = SDL_CreateRenderer(ctx->window, -1, SDL_RENDERER_ACCELERATED);
  if (!ctx->renderer)
    ctx->renderer = SDL_CreateRenderer(ctx->window, -1, SDL_RENDERER_SOFTWARE);

  ctx->texture = SDL_CreateTexture(ctx->renderer,
    SDL_PIXELFORMAT_RGB24,
    SDL_TEXTUREACCESS_STREAMING,
    (int)w, (int)h);

  struct RClass *cls = mrb_class_ptr(klass);
  struct RData *data = mrb_data_object_alloc(mrb, cls, ctx, &display_type);
  return mrb_obj_value(data);
}

/* disp.show(rgb_str, width, height)  — RGB888 bytes */
static mrb_value mrb_display_show(mrb_state *mrb, mrb_value self) {
  mrb_value rgb_str;
  mrb_int src_w, src_h;
  mrb_get_args(mrb, "Sii", &rgb_str, &src_w, &src_h);

  DisplayCtx *ctx = DATA_GET_PTR(mrb, self, &display_type, DisplayCtx);

  /* If source size differs from texture, recreate texture */
  if (src_w != ctx->width || src_h != ctx->height) {
    if (ctx->texture) SDL_DestroyTexture(ctx->texture);
    ctx->texture = SDL_CreateTexture(ctx->renderer,
      SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
      (int)src_w, (int)src_h);
    ctx->width  = (int)src_w;
    ctx->height = (int)src_h;
  }

  SDL_UpdateTexture(ctx->texture, NULL,
    RSTRING_PTR(rgb_str), (int)(src_w * 3));
  SDL_RenderClear(ctx->renderer);
  SDL_RenderCopy(ctx->renderer, ctx->texture, NULL, NULL);
  SDL_RenderPresent(ctx->renderer);

  return mrb_nil_value();
}

/* disp.draw_rect(rgb_str, src_w, src_h, x, y, w, h, r, g, b) -> String
 * CPU側でバウンディングボックスを RGB 文字列に書き込む (thickness=2px) */
static mrb_value mrb_display_draw_rect(mrb_state *mrb, mrb_value self) {
  mrb_value rgb_str;
  mrb_int src_w, src_h, rx, ry, rw, rh, cr, cg, cb;
  mrb_get_args(mrb, "Siiiiiiiii",
    &rgb_str, &src_w, &src_h, &rx, &ry, &rw, &rh, &cr, &cg, &cb);

  /* コピーして書き込む */
  mrb_value out = mrb_str_dup(mrb, rgb_str);
  uint8_t *px = (uint8_t *)RSTRING_PTR(out);
  int W = (int)src_w, H = (int)src_h;
  int thickness = 2;

  uint8_t R = (uint8_t)(cr & 0xFF);
  uint8_t G = (uint8_t)(cg & 0xFF);
  uint8_t B = (uint8_t)(cb & 0xFF);

  /* clamp rect to image bounds */
  int x1 = (int)rx,        y1 = (int)ry;
  int x2 = x1 + (int)rw,  y2 = y1 + (int)rh;
  if (x1 < 0) x1 = 0;  if (y1 < 0) y1 = 0;
  if (x2 > W) x2 = W;  if (y2 > H) y2 = H;

  for (int y = y1; y < y2; y++) {
    for (int x = x1; x < x2; x++) {
      int on_border = (y - y1 < thickness || y2 - y - 1 < thickness ||
                       x - x1 < thickness || x2 - x - 1 < thickness);
      if (!on_border) continue;
      int idx = (y * W + x) * 3;
      px[idx]   = R;
      px[idx+1] = G;
      px[idx+2] = B;
    }
  }
  return out;
}

/* Display.poll_quit -> bool */
static mrb_value mrb_display_poll_quit(mrb_state *mrb, mrb_value klass) {
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_QUIT)               return mrb_true_value();
    if (e.type == SDL_KEYDOWN &&
        e.key.keysym.sym == SDLK_ESCAPE)  return mrb_true_value();
  }
  return mrb_false_value();
}

/* disp.close */
static mrb_value mrb_display_close(mrb_state *mrb, mrb_value self) {
  DisplayCtx *ctx = DATA_GET_PTR(mrb, self, &display_type, DisplayCtx);
  if (ctx->texture)  { SDL_DestroyTexture(ctx->texture);   ctx->texture  = NULL; }
  if (ctx->renderer) { SDL_DestroyRenderer(ctx->renderer); ctx->renderer = NULL; }
  if (ctx->window)   { SDL_DestroyWindow(ctx->window);     ctx->window   = NULL; }
  SDL_Quit();
  return mrb_nil_value();
}

void mrb_display_gem_init(mrb_state *mrb) {
  struct RClass *cls = mrb_define_class(mrb, "Display", mrb->object_class);
  MRB_SET_INSTANCE_TT(cls, MRB_TT_CDATA);

  mrb_define_class_method(mrb, cls, "open",       mrb_display_open,       MRB_ARGS_REQ(2) | MRB_ARGS_OPT(1));
  mrb_define_class_method(mrb, cls, "poll_quit",  mrb_display_poll_quit,  MRB_ARGS_NONE());
  mrb_define_method(mrb, cls, "show",             mrb_display_show,       MRB_ARGS_REQ(3));
  mrb_define_method(mrb, cls, "draw_rect",        mrb_display_draw_rect,  MRB_ARGS_REQ(10));
  mrb_define_method(mrb, cls, "close",            mrb_display_close,      MRB_ARGS_NONE());
}

void mrb_display_gem_final(mrb_state *mrb) { (void)mrb; }
