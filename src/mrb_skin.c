/*
 * mrb_skin.c — SkinDetector class: skin-color based people counting
 *
 * SkinDetector.detect(rgb_str, width, height [, opts]) -> Hash
 *   { count: Integer, skin_ratio: Float, mask: String }
 *
 * Algorithm:
 *   1. RGB -> YCbCr (BT.601 integer math)
 *   2. Skin test: Cb in [cb_lo, cb_hi] && Cr in [cr_lo, cr_hi]
 *   3. Grid cells (CELL x CELL pixels) — count skin pixels per cell
 *   4. Mark cells with >= 30% skin pixels
 *   5. Flood-fill to group adjacent marked cells
 *   6. Group count = estimated face count
 */

#include <mruby.h>
#include <mruby/class.h>
#include <mruby/hash.h>
#include <mruby/string.h>
#include <string.h>
#include <stdlib.h>

#define DEFAULT_CELL      10
#define DEFAULT_CB_LO     77
#define DEFAULT_CB_HI    127
#define DEFAULT_CR_LO    133
#define DEFAULT_CR_HI    173
#define SKIN_CELL_THRESH  0.3f

/* ---- RGB -> YCbCr skin test (BT.601 integer approximation) ---- */
static inline int is_skin(uint8_t r, uint8_t g, uint8_t b,
                          int cb_lo, int cb_hi, int cr_lo, int cr_hi) {
  int cb = ((-43 * (int)r - 85 * (int)g + 128 * (int)b) >> 8) + 128;
  int cr = ((128 * (int)r - 107 * (int)g - 21 * (int)b) >> 8) + 128;
  return (cb >= cb_lo && cb <= cb_hi && cr >= cr_lo && cr <= cr_hi);
}

/* ---- Stack-based flood fill on grid ---- */
#define MAX_GRID_CELLS (256 * 192)  /* enough for 2560x1920 @ cell=10 */

static void flood_fill(const uint8_t *grid, uint8_t *visited,
                       int gw, int gh, int sx, int sy) {
  int stack[MAX_GRID_CELLS];
  int top = 0;
  int si = sy * gw + sx;
  stack[top++] = si;
  visited[si] = 1;

  while (top > 0) {
    int idx = stack[--top];
    int cx = idx % gw;
    int cy = idx / gw;
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};
    for (int d = 0; d < 4; d++) {
      int nx = cx + dx[d];
      int ny = cy + dy[d];
      if (nx < 0 || nx >= gw || ny < 0 || ny >= gh) continue;
      int ni = ny * gw + nx;
      if (!visited[ni] && grid[ni]) {
        visited[ni] = 1;
        if (top < MAX_GRID_CELLS) stack[top++] = ni;
      }
    }
  }
}

/* ---- helper: read integer from opts Hash ---- */
static int opt_int(mrb_state *mrb, mrb_value opts, const char *key, int def) {
  if (mrb_nil_p(opts)) return def;
  mrb_value v = mrb_hash_get(mrb, opts,
                  mrb_symbol_value(mrb_intern_cstr(mrb, key)));
  return mrb_nil_p(v) ? def : (int)mrb_integer(v);
}

/* ---- SkinDetector.detect(rgb, w, h [, opts]) -> Hash ---- */
static mrb_value mrb_skin_detect(mrb_state *mrb, mrb_value klass) {
  mrb_value rgb_str;
  mrb_int w, h;
  mrb_value opts = mrb_nil_value();
  mrb_get_args(mrb, "Sii|H", &rgb_str, &w, &h, &opts);

  int cb_lo     = opt_int(mrb, opts, "cb_low",  DEFAULT_CB_LO);
  int cb_hi     = opt_int(mrb, opts, "cb_high", DEFAULT_CB_HI);
  int cr_lo     = opt_int(mrb, opts, "cr_low",  DEFAULT_CR_LO);
  int cr_hi     = opt_int(mrb, opts, "cr_high", DEFAULT_CR_HI);
  int cell_size = opt_int(mrb, opts, "cell",    DEFAULT_CELL);
  if (cell_size < 1) cell_size = 1;

  const uint8_t *src = (const uint8_t *)RSTRING_PTR(rgb_str);
  int total_pixels = (int)(w * h);
  int skin_total = 0;

  /* Grid dimensions */
  int gw = ((int)w + cell_size - 1) / cell_size;
  int gh = ((int)h + cell_size - 1) / cell_size;
  int grid_size = gw * gh;
  int *cell_count = (int *)mrb_calloc(mrb, (size_t)grid_size, sizeof(int));

  /* Mask image (RGB, same size as input) */
  mrb_int mask_len = w * h * 3;
  mrb_value mask_str = mrb_str_buf_new(mrb, mask_len);
  uint8_t *mask = (uint8_t *)RSTRING_PTR(mask_str);
  memset(mask, 0, (size_t)mask_len);

  /* 1. Pixel scan: skin test + cell accumulation + mask */
  for (int py = 0; py < (int)h; py++) {
    for (int px = 0; px < (int)w; px++) {
      int idx = (py * (int)w + px) * 3;
      if (is_skin(src[idx], src[idx + 1], src[idx + 2],
                  cb_lo, cb_hi, cr_lo, cr_hi)) {
        skin_total++;
        cell_count[(py / cell_size) * gw + (px / cell_size)]++;
        mask[idx] = mask[idx + 1] = mask[idx + 2] = 255;
      }
    }
  }
  RSTR_SET_LEN(mrb_str_ptr(mask_str), mask_len);

  /* 2. Mark cells with sufficient skin ratio */
  uint8_t *grid = (uint8_t *)mrb_calloc(mrb, (size_t)grid_size, 1);
  for (int i = 0; i < grid_size; i++) {
    /* edge cells may be smaller, compute actual pixel count */
    int cx = i % gw, cy = i / gw;
    int cw = ((cx + 1) * cell_size > (int)w) ? ((int)w - cx * cell_size) : cell_size;
    int ch = ((cy + 1) * cell_size > (int)h) ? ((int)h - cy * cell_size) : cell_size;
    int cell_pixels = cw * ch;
    if (cell_pixels > 0 &&
        (float)cell_count[i] / (float)cell_pixels >= SKIN_CELL_THRESH) {
      grid[i] = 1;
    }
  }

  /* 3. Flood-fill to count connected groups */
  uint8_t *visited = (uint8_t *)mrb_calloc(mrb, (size_t)grid_size, 1);
  int count = 0;
  for (int i = 0; i < grid_size; i++) {
    if (grid[i] && !visited[i]) {
      flood_fill(grid, visited, gw, gh, i % gw, i / gw);
      count++;
    }
  }

  mrb_free(mrb, visited);
  mrb_free(mrb, grid);
  mrb_free(mrb, cell_count);

  /* Build result Hash */
  mrb_value result = mrb_hash_new(mrb);
  mrb_hash_set(mrb, result,
    mrb_symbol_value(mrb_intern_cstr(mrb, "count")),
    mrb_fixnum_value(count));
  mrb_hash_set(mrb, result,
    mrb_symbol_value(mrb_intern_cstr(mrb, "skin_ratio")),
    mrb_float_value(mrb, total_pixels > 0
      ? (mrb_float)skin_total / (mrb_float)total_pixels : 0.0));
  mrb_hash_set(mrb, result,
    mrb_symbol_value(mrb_intern_cstr(mrb, "mask")),
    mask_str);

  return result;
}

/* ---- gem init / final ---- */
void mrb_skin_gem_init(mrb_state *mrb) {
  struct RClass *cls = mrb_define_class(mrb, "SkinDetector", mrb->object_class);
  mrb_define_class_method(mrb, cls, "detect", mrb_skin_detect,
                          MRB_ARGS_REQ(3) | MRB_ARGS_OPT(1));
}

void mrb_skin_gem_final(mrb_state *mrb) {
  (void)mrb;
}
