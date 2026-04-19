puts "=" * 52
puts "  mruby-gpu People Counter Demo"
puts "=" * 52
puts

# ---- 引数解析 ----
# mruby examples/counter_demo.rb [method] [mode] [backend] [resolution]
# method:     skin | ncnn | both       (default: skin)
# mode:       snap | pan               (default: snap)
# backend:    cpu | gpu                (NCNN only, default: cpu)
# resolution: 480 | 720               (default: 720)
# --file PATH   : use image file instead of camera
# --video PATH  : use video file instead of camera
METHOD   = (ARGV[0] || "skin").to_sym
PAN_MODE = (ARGV[1] || "snap").to_sym
BACKEND  = (ARGV[2] || "cpu")
RES      = (ARGV[3] || "720").to_i

file_idx  = ARGV.index("--file")
video_idx = ARGV.index("--video")
FILE_INPUT  = file_idx  ? ARGV[file_idx + 1]  : nil
VIDEO_INPUT = video_idx ? ARGV[video_idx + 1] : nil

W = RES == 720 ? 1280 : 640
H = RES == 720 ? 720  : 480
SCALE = RES == 720 ? 3 : 2
THRESHOLD = 0.5

# ---- 初期化 ----
GPU.init("shader")
puts "GPU    : #{GPU.device_name}"

cam = nil
unless FILE_INPUT || VIDEO_INPUT
  cam = Camera.open("/dev/video0", W, H)
  puts "Camera : #{W}x#{H}"
end

disp = Display.open(W, H, "mruby counter demo — ESC to quit")

use_ncnn = (METHOD == :ncnn || METHOD == :both)
detector = use_ncnn ?
  FaceDetector.new("models/ultraface-slim", use_gpu: BACKEND == "gpu") : nil

puts "Method : #{METHOD}  |  Mode: #{PAN_MODE}  |  Backend: #{BACKEND}"
puts "Resolution: #{W}x#{H}  |  Scale: #{SCALE}"
puts "File: #{FILE_INPUT || VIDEO_INPUT || '(camera)'}"
puts

# ---- NMS (same as face_demo.rb) ----
def iou(a, b)
  ax2 = a[:x] + a[:w]; ay2 = a[:y] + a[:h]
  bx2 = b[:x] + b[:w]; by2 = b[:y] + b[:h]
  inter = [[0.0, [ax2, bx2].min - [a[:x], b[:x]].max].max,
           [0.0, [ay2, by2].min - [a[:y], b[:y]].max].max].inject(:*)
  ua = a[:w]*a[:h] + b[:w]*b[:h] - inter
  ua > 0 ? inter / ua : 0.0
end

def global_nms(faces, iou_thresh = 0.35)
  sorted = faces.sort_by { |f| -f[:score] }
  used   = Array.new(sorted.size, false)
  kept   = []
  sorted.each_with_index do |f, i|
    next if used[i]
    kept << f
    sorted.each_with_index do |g, j|
      used[j] = true if !used[j] && j != i && iou(f, g) > iou_thresh
    end
  end
  kept
end

# ---- NCNN tiling (Phase 3: 3x2 grid) ----
def make_tiles(w, h, cols, rows)
  tw = w / cols; th = h / rows
  tiles = []
  rows.times { |r| cols.times { |c| tiles << [c * tw, r * th, tw, th] } }
  tiles
end

TILES = use_ncnn ? make_tiles(W, H, 3, 2) : []

def detect_ncnn(rgb, detector, w, h, tiles, threshold)
  all = detector.detect_rgb(rgb, w, h, threshold: threshold)
  tiles.each do |tx, ty, tw, th|
    tile = Camera.crop_rgb(rgb, w, h, tx, ty, tw, th)
    detector.detect_rgb(tile, tw, th, threshold: threshold).each do |f|
      sx = w.to_f / tw; sy = h.to_f / th
      all << { x: f[:x]/sx+tx, y: f[:y]/sy+ty, w: f[:w]/sx, h: f[:h]/sy, score: f[:score] }
    end
  end
  global_nms(all)
end

# ---- 状態 ----
paused = false
snap_count = 0
total = 0
current_count = 0
show_guide = true
frame_no = 0
total_ms = 0.0
last_rgb = nil

# pan mode
pan_window = []
PAN_INTERVAL = 30

# ---- メインループ ----
loop do
  event = Display.poll_event
  case event
  when "quit" then break
  when "space"
    if PAN_MODE == :snap
      if paused
        paused = false
      else
        paused = true
        snap_count += 1
        total += current_count
        puts "SNAP ##{snap_count}: +#{current_count}  TOTAL: #{total}"
      end
    end
  when "g" then show_guide = !show_guide
  end

  # 一時停止中: 静止画 + PAUSED テキストを表示し続ける
  if paused && last_rgb
    rgb = last_rgb.dup
    rgb = disp.draw_text(rgb, W, H, 4, 4,
      "** PAUSED **  +#{current_count}  TOTAL:#{total}", 255, 255, 0, SCALE)
    rgb = disp.draw_text(rgb, W, H, 4, H - 8 * SCALE - 4,
      "[SPACE] resume  [ESC] quit", 200, 200, 200, SCALE)
    disp.show(rgb, W, H)
    next
  end

  # フレーム取得
  rgb = if FILE_INPUT
          Camera.load_rgb(FILE_INPUT)
        elsif VIDEO_INPUT
          Camera.read_frame(VIDEO_INPUT, W, H, frame_no)
        else
          yuyv = cam.capture
          Camera.yuyv_to_rgb(yuyv, W, H)
        end

  # 検出
  t0 = Time.now

  case METHOD
  when :skin
    result = SkinDetector.detect(rgb, W, H)
    current_count = result[:count]
  when :ncnn
    faces = detect_ncnn(rgb, detector, W, H, TILES, THRESHOLD)
    current_count = faces.size
    faces.each do |f|
      rgb = disp.draw_rect(rgb, W, H,
        f[:x].to_i, f[:y].to_i, f[:w].to_i, f[:h].to_i, 0, 255, 0)
    end
  when :both
    skin_r = SkinDetector.detect(rgb, W, H)
    faces = detect_ncnn(rgb, detector, W, H, TILES, THRESHOLD)
    current_count = skin_r[:count]
    faces.each do |f|
      rgb = disp.draw_rect(rgb, W, H,
        f[:x].to_i, f[:y].to_i, f[:w].to_i, f[:h].to_i, 0, 255, 0)
    end
  end

  elapsed = (Time.now - t0) * 1000.0
  total_ms += elapsed
  frame_no += 1
  last_rgb = rgb.dup

  # パンモード: 1秒ウィンドウの最大値を累積
  if PAN_MODE == :pan
    pan_window << current_count
    if pan_window.size >= PAN_INTERVAL
      total += pan_window.max
      pan_window.clear
    end
  end

  # テキストオーバーレイ
  fps = elapsed > 0 ? (1000.0 / elapsed) : 0
  top_text = "#{METHOD.to_s.upcase}  FPS:#{fps.round(0)}  NOW:#{current_count}  TOTAL:#{total}"
  rgb = disp.draw_text(rgb, W, H, 4, 4, top_text, 0, 255, 0, SCALE)

  if PAN_MODE == :snap
    bottom = "SNAP##{snap_count}  [SPACE]pause  [G]guide  [ESC]quit"
  else
    bottom = "PAN  [G]guide  [ESC]quit"
  end
  rgb = disp.draw_text(rgb, W, H, 4, H - 8 * SCALE - 4, bottom, 200, 200, 200, SCALE)

  # 補助線（3分割ガイド）
  if show_guide
    rgb = disp.draw_rect(rgb, W, H, W / 3, 0, 2, H, 0, 200, 0)
    rgb = disp.draw_rect(rgb, W, H, W * 2 / 3, 0, 2, H, 0, 200, 0)
  end

  disp.show(rgb, W, H)

  # ターミナル出力（30フレームごと）
  if frame_no % 30 == 0
    avg = total_ms / frame_no
    puts "Frame ##{frame_no}  |  #{current_count} 人  |  #{avg.round(1)}ms (#{(1000.0/avg).round(1)} FPS)  |  TOTAL: #{total}"
  end
end

# ---- 終了 ----
cam.close if cam
disp.close

puts
puts "=" * 52
puts "  終了サマリー"
puts "  モード       : #{METHOD} / #{PAN_MODE}"
puts "  総フレーム数 : #{frame_no}"
puts "  スナップ回数 : #{snap_count}" if PAN_MODE == :snap
puts "  合計人数     : #{total}"
if frame_no > 0
  avg = total_ms / frame_no
  puts "  平均処理時間 : #{avg.round(1)} ms/frame  (#{(1000.0/avg).round(1)} FPS)"
end
puts "=" * 52
