puts "=" * 52
puts "  mruby + Vulkan GPU  People Counter Demo 2"
puts "  SPACE を押すたびに現在の人数を合計に加算"
puts "=" * 52
puts

# ---- 設定 -----------------------------------------------------
# face_demo.rb の count cpu モードをベースにした固定設定
USE_GPU   = false
THRESHOLD = 0.5
# ---------------------------------------------------------------

GPU.init("shader")
puts "GPU    : #{GPU.device_name}"

cam      = Camera.open("/dev/video0", 640, 480)
detector = FaceDetector.new("models/ultraface-slim", use_gpu: USE_GPU)
disp     = Display.open(cam.width, cam.height, "mruby counter demo 2  —  SPACE to add, ESC to quit")

W = cam.width   # 640
H = cam.height  # 480

puts "Camera : #{W}x#{H} @ 30fps"
puts "Model  : UltraFace-slim (NCNN CPU NEON)"
puts "Mode   : count (5-tile)"
puts "操作   : [SPACE] 現在の人数を合計に加算  [ESC] 終了"
puts

# ---- タイル定義（count モードと同じ 5 タイル） ----
TILES = [
  [0,   0,   W/2, H/2],  # 左上
  [W/2, 0,   W/2, H/2],  # 右上
  [0,   H/2, W/2, H/2],  # 左下
  [W/2, H/2, W/2, H/2],  # 右下
  [W/4, H/4, W/2, H/2],  # 中央
]

# ---- グローバル NMS ----
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

# ---- 状態 ----
frame_count   = 0
total_ms      = 0.0
current_count = 0
total         = 0
snap_count    = 0
last_added    = 0

# ---- メインループ ----
loop do
  # イベント処理
  event = Display.poll_event
  case event
  when "quit"
    break
  when "space"
    last_added  = current_count
    total      += current_count
    snap_count += 1
    puts "SNAP ##{snap_count}: +#{current_count}  TOTAL: #{total}"
  end

  t0 = Time.now

  # 1. キャプチャ & RGB 変換
  yuyv = cam.capture
  rgb  = Camera.yuyv_to_rgb(yuyv, W, H)

  # 2. 全体画像で検出
  all_faces = detector.detect_rgb(rgb, W, H, threshold: THRESHOLD)

  # 3. タイルごとに検出
  TILES.each do |tx, ty, tw, th|
    tile = Camera.crop_rgb(rgb, W, H, tx, ty, tw, th)
    detector.detect_rgb(tile, tw, th, threshold: THRESHOLD).each do |f|
      sx = W.to_f / tw
      sy = H.to_f / th
      all_faces << {
        x: f[:x] / sx + tx, y: f[:y] / sy + ty,
        w: f[:w] / sx,       h: f[:h] / sy,
        score: f[:score]
      }
    end
  end

  # 4. グローバル NMS で重複除去
  faces = global_nms(all_faces)
  current_count = faces.size

  # 5. 検出枠を描画（緑）
  faces.each do |f|
    rgb = disp.draw_rect(rgb, W, H,
                         f[:x].to_i, f[:y].to_i, f[:w].to_i, f[:h].to_i,
                         0, 255, 0)
  end

  # 6. テキストオーバーレイ
  elapsed_tmp = (Time.now - t0) * 1000.0
  fps_now = elapsed_tmp > 0 ? (1000.0 / elapsed_tmp) : 0

  # 左上(小・緑): FPS + NOW
  info = "CPU  FPS:#{fps_now.round(0)}  NOW:#{current_count}"
  rgb = disp.draw_text(rgb, W, H, 4, 4, info, 0, 255, 0, 2)

  # 左上の下(大・黄色): 合計を目立たせる
  rgb = disp.draw_text(rgb, W, H, 4, 28, "TOTAL: #{total}", 255, 255, 0, 4)

  # 最下部(小・グレー): 操作ヒントと直近追加値
  hint = "SNAP##{snap_count}  +#{last_added}  [SPACE]add  [ESC]quit"
  rgb = disp.draw_text(rgb, W, H, 4, H - 20, hint, 200, 200, 200, 2)

  # 7. 表示
  disp.show(rgb, W, H)

  elapsed      = (Time.now - t0) * 1000.0
  total_ms    += elapsed
  frame_count += 1

  if frame_count % 30 == 0
    avg = total_ms / frame_count
    fps = 1000.0 / avg
    puts "Frame ##{frame_count}  |  NOW:#{current_count} 人  " \
         "|  #{avg.round(1)}ms (#{fps.round(1)} FPS)  " \
         "|  TOTAL: #{total}"
  end
end

disp.close
cam.close

puts
puts "=" * 52
puts "  終了サマリー"
puts "  総フレーム数 : #{frame_count}"
puts "  スナップ回数 : #{snap_count}"
puts "  合計人数     : #{total}"
if frame_count > 0
  avg = total_ms / frame_count
  puts "  平均処理時間 : #{avg.round(1)} ms/frame  (#{(1000.0/avg).round(1)} FPS)"
end
puts "=" * 52
