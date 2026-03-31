puts "=" * 52
puts "  mruby + Vulkan GPU  Face Detection Demo"
puts "  Raspberry Pi 5  —  People Counter"
puts "=" * 52
puts

# ---- モード選択 -----------------------------------------------
# 引数 fast  : 全体1回のみ  ~6 FPS, ~3m まで検出
# 引数 count : 全体+5タイル ~1 FPS, ~7m まで検出（デフォルト）
# 実行例: mruby face_demo.rb fast
MODE = (ARGV[0] || "count").to_sym

THRESHOLD = case MODE
            when :fast  then 0.6
            when :count then 0.5
            end
# ---------------------------------------------------------------

GPU.init("shader")
puts "GPU    : #{GPU.device_name}"

cam      = Camera.open("/dev/video0", 640, 480)
detector = FaceDetector.new("models/ultraface-slim", use_gpu: true)
disp     = Display.open(cam.width, cam.height, "mruby face demo  —  ESC to quit")

W = cam.width   # 640
H = cam.height  # 480

puts "Camera : #{W}x#{H} @ 30fps"
puts "Model  : UltraFace-slim (NCNN + Vulkan GPU)"
puts "Mode   : #{MODE}  (引数 fast|count で切替)"
puts "ESC またはウィンドウを閉じると終了します。"
puts

# ---- タイル定義（count モード）
# 全体 + 4隅タイル + 中央タイル = 6パス
# 各タイルは 320x240 = モデル入力と同サイズ、実効2倍解像度で遠距離対応
TILES = (MODE == :count) ? [
  [0,       0,       W/2, H/2],   # 左上
  [W/2,     0,       W/2, H/2],   # 右上
  [0,       H/2,     W/2, H/2],   # 左下
  [W/2,     H/2,     W/2, H/2],   # 右下
  [W/4,     H/4,     W/2, H/2],   # 中央
] : []

# ---- グローバル NMS（タイル間重複除去）
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

# ---- メインループ
frame_count = 0
total_ms    = 0.0
peak_people = 0

loop do
  break if Display.poll_quit

  t0 = Time.now

  # 1. キャプチャ & RGB 変換
  yuyv = cam.capture
  rgb  = Camera.yuyv_to_rgb(yuyv, W, H)

  # 2. 全体画像で検出
  all_faces = detector.detect_rgb(rgb, W, H, threshold: THRESHOLD)

  # 3. タイルごとに検出（count モードのみ）
  TILES.each do |tx, ty, tw, th|
    tile = Camera.crop_rgb(rgb, W, H, tx, ty, tw, th)
    detector.detect_rgb(tile, tw, th, threshold: THRESHOLD).each do |f|
      # タイル座標 → 元画像座標（アスペクト比を保って逆スケール）
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

  # 5. 検出枠を描画（緑）
  faces.each do |f|
    rgb = disp.draw_rect(rgb, W, H,
                         f[:x].to_i, f[:y].to_i, f[:w].to_i, f[:h].to_i,
                         0, 255, 0)
  end

  # 6. 表示
  disp.show(rgb, W, H)

  elapsed      = (Time.now - t0) * 1000.0
  total_ms    += elapsed
  frame_count += 1
  people_count = faces.size
  peak_people  = people_count if people_count > peak_people

  if frame_count % 30 == 0
    avg = total_ms / frame_count
    fps = 1000.0 / avg
    puts "Frame ##{frame_count}  |  #{people_count} 人検出  " \
         "|  #{avg.round(1)}ms (#{fps.round(1)} FPS)  " \
         "|  最大検出: #{peak_people} 人"
    faces.each_with_index do |f, i|
      puts "  [#{i}] (#{f[:x].round}, #{f[:y].round}) " \
           "#{f[:w].round}x#{f[:h].round}  score=#{f[:score].round(2)}"
    end
  end
end

disp.close
cam.close

puts
puts "=" * 52
puts "  終了サマリー"
puts "  モード       : #{MODE}"
puts "  総フレーム数 : #{frame_count}"
puts "  最大同時検出 : #{peak_people} 人"
if frame_count > 0
  avg = total_ms / frame_count
  puts "  平均処理時間 : #{avg.round(1)} ms/frame  (#{(1000.0/avg).round(1)} FPS)"
end
puts "=" * 52
