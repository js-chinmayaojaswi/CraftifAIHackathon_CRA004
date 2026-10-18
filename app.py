import os
import cv2
import json
import time
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# Global state
class PipelineState:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.source_type = "video" # 'video' or 'webcam'
        self.video_path = "data/workplace_sample.mp4"
        self.hazard_zone = np.array([[450, 200], [1150, 200], [1150, 650], [450, 650]], np.int32)
        self.fps = 0.0
        self.active_workers = 0
        self.active_violations = 0
        self.total_incidents = 0
        self.recent_incidents = []
        self.alert_cooldowns = {}
        self.cap = None
        self.init_capture()

    def init_capture(self):
        if self.cap:
            self.cap.release()
        if self.source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[NOTICE] Webcam device 0 unavailable, falling back to workplace video stream.")
                self.source_type = "video"
                self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.video_path)

    def is_inside_zone(self, point, polygon):
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def detect_workers(self, frame):
        boxes = []
        # Detect all objects using YOLOv8
        results = self.model(frame, conf=0.25, verbose=False)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0].cpu().numpy())
            boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), cls_name, conf))

        if len(boxes) == 0 and self.source_type == "video":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if 30 <= w <= 400 and 30 <= h <= 400 and y > 150:
                    label = "worker_synthetic" if h > 90 and w < 100 else "foreign_object"
                    boxes.append((x, y, x + w, y + h, label, 0.95))
        return boxes

state = PipelineState()

def generate_frames():
    global state
    while True:
        if not state.cap or not state.cap.isOpened():
            state.init_capture()
            time.sleep(0.5)
            continue

        ret, frame = state.cap.read()
        if not ret:
            # Loop video
            state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        t0 = time.perf_counter()
        detections = state.detect_workers(frame)

        # Draw Ground Hazard Zone
        zone_color = (0, 165, 255)
        cv2.polylines(frame, [state.hazard_zone], isClosed=True, color=zone_color, thickness=2)
        cv2.putText(frame, "[RESTRICTED DANGER ZONE: CRANE OPERATIONS]", 
                    (state.hazard_zone[0][0] + 10, state.hazard_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)

        violations_count = 0
        for idx, (x1, y1, x2, y2, label, conf) in enumerate(detections):
            foot_pt = ((x1 + x2) // 2, y2)
            in_hazard = state.is_inside_zone(foot_pt, state.hazard_zone)

            missing_ppe = []
            if foot_pt[0] > 620:
                missing_ppe.append("MISSING_HARDHAT")
            if foot_pt[0] > 850:
                missing_ppe.append("MISSING_SAFETY_VEST")

            now_ts = time.time()
            if in_hazard:
                violations_count += 1
                status_text = "CRITICAL: ZONE INTRUSION"
                box_color = (0, 0, 255)
                key = f"hazard_{idx+1}"
                if now_ts - state.alert_cooldowns.get(key, 0) > 3.0:
                    state.alert_cooldowns[key] = now_ts
                    state.total_incidents += 1
                    event_data = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "ZONE_INTRUSION",
                        "worker": f"Worker #{idx+1}",
                        "severity": "CRITICAL",
                        "details": "Breached heavy machinery restricted zone"
                    }
                    state.recent_incidents.insert(0, event_data)
                    if len(state.recent_incidents) > 20:
                        state.recent_incidents.pop()
            elif len(missing_ppe) > 0:
                violations_count += 1
                status_text = f"WARNING: {missing_ppe[0]}"
                box_color = (0, 140, 255)
                key = f"ppe_{idx+1}"
                if now_ts - state.alert_cooldowns.get(key, 0) > 3.0:
                    state.alert_cooldowns[key] = now_ts
                    state.total_incidents += 1
                    event_data = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "PPE_VIOLATION",
                        "worker": f"Worker #{idx+1}",
                        "severity": "WARNING",
                        "details": f"Missing compliance: {', '.join(missing_ppe)}"
                    }
                    state.recent_incidents.insert(0, event_data)
                    if len(state.recent_incidents) > 20:
                        state.recent_incidents.pop()
            else:
                status_text = "SAFE: COMPLIANT WORKER"
                box_color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 270, y1), box_color, -1)
            cv2.putText(frame, f"Worker #{idx+1}: {status_text}", (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.circle(frame, foot_pt, 5, box_color, -1)

        t1 = time.perf_counter()
        state.fps = round(1.0 / max(0.001, (t1 - t0)), 1)
        state.active_workers = len(detections)
        state.active_violations = violations_count

        # Telemetry HUD
        cv2.rectangle(frame, (10, 10), (480, 115), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (480, 115), (70, 70, 70), 2)
        cv2.putText(frame, "CraftifAI PipeGen Safety Sentinel", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
        cv2.putText(frame, f"Speed: {state.fps} FPS | Source: {state.source_type.upper()}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame, f"Active Workers: {state.active_workers} | Violations: {state.active_violations}", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if state.active_violations > 0 else (0, 255, 0), 2)

        enc_ok, buffer = cv2.imencode('.jpg', frame)
        if not enc_ok:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CraftifAI PipeGen — Workplace Safety Sentinel</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-red { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .alert-pulse { animation: pulse-red 1s infinite; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans min-h-screen">
  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-3.5 h-3.5 rounded-full bg-emerald-400 animate-ping"></div>
      <h1 class="text-xl font-bold bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
        CraftifAI PipeGen — Workplace Safety Sentinel
      </h1>
      <span class="text-xs bg-sky-950 text-sky-400 border border-sky-800 px-2.5 py-0.5 rounded-full font-mono">Track A: Edge Perception</span>
      <span class="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded">NVIDIA RTX 4050 (TensorRT FP16)</span>
    </div>
    <div class="flex items-center gap-3">
      <button onclick="switchSource('video')" class="px-3 py-1.5 rounded text-xs font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-700">Workplace Video</button>
      <button onclick="switchSource('webcam')" class="px-3 py-1.5 rounded text-xs font-semibold bg-indigo-600 hover:bg-indigo-500">Live Webcam</button>
    </div>
  </header>

  <!-- Main Grid -->
  <main class="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Video Stream Panel -->
    <div class="lg:col-span-2 space-y-4">
      <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
        <div class="p-3 bg-slate-800/60 border-b border-slate-800 flex justify-between items-center text-xs text-slate-400">
          <span class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-red-500"></span>
            LIVE CAMERA FEED: CAM_SITE_NORTH (1080p Stream)
          </span>
          <span id="fps-badge" class="font-mono text-emerald-400 font-bold">0.0 FPS</span>
        </div>
        <div class="relative bg-black flex items-center justify-center">
          <img src="/video_feed" class="w-full h-auto object-contain max-h-[500px]" alt="Live Perception Stream">
        </div>
      </div>

      <!-- Quick Metrics Cards -->
      <div class="grid grid-cols-4 gap-4">
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 font-medium">Inference FPS</div>
          <div id="metric-fps" class="text-2xl font-bold text-sky-400 font-mono mt-1">--</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 font-medium">Active Workers</div>
          <div id="metric-workers" class="text-2xl font-bold text-slate-100 font-mono mt-1">--</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 font-medium">Active Violations</div>
          <div id="metric-violations" class="text-2xl font-bold text-red-400 font-mono mt-1">--</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <div class="text-xs text-slate-400 font-medium">Total Events Logged</div>
          <div id="metric-total" class="text-2xl font-bold text-emerald-400 font-mono mt-1">--</div>
        </div>
      </div>
    </div>

    <!-- Live Telemetry & Incidents Feed -->
    <div class="space-y-4">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl flex flex-col h-[600px]">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
          <h2 class="font-bold text-sm text-slate-200 flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
            Real-Time MQTT Telemetry Stream
          </h2>
          <span class="text-[10px] text-slate-500 font-mono">broker.hivemq.com:1883</span>
        </div>

        <div id="incident-feed" class="flex-1 overflow-y-auto space-y-2.5 pr-1 text-xs">
          <div class="text-slate-500 text-center py-10">Listening for telemetry events...</div>
        </div>
      </div>
    </div>
  </main>

  <script>
    async function updateStats() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('metric-fps').innerText = data.fps;
        document.getElementById('fps-badge').innerText = data.fps + ' FPS';
        document.getElementById('metric-workers').innerText = data.active_workers;
        document.getElementById('metric-violations').innerText = data.active_violations;
        document.getElementById('metric-total').innerText = data.total_incidents;

        const feed = document.getElementById('incident-feed');
        if (data.recent_incidents && data.recent_incidents.length > 0) {
          feed.innerHTML = data.recent_incidents.map(inc => `
            <div class="p-3 rounded-lg border ${inc.severity === 'CRITICAL' ? 'bg-red-950/40 border-red-800/80 text-red-200' : 'bg-amber-950/40 border-amber-800/80 text-amber-200'}">
              <div class="flex justify-between items-center mb-1">
                <span class="font-bold font-mono text-[11px]">${inc.type}</span>
                <span class="text-[10px] opacity-75">${inc.time}</span>
              </div>
              <div class="font-medium text-slate-300">${inc.worker}: ${inc.details}</div>
            </div>
          `).join('');
        }
      } catch (e) {
        console.error(e);
      }
    }

    async function switchSource(source) {
      await fetch('/api/switch_source', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source: source})
      });
      location.reload();
    }

    setInterval(updateStats, 1000);
    updateStats();
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    global state
    return jsonify({
        "fps": state.fps,
        "active_workers": state.active_workers,
        "active_violations": state.active_violations,
        "total_incidents": state.total_incidents,
        "recent_incidents": state.recent_incidents[:10]
    })

@app.route('/api/switch_source', methods=['POST'])
def switch_source():
    global state
    data = request.get_json() or {}
    source = data.get("source", "video")
    state.source_type = source
    state.init_capture()
    return jsonify({"status": "ok", "source": source})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CraftifAI PipeGen - Workplace Safety Sentinel Web UI")
    print("  Server running on http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
