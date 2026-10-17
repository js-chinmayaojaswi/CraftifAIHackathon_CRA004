import os
import cv2
import time
import json
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

class UniversalRestrictedZoneSentinel:
    def __init__(self, video_source="data/workplace_sample.mp4", output_video="output_safety_sentinel.mp4", log_file="incident_log.jsonl", use_mqtt=True):
        self.video_source = video_source
        self.output_video = output_video
        self.log_file = log_file
        self.use_mqtt = use_mqtt

        print("[STAGE 5] Initializing CraftifAI PipeGen Universal Object Perception Engine...")
        
        # Load YOLO Model (PyTorch / ONNX / TensorRT accelerated)
        self.model = YOLO("yolov8n.pt")
        print("[SUCCESS] Loaded YOLOv8 Edge Perception Model (All Object Classes Enabled).")

        # Restricted Hazard Polygon (Heavy Crane / Machinery Zone)
        self.hazard_zone = np.array([[450, 200], [1150, 200], [1150, 650], [450, 650]], np.int32)

        # Setup MQTT Client
        self.mqtt_client = None
        if self.use_mqtt and MQTT_AVAILABLE:
            try:
                self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "PipeGen_Sentinel_Universal_Node")
                self.mqtt_client.connect("broker.hivemq.com", 1883, 60)
                self.mqtt_client.loop_start()
                print("[SUCCESS] Connected to live MQTT Broker (broker.hivemq.com:1883).")
            except Exception as e:
                print(f"[NOTICE] Running local telemetry logger (broker unreachable): {e}")
                self.mqtt_client = None

        # Reset incident log
        with open(self.log_file, "w") as f:
            pass
        print(f"[SUCCESS] Incident Audit Trail initialized: {self.log_file}")

    def is_inside_zone(self, point, polygon):
        """Returns True if point (x, y) is inside the polygon."""
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def does_bbox_intersect_zone(self, bbox, polygon):
        """Checks if any corner or center of bbox is inside the polygon."""
        x1, y1, x2, y2 = bbox
        pts = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            ((x1 + x2) // 2, (y1 + y2) // 2),
            ((x1 + x2) // 2, y2)
        ]
        for pt in pts:
            if self.is_inside_zone(pt, polygon):
                return True
        return False

    def dispatch_alert(self, event_type, object_type, object_id, bbox, violation_details):
        """Publish real-time telemetry over MQTT and write to JSON incident log."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        incident_record = {
            "timestamp": timestamp,
            "pipeline": "CraftifAI_PipeGen_Universal_Sentinel_v0.5.1",
            "camera_id": "CAM_SITE_NORTH",
            "event_type": event_type,
            "severity": "CRITICAL",
            "object_type": object_type,
            "target_id": object_id,
            "bounding_box": [int(v) for v in bbox],
            "violations": violation_details,
            "action": "BROADCAST_TRIGGERED"
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(incident_record) + "\n")

        if self.mqtt_client:
            payload = json.dumps(incident_record)
            self.mqtt_client.publish("craftifai/pipegen/workplace_safety/alerts", payload)

    def detect_all_objects(self, frame):
        """Detect any object (persons, vehicles, tools, cargo) using YOLOv8 + fallback blob contour detection."""
        boxes = []
        # 1. Run YOLOv8 on all classes
        results = self.model(frame, conf=0.25, verbose=False)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0].cpu().numpy())
            boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), cls_name, conf))

        # 2. Contour detection for any synthetic objects or unknown foreign items
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter reasonable object sizes
            if 30 <= w <= 400 and 30 <= h <= 400 and y > 150:
                # Avoid duplicates if already covered by YOLO
                already_covered = False
                for bx1, by1, bx2, by2, _, _ in boxes:
                    if abs(x - bx1) < 40 and abs(y - by1) < 40:
                        already_covered = True
                        break
                if not already_covered:
                    label = "worker_synthetic" if h > 90 and w < 100 else "foreign_object"
                    boxes.append((x, y, x + w, y + h, label, 0.95))

        return boxes

    def run(self, max_frames=None, show_window=False):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        print(f"\n[LIVE RUN] Processing stream: {self.video_source} ({width}x{height} @ {fps} FPS)...")

        frame_count = 0
        total_violations = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            frame_count += 1
            t_frame_start = time.perf_counter()

            # Detect ALL objects
            detections = self.detect_all_objects(frame)

            # Draw Ground Hazard Zone
            zone_color = (0, 165, 255)
            cv2.polylines(frame, [self.hazard_zone], isClosed=True, color=zone_color, thickness=2)
            cv2.putText(frame, "[RESTRICTED DANGER ZONE: CRANE OPERATIONS]", (self.hazard_zone[0][0] + 10, self.hazard_zone[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)

            frame_violations = 0

            # Process all detected objects
            for idx, (x1, y1, x2, y2, label, conf) in enumerate(detections):
                bbox = (x1, y1, x2, y2)
                in_hazard = self.does_bbox_intersect_zone(bbox, self.hazard_zone)

                if in_hazard:
                    frame_violations += 1
                    total_violations += 1
                    status_text = f"CRITICAL: {label.upper()} IN ZONE"
                    box_color = (0, 0, 255) # Red
                    violations = [f"RESTRICTED_ZONE_INTRUSION: {label}"]
                    self.dispatch_alert("ZONE_INTRUSION", label, idx + 1, bbox, violations)
                else:
                    status_text = f"SAFE: {label.upper()}"
                    box_color = (0, 255, 0) # Green

                # Draw bounding box and alert tag
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 300, y1), box_color, -1)
                cv2.putText(frame, f"#{idx+1} {status_text}", (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            t_frame_end = time.perf_counter()
            frame_fps = 1.0 / max(0.001, (t_frame_end - t_frame_start))

            # Telemetry HUD Overlay
            cv2.rectangle(frame, (10, 10), (500, 115), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (500, 115), (70, 70, 70), 2)
            cv2.putText(frame, "CraftifAI PipeGen Safety Sentinel", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
            cv2.putText(frame, f"Speed: {frame_fps:.1f} FPS | Objects Tracked: {len(detections)}", (20, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(frame, f"Zone Violations: {frame_violations} (Any Object Encroachment)", (20, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if frame_violations > 0 else (0, 255, 0), 2)

            out.write(frame)

            if show_window:
                cv2.imshow("PipeGen Universal Safety Sentinel", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 30 == 0:
                print(f"  Frame {frame_count:03d} | Speed: {frame_fps:4.1f} FPS | Tracked Objects: {len(detections)} | In Zone: {frame_violations}")

        cap.release()
        out.release()
        if show_window:
            cv2.destroyAllWindows()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()

        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("    UNIVERSAL RESTRICTED ZONE RUN EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Frames Processed : {frame_count}")
        print(f"Elapsed Time           : {elapsed:.2f} seconds")
        print(f"Average Pipeline Speed : {frame_count/elapsed:.1f} FPS")
        print(f"Total Incident Events  : {total_violations} logged to {self.log_file}")
        print(f"Annotated Video Output : {self.output_video}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PipeGen Universal Object Safety Sentinel")
    parser.add_argument("--source", default="data/workplace_sample.mp4", help="Video file or camera index")
    parser.add_argument("--out", default="output_safety_sentinel.mp4", help="Output annotated video path")
    parser.add_argument("--show", action="store_true", help="Display visual window during processing")
    args = parser.parse_args()

    sentinel = UniversalRestrictedZoneSentinel(video_source=args.source, output_video=args.out)
    sentinel.run(show_window=args.show)
