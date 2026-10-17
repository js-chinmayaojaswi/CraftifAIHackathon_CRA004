import os
import cv2
import numpy as np

def generate_sample_workplace_video(output_path="data/workplace_sample.mp4", num_frames=180, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[PREPARATION] Generating realistic synthetic workplace video: {output_path} ({num_frames} frames)...")

    # Hazard zone coordinates
    zone_pts = np.array([[600, 250], [1150, 250], [1150, 650], [600, 650]], np.int32)

    # Simulated workers: (x, y, vx, vy, has_helmet, has_vest)
    workers = [
        {"id": 101, "x": 150, "y": 420, "vx": 4.5, "vy": 0.5, "helmet": True, "vest": True, "name": "Worker-A"},
        {"id": 102, "x": 520, "y": 380, "vx": 3.0, "vy": 0.8, "helmet": False, "vest": True, "name": "Worker-B"},
        {"id": 103, "x": 280, "y": 550, "vx": 1.2, "vy": -0.4, "helmet": True, "vest": False, "name": "Worker-C"}
    ]

    for frame_idx in range(num_frames):
        # Industrial floor background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 45
        
        # Grid lines (factory floor)
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (55, 55, 55), 1)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (55, 55, 55), 1)

        # Machinery Area representation
        cv2.rectangle(frame, (750, 320), (1050, 580), (35, 35, 60), -1)
        cv2.putText(frame, "HEAVY CRANE & PRESS MACHINERY", (760, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 180), 2)

        # Draw Ground Hazard Zone
        cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.putText(frame, "[RESTRICTED HAZARD ZONE: CRANE OPERATIONS]", (610, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Update and draw workers
        for w in workers:
            w["x"] += w["vx"]
            w["y"] += w["vy"]

            # Reverse on bounds
            if w["x"] > width - 120 or w["x"] < 80:
                w["vx"] *= -1
            if w["y"] > height - 150 or w["y"] < 250:
                w["vy"] *= -1

            cx, cy = int(w["x"]), int(w["y"])
            bw, bh = 60, 140
            top_left = (cx - bw//2, cy - bh//2)
            bot_right = (cx + bw//2, cy + bh//2)

            # Draw worker body
            cv2.rectangle(frame, top_left, bot_right, (180, 160, 130), -1)
            # Draw head
            cv2.circle(frame, (cx, cy - bh//2 + 20), 16, (220, 200, 180), -1)
            # Draw helmet if equipped
            if w["helmet"]:
                cv2.ellipse(frame, (cx, cy - bh//2 + 12), (18, 12), 0, 180, 360, (0, 215, 255), -1)
            # Draw safety vest if equipped
            if w["vest"]:
                cv2.rectangle(frame, (cx - bw//2 + 8, cy - bh//2 + 35), (cx + bw//2 - 8, cy + 15), (0, 255, 128), -1)

            cv2.putText(frame, f"{w['name']} (ID:{w['id']})", (top_left[0], top_left[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

        # Timestamp banner
        cv2.putText(frame, f"CAM-SITE-NORTH | 2026-10-17 15:55:{frame_idx//30:02d} | Frame {frame_idx+1}/{num_frames}", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Sample workplace video created at: {output_path}")

if __name__ == "__main__":
    generate_sample_workplace_video()
