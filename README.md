# CraftifAI PipeGen — Workplace Safety Sentinel
### Track A: Edge Vision Perception Pipeline (TensorRT Accelerated)
**Embedded SF: CraftifAI Physical AI Hackathon**

---

## 🎯 Executive Overview
**Workplace Safety Sentinel** is an industrial edge Physical AI perception pipeline built following the complete **5-stage CraftifAI PipeGen methodology**. It addresses critical workplace safety in industrial, construction, and manufacturing facilities where personnel entering dangerous machinery zones or failing to wear required Personal Protective Equipment (PPE) causes severe preventable injuries.

- **Track Selected:** Track A — PipeGen (Perception) [100% Software-executable]
- **Target Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, 6GB VRAM, Tensor Cores)
- **Acceleration Framework:** DeepStream / TensorRT (FP16 Engine)
- **Actionable Outputs:** Real-time MQTT telemetry broadcast + structured JSON incident audit log + on-screen visual warning HUD

---

## 🔄 Five Stages of PipeGen Implementation

| Stage | PipeGen Phase | Implementation File | Status / Deliverable |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Requirement** | `01_requirement.json` | Natural language and structured specification defining hazard zones, model targets, camera inputs, and alert topics. |
| **Stage 2** | **Pipeline IR** | `02_pipeline_ir.json` / `generate_pipeline_ir_graph.py` | Directed node graph mapping Source $\rightarrow$ Preprocessor $\rightarrow$ Inference $\rightarrow$ Tracker $\rightarrow$ Analytics $\rightarrow$ Display/MQTT/JSON Sinks. Renders `pipeline_ir_graph.png`. |
| **Stage 3** | **Model Selection** | `03_export_model.py` | YOLOv8 Safety Perception model exported to ONNX format (`models/yolov8n_safety.onnx`) with static 640x640 shape. |
| **Stage 4** | **Compiler Benchmark** | `04_compile_tensorrt_benchmark.py` | Profiling TensorRT FP16 acceleration vs CUDA FP32 and CPU. Outputs `tensorrt_speed_report.json` and `tensorrt_speed_report.png`. |
| **Stage 5** | **Live Action Run** | `05_live_pipeline_run.py` | Live ingestion of video/webcam, multi-worker tracking, hazard geofencing, PPE verification, live MQTT broadcast, and `incident_log.jsonl` recording. |

---

## 📊 Benchmark & Speed Evidence

```text
Host CPU Baseline (FP32)       :  30.74 ms |  32.5 FPS (1.0x Baseline)
NVIDIA CUDA Core (FP32)         :  13.36 ms |  74.9 FPS (2.3x Speedup)
TensorRT FP16 Engine (Ada)      :   6.54 ms | 152.9 FPS (4.7x Speedup)
```

- **Speedup Factor:** **4.7x faster** with TensorRT FP16 over CPU.
- **Inference Latency:** **6.54 ms per frame** on NVIDIA RTX 4050 GPU.
- **Evidence Files:**
  - `pipeline_ir_graph.png` (Node graph topology)
  - `tensorrt_speed_report.png` (Speed and throughput comparison chart)
  - `output_safety_sentinel_frame.png` (Visual HUD and violation overlay)

---

## 🚀 How to Run

### Windows (One-Click):
Double-click `run.bat` or run:
```powershell
.\.venv\Scripts\python.exe 05_live_pipeline_run.py
```

### Run with Live Webcam:
```powershell
.\.venv\Scripts\python.exe 05_live_pipeline_run.py --source 0 --show
```

### Linux / DeepStream:
```bash
chmod +x run.sh
./run.sh
```

---

## 📡 Actionable Output Formats

### 1. Real-time MQTT Broadcast
- **Broker:** `broker.hivemq.com:1883`
- **Topic:** `craftifai/pipegen/workplace_safety/alerts`
- **Payload Schema:**
```json
{
  "timestamp": "2026-10-17T15:58:34.418Z",
  "pipeline": "CraftifAI_PipeGen_Workplace_Sentinel_v0.5.1",
  "camera_id": "CAM_SITE_NORTH",
  "event_type": "ZONE_INTRUSION",
  "severity": "CRITICAL",
  "target_id": 3,
  "bounding_box": [496, 311, 557, 452],
  "violations": ["RESTRICTED_ZONE_INTRUSION", "MISSING_HARDHAT"],
  "action": "BROADCAST_TRIGGERED"
}
```

### 2. Audit Trail Log (`incident_log.jsonl`)
Appended in real-time on every violation for compliance audits, shift supervisors, and automated sirens/relays.
