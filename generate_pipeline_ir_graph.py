import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline_ir_graph(ir_file="02_pipeline_ir.json", output_image="pipeline_ir_graph.png"):
    with open(ir_file, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")

    # Title & Header
    ax.text(0.5, 0.94, "CraftifAI PipeGen — Pipeline Intermediate Representation (IR)", 
            color="#38BDF8", fontsize=18, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.90, f"Pipeline: {data.get('pipeline_name', 'Workplace Safety Sentinel')} | Target: NVIDIA RTX 4050 GPU (DeepStream / TensorRT)", 
            color="#94A3B8", fontsize=12, ha="center", transform=ax.transAxes)

    node_coords = {
        "node_01_source": (0.10, 0.50, "#10B981", "Source Node\n(Video / RTSP / USB Cam)\nnvv4l2decoder"),
        "node_02_preprocessor": (0.28, 0.50, "#6366F1", "Preprocessor Node\n(NVMM Rescale 640x640)\nnvvideoconvert"),
        "node_03_inference": (0.47, 0.50, "#EC4899", "Inference Node\n(TensorRT Engine FP16)\nnvinfer / YOLOv8"),
        "node_04_tracker": (0.65, 0.50, "#F59E0B", "Tracker Node\n(NvDCF / ByteTrack)\nnvtracker"),
        "node_05_analytics": (0.82, 0.50, "#8B5CF6", "Analytics Node\n(Hazard Zone & PPE)\nnvdsanalytics"),
        "node_06_osd_sink": (0.95, 0.72, "#06B6D4", "Sink: Display\n(Live Annotations)\nnvdsosd"),
        "node_07_mqtt_sink": (0.95, 0.50, "#EF4444", "Sink: MQTT Alert\n(Realtime Telemetry)\nnvmsgbroker"),
        "node_08_json_sink": (0.95, 0.28, "#14B8A6", "Sink: JSON Audit\n(incident_log.jsonl)\nfile_sink"),
    }

    # Draw Connections
    connections = [
        ("node_01_source", "node_02_preprocessor"),
        ("node_02_preprocessor", "node_03_inference"),
        ("node_03_inference", "node_04_tracker"),
        ("node_04_tracker", "node_05_analytics"),
        ("node_05_analytics", "node_06_osd_sink"),
        ("node_05_analytics", "node_07_mqtt_sink"),
        ("node_05_analytics", "node_08_json_sink")
    ]

    for start_id, end_id in connections:
        x1, y1, _, _ = node_coords[start_id]
        x2, y2, _, _ = node_coords[end_id]
        
        ax.annotate(
            "", xy=(x2 - 0.045, y2), xytext=(x1 + 0.045, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                            color="#38BDF8", lw=2.5, mutation_scale=15)
        )

    # Draw Nodes
    for node_id, (x, y, color, label) in node_coords.items():
        w, h = 0.09, 0.14
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                      boxstyle="round,pad=0.015,rounding_size=0.02",
                                      facecolor="#1E293B", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, y, label, color="white", fontsize=9.5, fontweight="semibold",
                ha="center", va="center", multialignment="center")

    # Legend / Stage indicator at bottom
    stages = [
        ("Stage 1: Requirement", "#38BDF8"),
        ("Stage 2: Pipeline IR", "#818CF8"),
        ("Stage 3: Model Config", "#EC4899"),
        ("Stage 4: TensorRT Compiler", "#F59E0B"),
        ("Stage 5: Live Action Run", "#10B981")
    ]
    for i, (stage_name, col) in enumerate(stages):
        ax.text(0.12 + i * 0.19, 0.08, f"● {stage_name}", color=col, fontsize=11, fontweight="bold",
                ha="center", transform=ax.transAxes)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_image, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Pipeline IR Node Graph successfully rendered to: {output_image}")

if __name__ == "__main__":
    draw_pipeline_ir_graph()
