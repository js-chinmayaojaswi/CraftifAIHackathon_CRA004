#!/usr/bin/env bash
# CraftifAI PipeGen - Workplace Safety Sentinel (Track A)
# Deployment runner for Linux DeepStream / Jetson / x86_64

set -e

echo "============================================================"
echo "  CraftifAI PipeGen - Workplace Safety Sentinel (Track A)"
echo "  Target: NVIDIA GPU / DeepStream TensorRT Engine"
echo "============================================================"

python3 generate_pipeline_ir_graph.py
python3 04_compile_tensorrt_benchmark.py
python3 05_live_pipeline_run.py "$@"
