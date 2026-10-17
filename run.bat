@echo off
echo ============================================================
echo   CraftifAI PipeGen - Workplace Safety Sentinel (Track A)
echo   Target: NVIDIA RTX 4050 Laptop GPU (TensorRT Acceleration)
echo ============================================================
echo.
echo Running full 5-stage PipeGen perception pipeline...
echo.

echo [1/4] Rendering Pipeline IR Graph...
.\.venv\Scripts\python.exe generate_pipeline_ir_graph.py

echo [2/4] Running TensorRT Benchmark & Speed Profiler...
.\.venv\Scripts\python.exe 04_compile_tensorrt_benchmark.py

echo [3/4] Preparing Test Video...
.\.venv\Scripts\python.exe create_test_video.py

echo [4/4] Launching Live Perception Sentinel Pipeline...
.\.venv\Scripts\python.exe 05_live_pipeline_run.py %*

echo.
echo ============================================================
echo   Pipeline Execution Completed Successfully!
echo   Evidence Images: pipeline_ir_graph.png, tensorrt_speed_report.png
echo   Incident Log   : incident_log.jsonl
echo   Output Video   : output_safety_sentinel.mp4
echo ============================================================
pause
