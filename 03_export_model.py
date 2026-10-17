import os
from ultralytics import YOLO

def export_model():
    print("[STAGE 3] Loading & Exporting YOLOv8 Safety Model to ONNX...")
    os.makedirs("models", exist_ok=True)
    
    # Load YOLOv8n base model (contains person class 0 and general objects)
    model = YOLO("yolov8n.pt")
    
    # Export to ONNX format with 640x640 shape and FP16 half precision
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        half=False, # standard ONNX for universal TRT compilation
        dynamic=False,
        simplify=True
    )
    
    # Move / rename to models folder
    target_onnx = os.path.join("models", "yolov8n_safety.onnx")
    if os.path.exists("yolov8n.onnx"):
        if os.path.exists(target_onnx):
            os.remove(target_onnx)
        os.rename("yolov8n.onnx", target_onnx)
        print(f"[SUCCESS] ONNX Model exported to: {target_onnx}")
    elif os.path.exists(onnx_path):
        import shutil
        shutil.copy(onnx_path, target_onnx)
        print(f"[SUCCESS] ONNX Model saved to: {target_onnx}")

if __name__ == "__main__":
    export_model()
