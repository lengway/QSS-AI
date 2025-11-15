from ultralytics import YOLO
from pathlib import Path
import torch


def train_model():
    """
    Train YOLOv8 model for signature and stamp detection.
    """
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'config' / 'data.yaml'
    
    print("=" * 60)
    print("🎯 YOLOV8 TRAINING - SIGNATURE & STAMP DETECTION")
    print("=" * 60)
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Device: {device.upper()}")
    if device == 'cpu':
        print("⚠️  Warning: Training on CPU will be slow. Consider using GPU.")
    
    # Load model
    print("\n📦 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # YOLOv8 nano for fast trainin
    
    # Training parameters
    print("\n⚙️  Training Configuration:")
    print(f"  - Dataset: {config_path}")
    print("  - Model: YOLOv8n (nano)")
    print("  - Epochs: 100")
    print("  - Image size: 640")
    print("  - Batch size: 32 (GPU optimized)")
    print("  - Workers: 8")
    print("  - Classes: 3 (signature, stamp, qr)")
    
    # Start training
    print("\n🚀 Starting training...\n")
    
    results = model.train(
        data=str(config_path),
        epochs=100,
        imgsz=640,
        batch=32,  # Увеличен для GPU
        name='signature_stamp_qr_detector',  # Новое название
        device=device,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,  # Generate plots
        
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10.0,  # Rotation augmentation
        translate=0.1, # Translation augmentation
        scale=0.5,     # Scale augmentation
        shear=0.0,     # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,    # Vertical flip probability
        fliplr=0.5,    # Horizontal flip probability
        mosaic=1.0,    # Mosaic augmentation probability
        mixup=0.0,     # Mixup augmentation probability
        
        # Optimizer
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Other
        box=7.5,
        cls=0.5,
        workers=8,  # Увеличено для быстрой загрузки данных
        project=str(base_dir / 'runs' / 'detect'),
        exist_ok=True,
        verbose=True,
    )
    
    # Training complete
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    # Show results
    print("\n📊 Training Results:")
    print(f"  Best model: {base_dir / 'runs' / 'detect' / 'signature_stamp_qr_detector' / 'weights' / 'best.pt'}")
    print(f"  Last model: {base_dir / 'runs' / 'detect' / 'signature_stamp_qr_detector' / 'weights' / 'last.pt'}")
    
    # Validate on best model
    print("\n📊 Running validation on best model...")
    best_model = YOLO(base_dir / 'runs' / 'detect' / 'signature_stamp_qr_detector' / 'weights' / 'best.pt')
    metrics = best_model.val()
    
    print("\n🎯 Final Metrics:")
    print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    
    print("\n💡 Next steps:")
    print("  1. Check training plots in runs/detect/signature_stamp_qr_detector/")
    print("  2. Run: python src/predict.py --source <path_to_image>")
    print("  3. Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
