from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2


def predict_on_image(model_path: str, image_path: str, conf: float = 0.25, save: bool = True):
    """
    Run prediction on a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
        conf: Confidence threshold
        save: Whether to save results
    """
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save,
        save_txt=False,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    # Display results
    for r in results:
        print(f"\n🖼️  Image: {r.path}")
        print(f"  Image size: {r.orig_shape}")
        print(f"  Detections: {len(r.boxes)}")
        
        if len(r.boxes) > 0:
            print("\n  Detected objects:")
            for i, box in enumerate(r.boxes):
                cls = int(box.cls)
                conf = float(box.conf)
                name = r.names[cls]
                coords = box.xyxy[0].tolist()
                
                print(f"    {i+1}. {name}: {conf:.2%} confidence")
                print(f"       Bbox: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]")
        else:
            print("  ⚠️  No objects detected")
    
    return results


def predict_on_directory(model_path: str, dir_path: str, conf: float = 0.25):
    """
    Run prediction on all images in a directory.
    
    Args:
        model_path: Path to trained model
        dir_path: Path to directory containing images
        conf: Confidence threshold
    """
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(
        source=dir_path,
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    # Summary
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n📊 Summary:")
    print(f"  Total images: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections / len(results):.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Signature & Stamp Detection')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: best.pt from latest run)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Find model path
    if args.model:
        model_path = args.model
    else:
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / 'runs' / 'detect' / 'signature_stamp_qr_detector' / 'weights' / 'best.pt'
        
        if not model_path.exists():
            print("❌ Error: No trained model found!")
            print("   Train a model first: python src/train.py")
            print("   Or specify model path: --model path/to/model.pt")
            return
    
    print("=" * 60)
    print("🔍 YOLOV8 PREDICTION - SIGNATURE & STAMP DETECTION")
    print("=" * 60)
    print(f"\n🧠 Model: {model_path}")
    print(f"📷 Source: {args.source}")
    print(f"🎯 Confidence threshold: {args.conf}")
    
    # Check if source is file or directory
    source_path = Path(args.source)
    
    if source_path.is_file():
        print("\n🔍 Running prediction on single image...\n")
        predict_on_image(str(model_path), str(source_path), args.conf)
    elif source_path.is_dir():
        print("\n🔍 Running prediction on directory...\n")
        predict_on_directory(str(model_path), str(source_path), args.conf)
    else:
        print(f"\n❌ Error: Source not found: {args.source}")
        return
    
    print("\n" + "=" * 60)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 60)
    print("\n💾 Results saved to: runs/detect/predict/")


if __name__ == "__main__":
    main()
