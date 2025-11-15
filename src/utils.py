import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import random


def convert_bbox_to_yolo(bbox: Dict, page_size: Dict, class_id: int) -> str:
    """
    Convert JSON bbox to YOLO format.
    
    Args:
        bbox: {'x': x, 'y': y, 'width': w, 'height': h}
        page_size: {'width': w, 'height': h}
        class_id: 0 for signature, 1 for stamp
    
    Returns:
        YOLO format string: 'class_id x_center y_center width height'
    """
    x = bbox['x']
    y = bbox['y']
    w = bbox['width']
    h = bbox['height']
    
    page_w = page_size['width']
    page_h = page_size['height']
    
    # Calculate center and normalize
    x_center = (x + w / 2) / page_w
    y_center = (y + h / 2) / page_h
    norm_width = w / page_w
    norm_height = h / page_h
    
    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def parse_annotations(json_path: str) -> Dict:
    """
    Parse annotations JSON file.
    
    Returns:
        Dict with structure: {pdf_name: {page_num: {annotations, page_size}}}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_yolo_labels(annotations: Dict, output_dir: Path, class_mapping: Dict[str, int]):
    """
    Create YOLO format label files from annotations.
    
    Args:
        annotations: Parsed annotation data
        output_dir: Directory to save label files
        class_mapping: {'signature': 0, 'stamp': 1}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_count = 0
    
    for pdf_name, pages in annotations.items():
        # Clean PDF name for file naming
        base_name = pdf_name.replace('.pdf', '')
        
        for page_key, page_data in pages.items():
            page_num = page_key.replace('page_', '')
            
            # Create label filename
            label_filename = f"{base_name}_page_{page_num}.txt"
            label_path = output_dir / label_filename
            
            page_size = page_data['page_size']
            annotations_list = page_data['annotations']
            
            yolo_lines = []
            
            for ann_dict in annotations_list:
                # Each annotation is wrapped in another dict
                for ann_id, ann_data in ann_dict.items():
                    category = ann_data['category']
                    bbox = ann_data['bbox']
                    
                    if category in class_mapping:
                        class_id = class_mapping[category]
                        yolo_line = convert_bbox_to_yolo(bbox, page_size, class_id)
                        yolo_lines.append(yolo_line)
                        label_count += 1
            
            # Write label file
            if yolo_lines:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
    
    print(f"✅ Created {len(list(output_dir.glob('*.txt')))} label files with {label_count} annotations")


def split_dataset(images_dir: Path, labels_dir: Path, 
                  train_ratio: float = 0.7, val_ratio: float = 0.2, 
                  test_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset into train/val/test sets.
    
    Args:
        images_dir: Directory containing all images
        labels_dir: Directory containing all labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        print("⚠️  No images found!")
        return
    
    print(f"📊 Total images found: {len(image_files)}")
    
    # Shuffle
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"📂 Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create split directories
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        # Create directories
        img_split_dir = images_dir.parent / 'images' / split_name
        lbl_split_dir = labels_dir.parent / 'labels' / split_name
        
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_file in files:
            # Copy image
            shutil.copy(img_file, img_split_dir / img_file.name)
            
            # Copy corresponding label
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, lbl_split_dir / label_file.name)
        
        print(f"  ✅ {split_name}: {len(files)} files copied")


if __name__ == "__main__":
    # Configuration - ТЕПЕРЬ 3 КЛАССА!
    CLASS_MAPPING = {
        'signature': 0,
        'stamp': 1,
        'qr': 2  # QR-коды добавлены!
    }
    
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / 'docs'
    
    print("🚀 Starting YOLO label generation...\n")
    
    # Parse annotations
    annotations_path = docs_dir / 'selected_annotations.json'
    print(f"📖 Reading annotations from: {annotations_path}")
    annotations = parse_annotations(annotations_path)
    
    print(f"📊 Found {len(annotations)} PDFs with annotations\n")
    
    # Create labels
    temp_labels_dir = base_dir / 'data' / 'labels_temp'
    create_yolo_labels(annotations, temp_labels_dir, CLASS_MAPPING)
    
    print("\n✅ Label generation complete!")
    print(f"\n💡 Next step: Convert PDFs to images, then run split_dataset()")
