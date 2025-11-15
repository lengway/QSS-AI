#!/usr/bin/env python3
"""
Подготовка датасета с нуля: 3 класса (signature, stamp, qr) + 400 DPI
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from explore_data import convert_pdfs_to_images, analyze_annotations
from utils import parse_annotations, create_yolo_labels, split_dataset


# УКАЖИ ПУТЬ К POPPLER!
POPPLER_PATH = r'C:\poppler\Library\bin'


def main():
    base_dir = Path(__file__).parent
    docs_dir = base_dir / 'docs'
    data_dir = base_dir / 'data'
    
    # 3 КЛАССА!
    CLASS_MAPPING = {
        'signature': 0,
        'stamp': 1,
        'qr': 2
    }
    
    print("=" * 60)
    print("🚀 ПОДГОТОВКА ДАТАСЕТА (3 класса + 400 DPI)")
    print("=" * 60)
    
    # Проверка Poppler
    poppler_check = Path(POPPLER_PATH)
    if not poppler_check.exists():
        print(f"\n❌ Poppler не найден: {POPPLER_PATH}")
        print("Укажи правильный путь в строке 15 файла prepare_data.py")
        return
    
    print(f"\n✅ Poppler: {POPPLER_PATH}\n")
    
    # Шаг 1: Анализ
    print("📊 STEP 1/4: Analyzing annotations...")
    annotations_path = docs_dir / 'selected_annotations.json'
    analyze_annotations(annotations_path)
    
    # Шаг 2: Конвертация PDF → images (400 DPI)
    print("\n\n🖼️  STEP 2/4: Converting PDFs to images (400 DPI)...")
    print("⚠️  Это займет больше времени, но качество будет лучше!\n")
    
    pdf_dir = docs_dir / 'pdfs'
    images_temp_dir = data_dir / 'images_temp'
    
    convert_pdfs_to_images(pdf_dir, images_temp_dir, dpi=400, poppler_bin_path=POPPLER_PATH)
    
    # Шаг 3: Генерация YOLO labels
    print("\n\n🏷️  STEP 3/4: Generating YOLO labels (3 classes)...")
    annotations = parse_annotations(str(annotations_path))
    labels_temp_dir = data_dir / 'labels_temp'
    create_yolo_labels(annotations, labels_temp_dir, CLASS_MAPPING)
    
    # Шаг 4: Разделение на train/val/test
    print("\n\n📂 STEP 4/4: Splitting dataset...")
    split_dataset(
        images_dir=images_temp_dir,
        labels_dir=labels_temp_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    # Создание data.yaml
    print("\n\n⚙️  Generating data.yaml...")
    config_dir = base_dir / 'config'
    config_dir.mkdir(exist_ok=True)
    
    data_yaml_content = f"""# YOLOv8 Dataset Configuration
# 3 classes: signature, stamp, qr
# Resolution: 400 DPI

path: {data_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: signature
  1: stamp
  2: qr
"""
    
    data_yaml_path = config_dir / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"✅ Created: {data_yaml_path}")
    
    # Резюме
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print("\n📦 Датасет подготовлен:")
    print("  • Классы: 3 (signature, stamp, qr)")
    print("  • Разрешение: 400 DPI")
    print("  • Аннотаций: 258 (103 подписи + 60 печатей + 95 QR)")
    print("\n🚀 Следующий шаг - обучение:")
    print("   python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
