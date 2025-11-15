import json
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
import os


from typing import Optional


def convert_pdfs_to_images(pdf_dir: Path, output_dir: Path, dpi: int = 200, poppler_bin_path: Optional[str] = None):
    """
    Convert all PDFs to images.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save images
        dpi: Resolution for conversion (200 is good balance)
        poppler_bin_path: Path to poppler bin folder (e.g., 'C:/poppler/Library/bin')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find poppler
    base_dir = Path(__file__).parent.parent
    poppler_path = poppler_bin_path
    
    if not poppler_path:
        # Check for local poppler in project
        local_poppler = base_dir / 'poppler' / 'poppler-24.08.0' / 'Library' / 'bin'
        if local_poppler.exists():
            poppler_path = str(local_poppler)
            print(f"✅ Using local Poppler: {poppler_path}")
        else:
            local_poppler2 = base_dir / 'poppler' / 'Library' / 'bin'
            if local_poppler2.exists():
                poppler_path = str(local_poppler2)
                print(f"✅ Using local Poppler: {poppler_path}")
    
    # Common Poppler installation paths on Windows
    if not poppler_path:
        possible_paths = [
            r'C:\poppler\Library\bin',
            r'C:\poppler-24.08.0\Library\bin',
            r'C:\Program Files\poppler\Library\bin',
            r'C:\poppler\bin',
        ]
        for path in possible_paths:
            if Path(path).exists():
                poppler_path = path
                print(f"✅ Found Poppler at: {poppler_path}")
                break
    
    if poppler_path:
        print(f"📍 Using Poppler from: {poppler_path}")
    
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    if not pdf_files:
        print("⚠️  No PDF files found!")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    print(f"🔄 Converting to images (DPI={dpi})...\n")
    
    total_images = 0
    errors = 0
    
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            # Convert PDF to images
            if poppler_path:
                images = convert_from_path(
                    pdf_file, 
                    dpi=dpi,
                    fmt='jpeg',
                    poppler_path=poppler_path
                )
            else:
                images = convert_from_path(
                    pdf_file, 
                    dpi=dpi,
                    fmt='jpeg'
                )
            
            # Save each page
            base_name = pdf_file.stem
            
            for page_num, image in enumerate(images, start=1):
                # Match the naming convention used in annotations
                image_filename = f"{base_name}_page_{page_num}.jpg"
                image_path = output_dir / image_filename
                
                image.save(image_path, 'JPEG', quality=95)
                total_images += 1
        
        except Exception as e:
            if errors == 0:  # Show error message only once
                print(f"\n❌ Error converting PDFs: {e}")
                print("\n🔧 FIX: Install Poppler first!")
                print("   Run: python install_poppler.py")
                print("   Or manually install from: https://github.com/oschwartz10612/poppler-windows/releases\n")
            errors += 1
            continue
    
    if errors > 0:
        print(f"\n⚠️  Failed to convert {errors} PDFs (Poppler not installed)")
        print(f"\n🔧 Quick fix:")
        print(f"   1. Run: python install_poppler.py")
        print(f"   2. Then run this script again")
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Total images created: {total_images}")
    print(f"📁 Saved to: {output_dir}")


def analyze_annotations(annotations_path: Path):
    """
    Analyze annotation statistics.
    """
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_pdfs = len(data)
    total_pages = 0
    total_signatures = 0
    total_stamps = 0
    
    for pdf_name, pages in data.items():
        total_pages += len(pages)
        
        for page_key, page_data in pages.items():
            annotations_list = page_data['annotations']
            
            for ann_dict in annotations_list:
                for ann_id, ann_data in ann_dict.items():
                    category = ann_data['category']
                    if category == 'signature':
                        total_signatures += 1
                    elif category == 'stamp':
                        total_stamps += 1
    
    print("\n📊 DATASET STATISTICS")
    print("=" * 50)
    print(f"📄 Total PDFs: {total_pdfs}")
    print(f"📑 Total Pages: {total_pages}")
    print(f"✍️  Total Signatures: {total_signatures}")
    print(f"🔵 Total Stamps: {total_stamps}")
    print(f"📦 Total Annotations: {total_signatures + total_stamps}")
    print(f"📈 Average annotations per page: {(total_signatures + total_stamps) / total_pages:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / 'docs'
    
    print("🚀 PDF to Image Conversion Tool\n")
    
    # Check for Poppler
    poppler_check = base_dir / 'poppler'
    if not poppler_check.exists():
        print("⚠️  Poppler not found!")
        print("\n🔧 Install Poppler first:")
        print("   python install_poppler.py")
        print("\nOr download manually from:")
        print("   https://github.com/oschwartz10612/poppler-windows/releases\n")
    
    # Analyze annotations first
    annotations_path = docs_dir / 'selected_annotations.json'
    analyze_annotations(annotations_path)
    
    # Convert PDFs
    pdf_dir = docs_dir / 'pdfs'
    output_dir = base_dir / 'data' / 'images_temp'
    
    print(f"\n🔄 Starting conversion...")
    print(f"📂 Input: {pdf_dir}")
    print(f"📂 Output: {output_dir}\n")
    
    convert_pdfs_to_images(pdf_dir, output_dir, dpi=200)
    
    print("\n💡 Next steps:")
    print("  1. Run: python src/utils.py (to generate YOLO labels)")
    print("  2. Split dataset into train/val/test")
    print("  3. Configure data.yaml")
    print("  4. Start training!")
