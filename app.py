import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import tempfile
import os
import numpy as np
import zipfile
import io

st.set_page_config(page_title="QSS AI", layout="wide")

# Light minimal css
st.markdown("""
<style>
    body {background: #fff; color: #222; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    h1, h2, h3 {color: #111;}
    .sidebar .css-1d391kg {background: #f7f7f7;}
    .object-list {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .object-item {
        border-bottom: 1px solid #eee;
        padding: 0.5rem 0;
        cursor: pointer;
        transition: background 0.15s;
    }
    .object-item:hover {
        background: #eef6fc;
    }
    .object-item.selected {
        background: #c7defa;
        font-weight: 600;
    }
    img {
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        max-width: 100%;
        height: auto;
    }
    .thumbnail {
        width: 64px;
        height: 64px;
        object-fit: contain;
        border: 1px solid #ccc;
        border-radius: 6px;
        margin-right: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .object-row {
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 QSS AI - Document Analysis")

# Load model (using cached)
@st.cache_resource
def load_model():
    base_dir = Path(__file__).parent
    model_path = base_dir / 'runs' / 'detect' / 'signature_stamp_qr_detector' / 'weights' / 'best.pt'
    return YOLO(str(model_path))

model = load_model()

uploaded_files = st.file_uploader(
    "Upload Documents (JPG, PNG, PDF) or ZIP Archive - Max 50 files", 
    type=['jpg', 'jpeg', 'png', 'pdf', 'zip'],
    accept_multiple_files=True
)

if uploaded_files:
    # Limit to 50 files
    if len(uploaded_files) > 50:
        st.error(f"❌ Too many files ({len(uploaded_files)}). Maximum is 50.")
        st.stop()
    
    st.info(f"📁 Processing {len(uploaded_files)} file(s)...")
    
    # Process each file
    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        st.markdown(f"---")
        st.markdown(f"### 📄 {file_idx}. {uploaded_file.name}")
        
        from pdf2image import convert_from_path
        
        # Handle PDF
        if file_ext == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
            
            try:
                # Convert ALL pages
                images = convert_from_path(pdf_path, dpi=120)
                if not images:
                    st.warning("⚠️ Could not convert PDF")
                    os.unlink(pdf_path)
                    continue
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                os.unlink(pdf_path)
                continue
            
            os.unlink(pdf_path)
            
            # Process each page
            for page_num, image in enumerate(images, 1):
                st.markdown(f"#### 📑 Page {page_num}/{len(images)}")
                
                # Process page
                img_resized = image.copy()
                max_width = 400
                if img_resized.width > max_width:
                    ratio = max_width / img_resized.width
                    new_size = (max_width, int(img_resized.height * ratio))
                    img_resized = img_resized.resize(new_size, Image.Resampling.LANCZOS)
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img_file:
                    img_resized.save(tmp_img_file.name)
                    tmp_img_path = tmp_img_file.name
                
                results = model.predict(source=tmp_img_path, conf=0.25, verbose=False)
                result = results[0]
                
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                os.unlink(tmp_img_path)
                
                if len(boxes) == 0:
                    st.info("No objects detected")
                    st.image(img_resized, width=400)
                else:
                    # Stats
                    sigs = sum(1 for c in classes if c == 0)
                    stamps = sum(1 for c in classes if c == 1)
                    qrs = sum(1 for c in classes if c == 2)
                    
                    cols = st.columns([1, 2])
                    
                    with cols[0]:
                        st.metric("Total", len(boxes))
                        st.write(f"✍️ Signatures: {sigs}")
                        st.write(f"🔵 Stamps: {stamps}")
                        st.write(f"📱 QR Codes: {qrs}")
                    
                    with cols[1]:
                        # Draw boxes
                        img_np = np.array(img_resized)
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        
                        for box, cls in zip(boxes, classes):
                            x1, y1, x2, y2 = map(int, box)
                            if cls == 0:
                                color = (102, 126, 234)
                            elif cls == 1:
                                color = (240, 147, 251)
                            else:
                                color = (79, 172, 254)
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                        
                        img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        st.image(img_final, width=400)
                
                if page_num < len(images):
                    st.markdown("---")
            
            continue
        
        # Handle images
        elif file_ext in ['jpg', 'jpeg', 'png']:
            image = Image.open(uploaded_file).convert("RGB")
            
            img_resized = image.copy()
            max_width = 400
            if img_resized.width > max_width:
                ratio = max_width / img_resized.width
                new_size = (max_width, int(img_resized.height * ratio))
                img_resized = img_resized.resize(new_size, Image.Resampling.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img_file:
                img_resized.save(tmp_img_file.name)
                tmp_img_path = tmp_img_file.name
            
            results = model.predict(source=tmp_img_path, conf=0.25, verbose=False)
            result = results[0]
            
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            os.unlink(tmp_img_path)
            
            if len(boxes) == 0:
                st.info("No objects detected")
                st.image(img_resized, width=400)
            else:
                # Stats
                sigs = sum(1 for c in classes if c == 0)
                stamps = sum(1 for c in classes if c == 1)
                qrs = sum(1 for c in classes if c == 2)
                
                cols = st.columns([1, 2])
                
                with cols[0]:
                    st.metric("Total", len(boxes))
                    st.write(f"✍️ Signatures: {sigs}")
                    st.write(f"🔵 Stamps: {stamps}")
                    st.write(f"📱 QR Codes: {qrs}")
                
                with cols[1]:
                    # Draw boxes
                    img_np = np.array(img_resized)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    for box, cls in zip(boxes, classes):
                        x1, y1, x2, y2 = map(int, box)
                        if cls == 0:
                            color = (102, 126, 234)
                        elif cls == 1:
                            color = (240, 147, 251)
                        else:
                            color = (79, 172, 254)
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    
                    img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    st.image(img_final, width=400)
        
        # Skip unsupported
        else:
            st.warning(f"⚠️ Unsupported format: {file_ext}")
            continue
else:
    st.info("👆 Upload documents (JPG, PNG, PDF) - You can select multiple files at once (max 50)")

