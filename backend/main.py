from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import tempfile
import io
import base64
from typing import List
from pdf2image import convert_from_path

app = FastAPI(title="QSS AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model - Patch ultralytics for PyTorch 2.6
import torch
import os
import sys

# Monkey-patch torch.load before importing YOLO
original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False  # Force disable for our trusted model
    return original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

base_dir = Path(__file__).parent.parent
model_path = base_dir / 'runs' / 'detect' / 'signature_stamp_qr_detector' / 'weights' / 'best.pt'

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}. Train model first!")

model = YOLO(str(model_path))

@app.get("/")
def read_root():
    return {"status": "QSS AI API is running"}

@app.post("/api/detect")
async def detect_objects(files: List[UploadFile] = File(...)):
    """Detect signatures, stamps, and QR codes in uploaded files"""
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed")
    
    results = []
    
    for file in files:
        try:
            file_ext = file.filename.split('.')[-1].lower()
            
            # Handle PDF
            if file_ext == 'pdf':
                content = await file.read()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(content)
                    pdf_path = tmp.name
                
                try:
                    images = convert_from_path(pdf_path, dpi=120)
                except:
                    Path(pdf_path).unlink()
                    continue
                
                Path(pdf_path).unlink()
                
                # Process each page
                for page_num, image in enumerate(images, 1):
                    detection_result = process_image(image, f"{file.filename} - Page {page_num}")
                    if detection_result:
                        results.append(detection_result)
            
            # Handle images
            elif file_ext in ['jpg', 'jpeg', 'png']:
                content = await file.read()
                image = Image.open(io.BytesIO(content)).convert('RGB')
                detection_result = process_image(image, file.filename)
                if detection_result:
                    results.append(detection_result)
        
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            continue
    
    return JSONResponse(content={"results": results})

def process_image(image: Image.Image, filename: str):
    """Process single image and return detection results"""
    
    try:
        # Resize
        max_width = 800
        if image.width > max_width:
            ratio = max_width / image.width
            new_size = (max_width, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, 'JPEG')
            tmp_path = tmp.name
        
        # Predict
        prediction = model.predict(source=tmp_path, conf=0.25, verbose=False)
        result = prediction[0]
        
        Path(tmp_path).unlink()
        
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        # Parse detections
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        # Draw boxes on image
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        detections = []
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            if cls == 0:
                label = "Signature"
                color = (102, 126, 234)  # Blue
            elif cls == 1:
                label = "Stamp"
                color = (240, 147, 251)  # Pink
            else:
                label = "QR Code"
                color = (79, 172, 254)  # Cyan
            
            # Draw rectangle
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"{label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(img_cv, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })
        
        # Convert to base64
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"Generated base64 image for {filename}, length: {len(img_base64)}")
        
        # Count by type
        signatures = sum(1 for d in detections if d['label'] == 'Signature')
        stamps = sum(1 for d in detections if d['label'] == 'Stamp')
        qr_codes = sum(1 for d in detections if d['label'] == 'QR Code')
        
        return {
            "filename": filename,
            "total": len(detections),
            "signatures": signatures,
            "stamps": stamps,
            "qr_codes": qr_codes,
            "detections": detections,
            "image_size": {"width": image.width, "height": image.height},
            "image_base64": img_base64
        }
    except Exception as e:
        print(f"Error processing image {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
