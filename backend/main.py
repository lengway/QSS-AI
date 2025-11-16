from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import tempfile
import io
import base64
from typing import List, Tuple, Dict
from pdf2image import convert_from_path
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования с поддержкой UTF-8

# Настройка файлового handler с UTF-8
file_handler = RotatingFileHandler(
    'app.log', 
    maxBytes=10*1024*1024, 
    backupCount=5,
    encoding='utf-8'
)

# Обычный консольный handler - пусть система сама разбирается с кодировкой
console_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Ray - опциональная зависимость (не поддерживает Python 3.13)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Install with: pip install 'ray[default]>=2.8.0' (requires Python <= 3.12)")

app = FastAPI(title="QSS AI API", version="1.0.0")

# CORS - настройка через переменные окружения
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model - Patch ultralytics for PyTorch 2.6
import torch

# Monkey-patch torch.load before importing YOLO (workaround for PyTorch 2.6)
original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False  # Force disable for our trusted model
    return original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

# Конфигурация через переменные окружения
base_dir = Path(__file__).parent.parent
model_path_env = os.getenv("MODEL_PATH")
if model_path_env:
    model_path = Path(model_path_env)
else:
    model_path = base_dir / 'runs' / 'detect' / 'newdataset_from_scratch' / 'weights' / 'best.pt'

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}. Train model first or set MODEL_PATH env variable!")

logger.info(f"Loading model from: {model_path}")
model = YOLO(str(model_path))
logger.info("Model loaded successfully")

# Инициализация Ray для распараллеливания (только если доступен)
USE_RAY = os.getenv("USE_RAY", "true").lower() == "true" and RAY_AVAILABLE
RAY_NUM_CPUS = int(os.getenv("RAY_NUM_CPUS", 0))  # 0 = использовать все доступные CPU

if USE_RAY and RAY_AVAILABLE:
    try:
        if not ray.is_initialized():
            # Отключаем usage stats через переменную окружения
            os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
            
            # Упрощенная инициализация Ray - убираем проблемные параметры
            ray.init(
                num_cpus=RAY_NUM_CPUS if RAY_NUM_CPUS > 0 else None,
                ignore_reinit_error=True,
                logging_level=logging.ERROR,  # Только ошибки
                _system_config={
                    "metrics_report_interval_ms": 0,
                    "enable_metrics_collection": False
                },
                include_dashboard=False  # Отключаем dashboard
            )
        logger.info(f"Ray initialized successfully (CPUs: {ray.available_resources().get('CPU', 'all')})")
    except Exception as e:
        logger.warning(f"Ray initialization failed: {e}. Falling back to sequential processing.")
        USE_RAY = False
elif not RAY_AVAILABLE:
    logger.info("Ray not available (requires Python <= 3.12). Using sequential processing.")
    USE_RAY = False
else:
    logger.info("Ray disabled, using sequential processing")

# Ray Actor для модели YOLO (модель загружается один раз на worker)
if RAY_AVAILABLE:
    @ray.remote
    class YOLOModelActor:
        def __init__(self, model_path: str):
            """Инициализация модели на Ray worker"""
            import torch
            # Применяем тот же патч для torch.load
            original_torch_load = torch.load
            def patched_torch_load(f, *args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(f, *args, **kwargs)
            torch.load = patched_torch_load
            
            from ultralytics import YOLO
            # Загружаем модель и явно указываем device
            # Используем GPU если доступен, иначе CPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded on Ray worker: {model_path} (device: {self.device})")
        
        def predict(self, image_array: np.ndarray, conf_threshold: float) -> dict:
            """Предсказание на изображении (numpy array) - оптимизировано без временных файлов"""
            # Передаем numpy array напрямую в модель (быстрее чем через файл)
            # Явно указываем device и half precision для GPU
            prediction = self.model.predict(
                source=image_array,
                conf=conf_threshold,
                verbose=False,
                device=self.device,
                half=(self.device == 'cuda'),  # FP16 для GPU - ускоряет в 2 раза
                imgsz=640,  # Фиксированный размер для ускорения
                iou=0.45,  # IoU threshold для NMS (по умолчанию 0.7, снижаем для ускорения)
                max_det=300,  # Максимальное количество детекций (ограничиваем для ускорения)
                agnostic_nms=False,  # Не использовать agnostic NMS (быстрее)
                retina_masks=False  # Отключаем retina masks для ускорения
            )
            result = prediction[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                return {"boxes": None}
            
            # Возвращаем результаты
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            return {
                "boxes": boxes.tolist(),
                "classes": classes.tolist(),
                "confidences": confidences.tolist()
            }
        
        def predict_batch(self, image_arrays: List[np.ndarray], conf_threshold: float) -> List[Dict]:
            """Batch предсказание на нескольких изображениях (быстрее чем по одному)"""
            # Обрабатываем батч изображений за раз
            predictions = self.model.predict(
                source=image_arrays,
                conf=conf_threshold,
                verbose=False,
                device=self.device,
                half=(self.device == 'cuda'),  # FP16 для GPU
                imgsz=640,
                iou=0.45,
                max_det=300,
                agnostic_nms=False,
                retina_masks=False
            )
            
            results = []
            for result in predictions:
                if result.boxes is None or len(result.boxes) == 0:
                    results.append({"boxes": None})
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                results.append({
                    "boxes": boxes.tolist(),
                    "classes": classes.tolist(),
                    "confidences": confidences.tolist()
                })
            
            return results
else:
    # Заглушка если Ray недоступен
    class YOLOModelActor:
        pass

# Создаем пул Ray Actor'ов для модели (если Ray включен)
# Несколько Actor'ов позволяют обрабатывать запросы параллельно
model_actors = []
if USE_RAY and RAY_AVAILABLE:
    try:
        import torch
        # Проверяем доступность GPU
        has_gpu = torch.cuda.is_available()
        
        num_actors = int(os.getenv("RAY_NUM_ACTORS", 0))  # 0 = автоматический выбор
        if num_actors <= 0:
            import multiprocessing
            if has_gpu:
                # Для GPU используем 4 Actors (оптимально для RTX 4060)
                num_actors = 4
                logger.info(f"GPU detected, using {num_actors} Ray Actors")
            else:
                # Для CPU используем количество CPU
                num_actors = max(1, multiprocessing.cpu_count())
                logger.info(f"CPU only, using {num_actors} Ray Actors")
        
        # Создаем Actors с задержкой, чтобы избежать одновременной загрузки на GPU
        for i in range(num_actors):
            actor = YOLOModelActor.remote(str(model_path))
            model_actors.append(actor)
            # Небольшая задержка между созданием Actors для GPU
            if has_gpu and i < num_actors - 1:
                import time
                time.sleep(0.5)  # Даем время GPU освободиться
        
        logger.info(f"Created {len(model_actors)} Ray YOLO Actors for parallel processing")
    except Exception as e:
        logger.error(f"Failed to create Ray Actors: {e}. Falling back to sequential processing.")
        USE_RAY = False
        model_actors = []

# Конфигурация
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024  # 50MB по умолчанию
MAX_FILES = int(os.getenv("MAX_FILES", 100))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
# Оптимизация: используем 720 пикселей (баланс между скоростью и качеством)
# Можно снизить до 640 для еще большей скорости, но качество может ухудшиться
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", 720))
# Глобальная переменная для текущего DPI (для экспорта JSON)
CURRENT_PDF_DPI = int(os.getenv("PDF_DPI", 100))


# Функции валидации типа файла по magic bytes
def is_pdf(content: bytes) -> bool:
    """Проверка что файл действительно PDF по magic bytes"""
    return len(content) >= 4 and content[:4] == b'%PDF'

def is_jpeg(content: bytes) -> bool:
    """Проверка что файл действительно JPEG по magic bytes"""
    return len(content) >= 3 and content[:3] == b'\xff\xd8\xff'

def is_png(content: bytes) -> bool:
    """Проверка что файл действительно PNG по magic bytes"""
    return len(content) >= 8 and content[:8] == b'\x89PNG\r\n\x1a\n'

def validate_file_type(content: bytes, file_ext: str) -> Tuple[bool, str]:
    """
    Валидация типа файла по magic bytes
    Возвращает (is_valid, error_message)
    """
    file_ext = file_ext.lower()
    
    if file_ext == 'pdf':
        if not is_pdf(content):
            return False, "File extension is .pdf but file is not a valid PDF"
        return True, ""
    
    elif file_ext in ['jpg', 'jpeg']:
        if not is_jpeg(content):
            return False, f"File extension is .{file_ext} but file is not a valid JPEG"
        return True, ""
    
    elif file_ext == 'png':
        if not is_png(content):
            return False, "File extension is .png but file is not a valid PNG"
        return True, ""
    
    return False, f"Unsupported file type: .{file_ext}"

@app.get("/")
def read_root():
    return {
        "status": "QSS AI API is running",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/api/detect"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

def convert_pdf_to_images(pdf_path: str, filename: str) -> List[dict]:
    """Конвертирует PDF в изображения (синхронная функция для ThreadPoolExecutor)"""
    try:
        images = convert_from_path(pdf_path, dpi=CURRENT_PDF_DPI, thread_count=1)
        logger.info(f"PDF {filename} converted to {len(images)} pages")
        
        result = []
        for page_num, image in enumerate(images, 1):
            result.append({
                'image': image,
                'filename': f"{filename} - Page {page_num}",
                'original_size': (image.width, image.height)
            })
        return result
    except Exception as e:
        logger.error(f"PDF conversion failed for {filename}: {str(e)}")
        return []
    finally:
        if Path(pdf_path).exists():
            Path(pdf_path).unlink()

@app.post("/api/detect")
async def detect_objects(
    files: List[UploadFile] = File(...),
    export_json: bool = Query(False, description="Export results to JSON file"),
    include_images: bool = Query(True, description="Include base64 images in response (slower)")
):
    """Detect signatures, stamps, and QR codes in uploaded files"""
    
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, 
            detail=f"Maximum {MAX_FILES} files allowed"
        )
    
    # Шаг 1: Параллельная валидация и чтение файлов
    pdf_tasks = []  # Задачи для конвертации PDF
    image_tasks = []  # Готовые изображения
    
    # Оптимизация: читаем все файлы параллельно
    async def process_file(file: UploadFile):
        """Обработка одного файла"""
        try:
            # Валидация файла
            if not file.filename:
                logger.warning("File without filename skipped")
                return None
            
            # Проверка размера файла
            content = await file.read()
            file_size = len(content)
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File {file.filename} too large: {file_size} bytes")
                return None
            
            if file_size == 0:
                logger.warning(f"Empty file {file.filename} skipped")
                return None
            
            file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            
            # Валидация типа файла по magic bytes (безопасность)
            is_valid, error_msg = validate_file_type(content, file_ext)
            if not is_valid:
                logger.warning(f"File {file.filename} validation failed: {error_msg}")
                return None
            
            return (file_ext, content, file.filename)
        except Exception as e:
            logger.error(f"Error processing file {file.filename if file.filename else 'unknown'}: {str(e)}", exc_info=True)
            return None
    
    # Параллельно обрабатываем все файлы
    file_results = await asyncio.gather(*[process_file(file) for file in files])
    
    # Обрабатываем результаты
    for result in file_results:
        if result is None:
            continue
        
        file_ext, content, filename = result
        try:
            # Handle PDF - добавляем в очередь для параллельной конвертации
            if file_ext == 'pdf':
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(content)
                    pdf_path = tmp.name
                
                pdf_tasks.append((pdf_path, filename))
            
            # Handle images - сразу добавляем
            elif file_ext in ['jpg', 'jpeg', 'png']:
                try:
                    image = Image.open(io.BytesIO(content)).convert('RGB')
                    image_tasks.append({
                        'image': image,
                        'filename': filename,
                        'original_size': (image.width, image.height)
                    })
                except Exception as e:
                    logger.error(f"Image processing failed for {filename}: {str(e)}")
                    continue
            else:
                logger.warning(f"Unsupported file type: {file_ext} for {filename}")
                continue
        
        except Exception as e:
            logger.error(f"Error processing file {filename if 'filename' in locals() else 'unknown'}: {str(e)}", exc_info=True)
            continue
    
    # Шаг 2: Параллельная конвертация PDF через ThreadPoolExecutor
    if pdf_tasks:
        loop = asyncio.get_event_loop()
        import multiprocessing
        max_workers = min(len(pdf_tasks), max(24, multiprocessing.cpu_count() * 3))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, convert_pdf_to_images, pdf_path, filename)
                for pdf_path, filename in pdf_tasks
            ]
            pdf_results = await asyncio.gather(*futures)
            
            for result in pdf_results:
                image_tasks.extend(result)
    
    # Шаг 3: Параллельная обработка всех изображений через Ray
    results = []
    
    if USE_RAY and len(model_actors) > 0 and len(image_tasks) > 0:
        # Оптимизация: resize ДО отправки в Ray (меньше данных для сериализации)
        # Конвертируем в numpy arrays для быстрой передачи
        optimized_tasks = []
        for task in image_tasks:
            image = task['image']
            # Resize если нужно (делаем ДО отправки в Ray)
            if image.width > MAX_IMAGE_WIDTH:
                ratio = MAX_IMAGE_WIDTH / image.width
                new_size = (MAX_IMAGE_WIDTH, int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.BILINEAR)
            
            # Конвертируем в numpy array (быстрее сериализуется чем PIL Image)
            image_array = np.array(image)
            optimized_tasks.append({
                'image_array': image_array,
                'filename': task['filename'],
                'original_size': task['original_size'],
                'current_size': (image.width, image.height)
            })
        
        # Batch processing: группируем задачи по актерам для batch inference
        # Это ускоряет обработку, т.к. модель обрабатывает несколько изображений за раз
        batch_size = 4  # Обрабатываем по 4 изображения за раз на каждом актере
        futures = []
        
        # Распределяем задачи по актерам с батчингом
        for actor_idx, actor in enumerate(model_actors):
            # Берем задачи для этого актера (round-robin)
            actor_tasks = [optimized_tasks[i] for i in range(actor_idx, len(optimized_tasks), len(model_actors))]
            
            # Разбиваем на батчи
            for batch_start in range(0, len(actor_tasks), batch_size):
                batch_tasks = actor_tasks[batch_start:batch_start + batch_size]
                
                if len(batch_tasks) == 1:
                    # Одна задача - обычная обработка
                    task = batch_tasks[0]
                    future = process_image_ray_remote_optimized.remote(
                        actor,
                        task['image_array'],
                        task['filename'],
                        task['original_size'],
                        task['current_size'],
                        CONFIDENCE_THRESHOLD,
                        include_images
                    )
                    futures.append(future)
                else:
                    # Несколько задач - batch processing
                    batch_arrays = [t['image_array'] for t in batch_tasks]
                    batch_filenames = [t['filename'] for t in batch_tasks]
                    batch_sizes = [t['current_size'] for t in batch_tasks]
                    batch_original_sizes = [t['original_size'] for t in batch_tasks]
                    
                    future = process_batch_ray_remote.remote(
                        actor,
                        batch_arrays,
                        batch_filenames,
                        batch_sizes,
                        batch_original_sizes,
                        CONFIDENCE_THRESHOLD,
                        include_images
                    )
                    futures.append(future)
        
        logger.info(f"Processing {len(optimized_tasks)} images via Ray ({len(model_actors)} actors, batch_size={batch_size})...")
        detection_results = ray.get(futures)
        
        # Разворачиваем результаты (batch может вернуть список)
        for detection_result in detection_results:
            if detection_result:
                if isinstance(detection_result, list):
                    results.extend(detection_result)
                else:
                    results.append(detection_result)
    else:
        for task in image_tasks:
            detection_result = process_image(task['image'], task['filename'], include_images)
            if detection_result:
                results.append(detection_result)
    
    logger.info(f"Processed {len(files)} files, got {len(results)} results")
    
    response_data = {
        "results": results,
        "summary": {
            "total_files": len(files),
            "total_pages": len(image_tasks),
            "total_detections": len(results),
            "signatures": sum(1 for r in results if r.get('signatures', 0) > 0),
            "stamps": sum(1 for r in results if r.get('stamps', 0) > 0),
            "qr_codes": sum(1 for r in results if r.get('qr_codes', 0) > 0)
        }
    }
    
    # Если запрошен экспорт JSON - возвращаем полные данные для скачивания
    if export_json:
        export_data = prepare_export_data(results, files, len(image_tasks))
        response_data["export_data"] = export_data
        response_data["export_filename"] = generate_export_filename()
        logger.info("Export data prepared for download")
    
    return JSONResponse(content=response_data)

def prepare_export_data(results: List[dict], files: List[UploadFile], total_pages: int) -> dict:
    """Подготавливает данные для экспорта JSON в формате all_annotations"""
    from collections import defaultdict
    import re
    
    # Группируем результаты по файлам
    files_data = defaultdict(dict)
    
    for result in results:
        filename = result.get('filename', '')
        detections = result.get('detections', [])
        image_size = result.get('image_size', {})
        
        # Парсим filename: "filename.pdf - Page 1" -> ("filename.pdf", 1)
        # или просто "filename.jpg" -> ("filename.jpg", 1)
        page_match = re.search(r' - Page (\d+)$', filename)
        if page_match:
            page_num = int(page_match.group(1))
            base_filename = filename[:page_match.start()]
        else:
            # Одиночное изображение - считаем страницей 1
            page_num = 1
            base_filename = filename
        
        # Формируем ключ страницы
        page_key = f"page_{page_num}"
        
        # Преобразуем детекции в нужный формат
        annotations = []
        for idx, det in enumerate(detections):
            bbox = det.get('bbox', {})
            label = det.get('label', '').lower()
            
            # Преобразуем label: "Signature" -> "signature", "Stamp" -> "stamp", "QR Code" -> "qr_code"
            if label == "signature":
                category = "signature"
            elif label == "stamp":
                category = "stamp"
            elif label == "qr code":
                category = "qr_code"
            else:
                category = label.lower().replace(' ', '_')
            
            # Преобразуем bbox из формата x1, y1, x2, y2 в x, y, width, height
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0)
            y2 = bbox.get('y2', 0)
            
            x = float(x1)
            y = float(y1)
            width = float(x2 - x1)
            height = float(y2 - y1)
            area = width * height
            
            # Масштабируем координаты обратно к оригинальному размеру страницы
            # image_size - это размер после resize, нужно вернуться к оригинальному
            # Но в result нет original_size, используем image_size как есть
            # (предполагаем, что координаты уже в правильном масштабе)
            
            annotation_key = f"annotation_{idx}"
            annotations.append({
                annotation_key: {
                    "category": category,
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    },
                    "area": area
                }
            })
        
        # Получаем размеры страницы
        original_size = result.get('original_size', image_size)
        current_size = image_size
        
        page_width = original_size.get('width', 2480)  # Оригинальный размер
        page_height = original_size.get('height', 3509)
        
        current_width = current_size.get('width', 720)
        current_height = current_size.get('height', 1018)
        
        # Масштабируем координаты обратно к оригинальному размеру, если было resize
        if current_width != page_width or current_height != page_height:
            scale_x = page_width / current_width
            scale_y = page_height / current_height
            
            # Масштабируем все аннотации
            for ann_dict in annotations:
                for ann_key, ann_data in ann_dict.items():
                    bbox = ann_data['bbox']
                    # Масштабируем координаты
                    bbox['x'] = bbox['x'] * scale_x
                    bbox['y'] = bbox['y'] * scale_y
                    bbox['width'] = bbox['width'] * scale_x
                    bbox['height'] = bbox['height'] * scale_y
                    # Пересчитываем area
                    ann_data['area'] = bbox['width'] * bbox['height']
        
        # Если это новая страница или страница уже существует, обновляем
        if page_key not in files_data[base_filename]:
            files_data[base_filename][page_key] = {
                "annotations": annotations,
                "page_size": {
                    "width": int(page_width),
                    "height": int(page_height)
                }
            }
        else:
            # Если страница уже существует, добавляем аннотации
            existing_annotations = files_data[base_filename][page_key]["annotations"]
            # Обновляем индексы для новых аннотаций
            start_idx = len(existing_annotations)
            for idx, ann in enumerate(annotations):
                old_key = f"annotation_{idx}"
                new_key = f"annotation_{start_idx + idx}"
                ann_data = ann[old_key]
                annotations[idx] = {new_key: ann_data}
            existing_annotations.extend(annotations)
    
    return dict(files_data)

def generate_export_filename() -> str:
    """Генерирует имя файла для экспорта"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"dpi{CURRENT_PDF_DPI}_width{MAX_IMAGE_WIDTH}_conf{CONFIDENCE_THRESHOLD}"
    return f"detections_{timestamp}_{config_str}.json"

def process_image(image: Image.Image, filename: str, include_image: bool = True):
    """Process single image and return detection results"""
    
    try:
        original_size = (image.width, image.height)
        
        # Resize if needed
        if image.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / image.width
            new_size = (MAX_IMAGE_WIDTH, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized {filename} from {original_size} to {image.size}")
        
        # Обычная обработка (fallback, когда Ray не используется)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, 'JPEG', quality=95)
            tmp_path = tmp.name
        
        try:
            prediction = model.predict(
                source=tmp_path,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                iou=0.45,
                max_det=300,
                agnostic_nms=False,
                retina_masks=False
            )
            result = prediction[0]
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug(f"No detections in {filename}")
            return None
        
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
        
        # Count by type
        signatures = sum(1 for d in detections if d['label'] == 'Signature')
        stamps = sum(1 for d in detections if d['label'] == 'Stamp')
        qr_codes = sum(1 for d in detections if d['label'] == 'QR Code')
        
        result = {
            "filename": filename,
            "total": len(detections),
            "signatures": signatures,
            "stamps": stamps,
            "qr_codes": qr_codes,
            "detections": detections,
            "image_size": {"width": image.width, "height": image.height},
            "original_size": {"width": original_size[0], "height": original_size[1]}
        }
        
        # Генерируем base64 только если нужно (оптимизация производительности)
        if include_image:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            buffer = io.BytesIO()
            # Оптимизация: снижаем качество JPEG до 75 и используем subsampling=0 для максимальной скорости
            img_pil.save(buffer, format='JPEG', quality=75, optimize=True, subsampling=0)
            buffer.seek(0)
            img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
            result["image_base64"] = img_base64
        
        logger.info(f"Found {len(detections)} detections in {filename}: {signatures} signatures, {stamps} stamps, {qr_codes} QR codes")
        
        return result
    except Exception as e:
        logger.error(f"Error processing image {filename}: {str(e)}", exc_info=True)
        return None

# Ray remote функция для обработки изображений
# Использует model_actor для предсказания, остальная обработка локально
if RAY_AVAILABLE:
    @ray.remote
    def process_image_ray_remote_optimized(actor_handle, image_array: np.ndarray, filename: str, original_size: tuple, current_size: tuple, conf_threshold: float, include_image: bool = True):
        """Оптимизированная Ray remote функция - принимает уже готовый numpy array"""
        from PIL import Image
        import cv2
        import base64
        import io
        
        # Используем actor_handle для предсказания (image_array уже готов)
        prediction_result = ray.get(actor_handle.predict.remote(image_array, conf_threshold))
        
        if prediction_result["boxes"] is None:
            return None
        
        boxes = np.array(prediction_result["boxes"])
        classes = np.array(prediction_result["classes"]).astype(int)
        confidences = np.array(prediction_result["confidences"])
        
        # Draw boxes on image (используем уже готовый image_array)
        img_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
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
        
        # Count by type
        signatures = sum(1 for d in detections if d['label'] == 'Signature')
        stamps = sum(1 for d in detections if d['label'] == 'Stamp')
        qr_codes = sum(1 for d in detections if d['label'] == 'QR Code')
        
        result = {
            "filename": filename,
            "total": len(detections),
            "signatures": signatures,
            "stamps": stamps,
            "qr_codes": qr_codes,
            "detections": detections,
            "image_size": {"width": current_size[0], "height": current_size[1]},
            "original_size": {"width": original_size[0], "height": original_size[1]}
        }
        
        # Генерируем base64 только если нужно (оптимизация)
        if include_image:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            buffer = io.BytesIO()
            # Оптимизация: качество 75 для максимальной скорости
            img_pil.save(buffer, format='JPEG', quality=75, optimize=True, subsampling=0)
            buffer.seek(0)
            img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
            result["image_base64"] = img_base64
        
        return result
    
    @ray.remote
    def process_batch_ray_remote(actor_handle, image_arrays: List[np.ndarray], filenames: List[str], current_sizes: List[tuple], original_sizes: List[tuple], conf_threshold: float, include_image: bool = True):
        """Batch обработка нескольких изображений за раз (быстрее)"""
        from PIL import Image
        import cv2
        import base64
        import io
        
        # Batch prediction через actor
        batch_results = ray.get(actor_handle.predict_batch.remote(image_arrays, conf_threshold))
        
        results = []
        for idx, (prediction_result, image_array, filename, current_size, original_size) in enumerate(zip(batch_results, image_arrays, filenames, current_sizes, original_sizes)):
            if prediction_result["boxes"] is None:
                continue
            
            boxes = np.array(prediction_result["boxes"])
            classes = np.array(prediction_result["classes"]).astype(int)
            confidences = np.array(prediction_result["confidences"])
            
            # Draw boxes
            img_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            detections = []
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                if cls == 0:
                    label = "Signature"
                    color = (102, 126, 234)
                elif cls == 1:
                    label = "Stamp"
                    color = (240, 147, 251)
                else:
                    label = "QR Code"
                    color = (79, 172, 254)
                
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                cv2.putText(img_cv, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                })
            
            signatures = sum(1 for d in detections if d['label'] == 'Signature')
            stamps = sum(1 for d in detections if d['label'] == 'Stamp')
            qr_codes = sum(1 for d in detections if d['label'] == 'QR Code')
            
            result = {
                "filename": filename,
                "total": len(detections),
                "signatures": signatures,
                "stamps": stamps,
                "qr_codes": qr_codes,
                "detections": detections,
                "image_size": {"width": current_size[0], "height": current_size[1]}
            }
            
            if include_image:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=75, optimize=True, subsampling=0)
                buffer.seek(0)
                result["image_base64"] = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            results.append(result)
        
        return results
else:
    # Заглушка если Ray недоступен
    def process_image_ray_remote(*args, **kwargs):
        return None

# Shutdown handler для Ray
@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке сервера"""
    if USE_RAY and RAY_AVAILABLE:
        try:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown successfully")
        except Exception as e:
            logger.warning(f"Error during Ray shutdown: {e}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        # Очистка Ray при выходе
        if USE_RAY and RAY_AVAILABLE:
            try:
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass
