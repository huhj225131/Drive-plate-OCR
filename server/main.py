from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import base64
import cv2
import numpy as np

# Import class nhận diện của bạn
from model_yolo_paddle import ImgToPlate

# Biến toàn cục để chứa model
ml_models = {}

# --- 1. LIFESPAN: Chạy khi server bật ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Đang load các Model AI... Vui lòng chờ...")
    
    # Khởi tạo model ở đây. Nó sẽ chỉ chạy 1 lần duy nhất.
    # 'ocr_version'='PP-OCRv4' thường mặc định là mobile, nhẹ hơn server
    ml_models["plate_detector"] = ImgToPlate() 
    
    # Mẹo: Chạy thử 1 lần dummy (Warm-up) để các thư viện load hết vào cache
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    print("🔥 Warming up model...")
    try:
        ml_models["plate_detector"](dummy_img) 
    except:
        pass # Bỏ qua lỗi nếu có, mục đích chỉ để load thư viện
        
    print("✅ Model đã sẵn sàng!")
    yield
    
    # Code chạy khi server tắt (dọn dẹp bộ nhớ nếu cần)
    ml_models.clear()
    print("🛑 Server shutting down")

class ImagePayload(BaseModel):
    image: str  # Base64 string

# Gắn lifespan vào app
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/get_plate")
async def img_plate(payload: ImagePayload):
    # 1. Decode Base64
    try:
        img_bytes = base64.b64decode(payload.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": "Invalid image data"}

    # 2. Gọi model đã load sẵn từ lifespan
    # Lưu ý: Cần sửa class ImgToPlate để nhận numpy array (xem Bước 2 dưới)
    detector = ml_models["plate_detector"]
    result = detector(img) 
    
    return {"message": result}