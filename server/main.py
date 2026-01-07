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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Đang load các Model AI... Vui lòng chờ...")
    
    
    ml_models["plate_detector"] = ImgToPlate() 
    
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    print("🔥 Warming up model...")
    try:
        ml_models["plate_detector"](dummy_img) 
    except:
        pass 
        
    print("✅ Model đã sẵn sàng!")
    yield
    
 
    ml_models.clear()
    print("🛑 Server shutting down")

class ImagePayload(BaseModel):
    image: str  

# Gắn lifespan vào app
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/get_plate")
async def img_plate(payload: ImagePayload):
    
    try:
        img_bytes = base64.b64decode(payload.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": "Invalid image data"}

    # Gọi model đã load sẵn từ lifespan

    detector = ml_models["plate_detector"]
    result = detector(img) 
    
    return {"message": result}