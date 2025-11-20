import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

class ImgToPlate():
    def __init__(self,
                 model_path=r".\50ep1000imgcar.onnx", # Đảm bảo đường dẫn đúng
                 confidence=0.5,
                 threshold=0.7,
                 language='en'):
        
        # 1. Load YOLO
        print("Loading YOLO...")
        self.detect_model = YOLO(model_path, task="detect")
        
        # 2. Load PaddleOCR (Cấu hình tối ưu cho CPU như đã bàn)
        print("Loading PaddleOCR...")
        self.ocr_model = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,     # Tắt angle cls cho nhanh nếu biển số thẳng
            enable_mkldnn=True,      # Tăng tốc CPU        
        )
        
        self.confidence = confidence
        self.threshold = threshold

    def __call__(self, image):
        # BƯỚC 1: Đọc ảnh từ đường dẫn thành mảng numpy
        # Đây là bước bạn bị thiếu
        img_matrix = image
        
        if img_matrix is None:
            print(f"Error: Không đọc được ảnh tại {image_path}")
            return []

        # BƯỚC 2: Detect biển số bằng YOLO
        # YOLO có thể nhận mảng numpy trực tiếp
        detected_results = self.detect_model.predict(img_matrix, conf=self.confidence, iou=self.threshold, verbose=False)
        
        boxes = detected_results[0].boxes.xyxy.tolist()
        final_plates = []

        # BƯỚC 3: Lặp qua từng biển số tìm được
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box) # Chuyển tọa độ sang số nguyên
            
            # Crop ảnh biển số
            crop_object = img_matrix[y1:y2, x1:x2]
            
            
            ocr_result = self.ocr_model.ocr(crop_object)

            # BƯỚC 5: Xử lý kết quả trả về (Parsing)
            # Kết quả PaddleOCR trả về dạng list lồng nhau rất phức tạp
            # Cấu trúc: [ [ [coords], (text, conf) ], ... ]
            
            plate_text = ""
            # for res in ocr_result:
                # res.print()
                # res.save_to_json("output")
            # print (ocr_result)
            if ocr_result and ocr_result[0] is not None:
                for line in ocr_result[0]["rec_texts"]:
                    plate_text += line
            # print(ocr_result[0]['rec_texts'])
            
            # if ocr_result and ocr_result[0] is not None:
            #     # Gộp các dòng text lại (ví dụ biển 2 dòng)
            #     for line in ocr_result:
            #         text_content = line[1][0]
            #         plate_text += text_content + " "
                
            #     plate_text = plate_text.strip()
            #     final_plates.append(plate_text)
            #     print(f"Plate {i}: {plate_text}")
            # else:
            #     print(f"Plate {i}: Unable to read text")

        return plate_text

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Sửa lại đường dẫn model YOLO của bạn cho đúng nếu cần
    # Nếu không có file onnx, bạn có thể test tạm bằng 'yolov8n.pt'
    try:
        test_model = ImgToPlate() # Ví dụ dùng model chuẩn để test code
        
        # Đường dẫn ảnh của bạn
        img_path = r"D:\2025.1\iot\drplate_ai\bien_so_xe_may_2.jpg"
        
        results = test_model(cv2.imread(img_path))
        print("\n--- FINAL RESULTS ---")
        print(results)
        
    except Exception as e:
        print(f"Lỗi xảy ra: {e}")