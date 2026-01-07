# import cv2
# import numpy as np
# from ultralytics import YOLO
# from paddleocr import PaddleOCR

# class ImgToPlate():
#     def __init__(self,
#                  model_path=r".\50ep1000imgcar.onnx", # Đảm bảo đường dẫn đúng
#                  confidence=0.5,
#                  threshold=0.7,
#                  language='en'):
        
#         # 1. Load YOLO
#         print("Loading YOLO...")
#         self.detect_model = YOLO(model_path, task="detect")
        
#         # 2. Load PaddleOCR (Cấu hình tối ưu cho CPU như đã bàn)
#         print("Loading PaddleOCR...")
#         self.ocr_model = PaddleOCR(
#             lang=language,
#             use_doc_orientation_classify=False,
#             use_doc_unwarping=False,
#             use_textline_orientation=False,     # Tắt angle cls cho nhanh nếu biển số thẳng
#             enable_mkldnn=True,      # Tăng tốc CPU        
#         )
        
#         self.confidence = confidence
#         self.threshold = threshold

#     def __call__(self, image):
#         # BƯỚC 1: Đọc ảnh từ đường dẫn thành mảng numpy
#         # Đây là bước bạn bị thiếu
#         img_matrix = image
        
#         if img_matrix is None:
#             print(f"Error: Không đọc được ảnh tại {image_path}")
#             return []

#         # BƯỚC 2: Detect biển số bằng YOLO
#         # YOLO có thể nhận mảng numpy trực tiếp
#         detected_results = self.detect_model.predict(img_matrix, conf=self.confidence, iou=self.threshold, verbose=False)
        
#         boxes = detected_results[0].boxes.xyxy.tolist()
#         final_plates = []

#         # BƯỚC 3: Lặp qua từng biển số tìm được
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box) # Chuyển tọa độ sang số nguyên
            
#             # Crop ảnh biển số
#             crop_object = img_matrix[y1:y2, x1:x2]
            
            
#             ocr_result = self.ocr_model.ocr(crop_object)

#             # BƯỚC 5: Xử lý kết quả trả về (Parsing)
#             # Kết quả PaddleOCR trả về dạng list lồng nhau rất phức tạp
#             # Cấu trúc: [ [ [coords], (text, conf) ], ... ]
            
#             plate_text = ""
#             # for res in ocr_result:
#                 # res.print()
#                 # res.save_to_json("output")
#             # print (ocr_result)
#             if ocr_result and ocr_result[0] is not None:
#                 for line in ocr_result[0]["rec_texts"]:
#                     plate_text += line
#             # print(ocr_result[0]['rec_texts'])
            
#             # if ocr_result and ocr_result[0] is not None:
#             #     # Gộp các dòng text lại (ví dụ biển 2 dòng)
#             #     for line in ocr_result:
#             #         text_content = line[1][0]
#             #         plate_text += text_content + " "
                
#             #     plate_text = plate_text.strip()
#             #     final_plates.append(plate_text)
#             #     print(f"Plate {i}: {plate_text}")
#             # else:
#             #     print(f"Plate {i}: Unable to read text")

#         return plate_text

# # --- CHẠY THỬ ---
# if __name__ == "__main__":
#     # Sửa lại đường dẫn model YOLO của bạn cho đúng nếu cần
#     # Nếu không có file onnx, bạn có thể test tạm bằng 'yolov8n.pt'
#     try:
#         test_model = ImgToPlate() # Ví dụ dùng model chuẩn để test code
        
#         # Đường dẫn ảnh của bạn
#         img_path = r"D:\2025.1\iot\drplate_ai\bien_so_xe_may_2.jpg"
        
#         results = test_model(cv2.imread(img_path))
#         print("\n--- FINAL RESULTS ---")
#         print(results)
        
#     except Exception as e:
#         print(f"Lỗi xảy ra: {e}")
import cv2
import numpy as np
import onnxruntime as ort
from paddleocr import PaddleOCR

class ImgToPlate:
    def __init__(self, 
                 model_path=r"50ep1000imgcar.onnx", 
                 confidence=0.5, 
                 threshold=0.45,  
                 input_size=640): 
        
        self.input_size = input_size
        self.conf_threshold = confidence
        self.iou_threshold = threshold

        print(f">> Loading YOLO ONNX ({model_path})...")
        try:
            self.yolo_session = ort.InferenceSession(
                model_path, 
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.yolo_session.get_inputs()[0].name
        except Exception as e:
            print(f"Lỗi load YOLO: {e}")
            exit()

        
        print(">> Loading PaddleOCR...")
        self.ocr_model = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,   
            enable_mkldnn=True,              
        )

    def preprocess(self, img):
        """Chuẩn hóa ảnh đầu vào cho YOLO (Letterbox resize)"""
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        nh, nw = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (nw, nh))
        
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw, :] = img_resized
    
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1) 
        blob = np.expand_dims(blob, axis=0) 
        
        return blob, scale

    def __call__(self, image):
        if image is None:
            return ""

        # BƯỚC 1: Detect biển số (YOLO ONNX)
        blob, scale = self.preprocess(image)
        outputs = self.yolo_session.run(None, {self.input_name: blob})
        
        pred = np.squeeze(outputs[0]).T
        
        scores = np.max(pred[:, 4:], axis=1)
        keep = scores > self.conf_threshold
        pred = pred[keep]
        scores = scores[keep]
        
        if len(scores) == 0:
            return ""

        boxes = pred[:, :4]
        boxes[:, 0] -= boxes[:, 2] / 2  # x1
        boxes[:, 1] -= boxes[:, 3] / 2  # y1
        boxes[:, 2] += boxes[:, 0]      # x2
        boxes[:, 3] += boxes[:, 1]      # y2

        boxes /= scale
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)

        final_plate_text = ""

        
        if len(indices) > 0:
            i = indices.flatten()[0] 
            x1, y1, x2, y2 = map(int, boxes[i])
            
            
            h_img, w_img = image.shape[:2]
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)

            plate_img = image[y1:y2, x1:x2]
            
            
            ocr_result = self.ocr_model.ocr(plate_img)
            plate_text = ""
           
            if ocr_result and ocr_result[0] is not None:
                for line in ocr_result[0]["rec_texts"]:
                    plate_text += line
            

            return plate_text
            

if __name__ == "__main__":
    try:
        
        model_path = r"50ep1000imgcar.onnx" 
        img_path = r"D:\2025.1\iot\drplate_ai\test_data\bien-so-xe-99999.jpg"
        
        app = ImgToPlate(model_path=model_path) 
        
        img = cv2.imread(img_path)
        if img is None:
            print("Không đọc được ảnh!")
        else:
            import time
            start_time = time.time()
            
            result = app(img)
            
            print(f"Thời gian xử lý: {time.time() - start_time:.4f}s")
            print("\n--- KẾT QUẢ ---")
            print(f"Biển số: {result}")
        
    except Exception as e:
        print(f"Lỗi: {e}")
