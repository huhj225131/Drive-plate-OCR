import numpy as np
import argparse
import cv2
import os
from ultralytics import YOLO
import easyocr

class ImgToPlate():
    def __init__(self,
                 model_path=r".\50ep1000imgcar.onnx",
                 confidence = 0.5,
                 threshold=0.7,
                 language=['en'],
                 ):
        self.detect_model = YOLO(model_path,task="detect")
        self.ocr_model = easyocr.Reader(language)
        self.confidence = confidence
        self.threshold = threshold
    def __call__(self, image):
        detected_img = self.detect_model.predict(image, conf=self.threshold, iou= self.threshold)
        # img = cv2.imread(image_path)
        boxes = detected_img[0].boxes.xyxy.tolist()

        # Iterate through the bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            crop_object = image[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite('ultralytics_crop_' + str(i) + '.jpg', crop_object)
        results = self.ocr_model.readtext(crop_object)
        if (len(results) == 2):
            _, line_1, _ = results[0]
            _, line_2, _ = results[1]
            return(f"{line_1}{line_2}")
        else:
            _, line, _ = results[0]
            return(line)
if __name__ == "__main__":
    testmodel = ImgToPlate()
    print(testmodel(cv2.imread(r"D:\2025.1\iot\drplate_ai\dich-bien-so-xe-phong-thuy (8).bcaf2533.jpg")))
