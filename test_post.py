# test_send_image.py
import requests
import base64
import time

# Đường dẫn ảnh muốn gửi
image_path = r"D:\2025.1\iot\drplate_ai\anh-bia-6.jpg"

# đọc ảnh, encode Base64
with open(image_path, "rb") as f:
    img_bytes = f.read()
img_base64 = base64.b64encode(img_bytes).decode("utf-8")

# tạo payload JSON
payload = {"image": img_base64}

# gửi POST request lên API
url = "http://127.0.0.1:8000/get_plate"  # đổi nếu API chạy ở host khác
time_a = time.time()
response = requests.post(url, json=payload)
time_b = time.time()
print(f"Time: {time_b - time_a}")
# in kết quả trả về
print(response.status_code)
print(response.json())
