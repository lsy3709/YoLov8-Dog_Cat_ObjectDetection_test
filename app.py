from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# YOLOv8 모델 로드
model = YOLO("best.pt")  # 학습된 YOLOv8 모델 파일 경로

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # 이미지를 읽음
    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # YOLO 모델을 사용하여 추론
    results = model(image)

    # 탐지 결과를 이미지에 시각화
    annotated_image = results[0].plot()  # 탐지 결과가 그려진 이미지 (NumPy 배열)

    # NumPy 배열을 PIL 이미지로 변환
    result_image = Image.fromarray(annotated_image)

    # 이미지를 Base64로 인코딩
    img_io = io.BytesIO()
    result_image.save(img_io, format="JPEG")
    img_io.seek(0)
    encoded_image = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # JSON 응답으로 Base64 인코딩된 이미지 전달
    return jsonify({"image": encoded_image})

# JSON 결과 반환 엔드포인트
@app.route('/predict/json', methods=['POST'])
def predict_json():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # 이미지를 읽음
    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # YOLO 모델을 사용하여 추론
    results = model(image)

    # 탐지 결과 처리
    detections = []
    for result in results[0].boxes:
        detections.append({
            "class": int(result.cls.item()),  # 클래스 ID
            "label": model.names[int(result.cls.item())],  # 클래스 이름
            "confidence": float(result.conf.item()),  # 신뢰도
            "box": [float(coord) for coord in result.xyxy[0].tolist()]  # 바운딩 박스 좌표
        })

    # JSON 형식으로 반환
    return jsonify({
        "detections": detections
    })

# 이미지 반환 엔드포인트
@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # 이미지를 읽음
    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # YOLO 모델을 사용하여 추론
    results = model(image)

    # 탐지 결과를 이미지에 시각화
    annotated_image = results[0].plot()  # 탐지 결과가 그려진 이미지 (NumPy 배열)

    # NumPy 배열을 PIL 이미지로 변환
    result_image = Image.fromarray(annotated_image)

    # 이미지를 메모리에 저장
    img_io = io.BytesIO()
    result_image.save(img_io, format="JPEG")
    img_io.seek(0)

    # 클라이언트에게 이미지 반환
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
