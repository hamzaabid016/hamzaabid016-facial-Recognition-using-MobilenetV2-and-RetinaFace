from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import List, Dict, Any
import uuid
import datetime
import os
import base64
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import Normalizer
from face_detection import RetinaFace
from scipy.spatial.distance import cosine
import tools.database as db

app = FastAPI()

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
face_encoder = Model(inputs=base_model.input, outputs=base_model.output)
face_detector = RetinaFace()

l2_normalizer = Normalizer('l2')
confidence_t = 0.94  
recognition_t = 0.2  
required_size = (160, 160)

last_logged_time: Dict[str, datetime.datetime] = {}

def normalize(img: np.ndarray) -> np.ndarray:
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_encode(face_encoder, face: np.ndarray, size: tuple) -> np.ndarray:
    face = cv2.resize(face, size)
    face = preprocess_input(face)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def get_face(img: np.ndarray, box: List[int]) -> tuple:
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def change_box_value(bbox: List[float]) -> List[int]:
    return [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]

def image_to_bytes(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

def log_person(name: str) -> bool:
    current_time = datetime.datetime.now()
    if name in last_logged_time:
        last_log = last_logged_time[name]
        if (current_time - last_log).total_seconds() > 7200:
            db.log_entry(name)
            last_logged_time[name] = current_time
            return True
    else:
        db.log_entry(name)
        last_logged_time[name] = current_time
        return True
    return False

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector(img_rgb)

    face_data: List[Dict[str, Any]] = []
    for res in results:
        if res[2] < confidence_t:
            continue
        bbox = change_box_value(res[0])
        face, pt_1, pt_2 = get_face(img_rgb, bbox)
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

        name = 'unknown'
        min_dist = float("inf")
        datas_s = db.get_all_data()
        datas = db.data_setting(datas_s)
        for data in datas:
            dist = cosine(data[-1], encode)
            if dist < recognition_t and dist < min_dist:
                name = data[0]
                min_dist = dist

        color = (0, 255, 0) if name != 'unknown' else (0, 0, 255)
        cv2.rectangle(img, pt_1, pt_2, color, 2)
        cv2.putText(img, name, (pt_1[0], pt_1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name == 'unknown':
            user_id = str(uuid.uuid4()).split('-')[0]
            db.insert_data(user_id, name, 'ref_no', 'summary', image_to_bytes(img), encode)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            face_data.append({"id": user_id, "status": "new", "timestamp": timestamp})
        else:
            if log_person(name):
                face_data.append({"name": name, "status": "logged", "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")})

    return JSONResponse(content={"faces": face_data})

@app.get("/logs/")
async def get_logs() -> JSONResponse:
    logs = db.get_all_logs()
    labeled_logs = [{"log_id": log[0], "uu_id": log[1], "timestamp": log[2]} for log in logs]
    return JSONResponse(content={"logs": labeled_logs})

@app.get("/faces/")
async def get_faces() -> JSONResponse:
    datas_s = db.get_all_data()
    faces_data = []
    for data in datas_s:
        uu_id, name, _, _, image_bytes, image_encodes_bytes = data
        
        image_encodes = np.frombuffer(image_encodes_bytes, dtype=np.float32)
        image_encodes_base64 = base64.b64encode(image_encodes).decode('utf-8')
        face_info = {
            "uu_id": uu_id,
            "name": name,
            "image_encodes": image_encodes_base64
        }
        faces_data.append(face_info)
    return JSONResponse(content={"faces": faces_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
