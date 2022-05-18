from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.templating import Jinja2Templates
import shutil
import ocr
import numpy as np
import os
import uuid
from PIL import Image
from io import BytesIO
import json
import cv2
from paddleocr import PaddleOCR,draw_ocr
import easyocr

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
reader = easyocr.Reader(download_enabled=False, model_storage_directory='ocr-model/', lang_list=['en'])
app = FastAPI()
templates = Jinja2Templates(directory="templates")




@app.get("/")
def home(request: Request):
    return {"message": "Welcome"}


@app.post("/image_extract_text")
async def images(img1: UploadFile = File(...)):
    contents = await img1.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    text = await ocr.read_image(img, reader)
    return {"filename":img1.filename, "message": text}


@app.post("/api/v1/bulk_extract_text")
async def bulk_extract_text(request: Request, bg_task: BackgroundTasks):
    images = await request.form()
    folder_name = str(uuid.uuid4())
    os.mkdir(folder_name)

    for image in images.values():
        temp_file = _save_file_to_disk(image, path=folder_name, save_as=image.filename)

    bg_task.add_task(ocr.read_images_from_dir, folder_name, write_to_file=True)
    return {"task_id": folder_name, "num_files": len(images)}

@app.get("/api/v1/bulk_output/{task_id}")
async def bulk_output(task_id):
    text_map = {}
    for file_ in os.listdir(task_id):
        if file_.endswith("txt"):
            text_map[file_] = open(os.path.join(task_id, file_)).read()
    return {"task_id": task_id, "output": text_map}

def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file

def _save_list_to_disk(results,path, save_as='default'):
    extension='.txt'
    name = os.path.join(path, save_as + extension)
    with open(name, 'w') as f:
        for item in results:
            f.write("%s\n" % item)

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))