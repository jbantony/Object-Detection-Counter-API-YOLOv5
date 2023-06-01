from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import numpy as np
import cv2
from collections import defaultdict
import urllib.request

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES = "model/coco.names"
MODEL = "model/yolov5s.onnx"
IS_CUDA = False

async def load_model(is_cuda):
    #load the YOLOv5 model
    net = cv2.dnn.readNet(MODEL)
    if is_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("Model is configured on CUDA")
        # if not working, then re-install opencv with CUDA support 
        # recheck the GPU performance: https://github.com/opencv/opencv/issues/16348 
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model is configured for CPU as no-CUDA is provided")
    return net

async def image_preprocess(img):
    #function to return a square image
    row, col, ch = img.shape
    max_size = max(row, col)
    img_mod = np.zeros((max_size, max_size, 3), np.uint8)
    img_mod[0:row, 0:col] = img
    return img_mod


async def detect_object(img, net):
    
    img_copy = img

    img_mod = await image_preprocess(img_copy)
    # TODO: Recheck the need of this resize

    # Perform detection
    blob = cv2.dnn.blobFromImage(img_mod, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop= False)
    net.setInput(blob)
    preds = net.forward()
    

    #load object classes
    classes = []
    with open(CLASS_NAMES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    #print(classes)

    #layer_names= net.getLayerNames()
    #print(layer_names)
    #output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)

    #preds = net.forward(output_layers)
    #print(preds)
    #print(preds[0].shape)

    class_ids = []
    confidences = []
    boxes = []

    output_data = preds[0]
    rows = output_data.shape[0]

    image_width, image_height, _ = img.shape
    # TODO: recheck the boxes dimension adanist image

    x_factor = image_width/INPUT_WIDTH
    y_factor = image_height/INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > SCORE_THRESHOLD):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
    ordered_classes = {"label": [], "confidence": [], "boxes": []}
    object_count = defaultdict(int)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            ordered_classes['boxes'].append([int(x),int(y),int(w),int(h)])
            label = str(classes[class_ids[i]])
            ordered_classes['label'].append(label)
            confidence_label = int(confidences[i] * 100)
            ordered_classes['confidence'].append(confidence_label)
            object_count[str(label)] +=1

    return({"detected objects":ordered_classes, "counts":object_count, "image height":image_height, "image width": image_width})


app = FastAPI()

# adding origins to support CORS: https://fastapi.tiangolo.com/tutorial/cors/
origins = ["http://127.0.0.1","*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message":"Welcome to object detection API"}

@app.post("/detect/")

async def model_inference(file: UploadFile = File(...)):

    image_data = await file.read()
    image_name = file.filename
    np_image = np.frombuffer(image_data, dtype=np.uint8)
    #decode the image from the NumPy array
    image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)

    #print(image.shape)
    net = await load_model(IS_CUDA)
    result = await detect_object(image, net)

    #print(type(image_name))

    return {"filename": image_name, "detection results": result}


if __name__ == "__main__":

    PORT = int(os.environ.get('PORT', 5000))
    uvicorn.run("detection_app:app",host="0.0.0.0", port=PORT)