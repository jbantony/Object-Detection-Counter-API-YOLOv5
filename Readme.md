# Object Detection and Counter as API using YOLOv5 

The project is aimed to target AI-as-a-Service deployment using YOLOv5 model, served in Docker. The user can send an image to the Docker to the API endpoint using a POST request and the detected results as well as the counts are given as output in JSON format.

## Goal
The goal of this project is to test and demonstrate the AI-Serving concept, by deploying the Object Detection model as a docker image, so that any application can be interacted with this docker by making a POST request at the API endpoint with an image data and can receive the detection results and counts as the response. This project uses the popular YOLOv5 model for the deployment. The custom trained YOLOv5 Model can be converted into ONNX format and then deployed in docker.

The API has been developed in FastAPI and made available as docker.

## Model
The project deploys YOLOv5 model, which has been trained on your custom dataset. For information on training YOLOv5 object detection model on a custom dataset, use this [Notebook on Colab](https://github.com/jbantony/yolov5-custom-training-tutorial) 

## Make the Model Running
1. After cloning this reposistory, get your trained model file using the above mentioned link.
2. Replace the contents in the `model/coco.names` file with your custom classes and indexes
3. Convert the downloaded model (from step 1) to the ONNX format using the official script
     Follow this Colab for [more information on conversion to ONNX](https://github.com/jbantony/yolov5-custom-training-tutorial/blob/1ddd26ead95f68f74f04e0efb5474602fbc1f229/Convert_YOLOv5_ONNX_for_Inference.ipynb)
     
4. Copy the converted Model file in the `model` folder and rename it as `yolov5s.onnx` (feel free to change the name, also edit in the main script)

**PS**: If you wish to keep the original names for your application, please match the below global variables in the `detection_app.py` script to your desired names for model and classes:
```
CLASS_NAMES = "model/coco.names"
MODEL = "model/yolov5s.onnx" 
```

5. Build and tag the docker image using:
`docker build -t od-api .`

6. After sucessfull building, run the docker using:
`docker run -ti --rm -p 5000:5000 od-api`

### [OR] Use Docker compose
Skip step 5 & 6 and use docker compose to build & run the images

Build & Run:  `docker compose up -d`

Stop the docker: `docker compose down`

#### Endpoint `/detect/`

The endpoint `/detect/` is used to perform object detection on the given image. This return a JSON file in the below format:
```json
{
    "filename": "...",
    "detection results": {
        "detected objects":{
        "label": [...],
        "confidence": [...],
        "boxes": [[x,y,w,h], [x1,y1,w1,h1], ...]
    },
    "counts":{
        "class name": "no of instances", ...
    },
    "image height": ...,
    "image width": ...
    }
}
```

where the  `detection results` gives the information on the `detected objects` including the `label`, `confidence` and bounding `boxes`.  The `counts` gives the total count of each classes in the given image.
 

## Usage
After deploying the docker locally, one can make a POST request using the `curl` command:

`curl -X POST -F "file=@/path/to/image.jpg" http://localhost:5000/detect/` and gives the JSON data in the above mentioned format as response.

## Demo

The `/detect/` endpoint responses as follows: From FastAPI Documentation



##ToDo
- Add the ONNX conversion script and the requirements file
- update docker to receive model and class file as env varaibles

