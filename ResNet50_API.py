# NOTE PLEASE USE http://localhost:8002/docs# to access the API

from PIL import Image
import numpy as np #  A library for numerical computing in Python.
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn # ASGI server to run the FastAPI application.
import io
from io import BytesIO # A class for working with binary data in memory.
from PIL import Image # A library for image processing.
from typing import Tuple # A library for type hints.
import tensorflow as tf # A library for machine learning.
from pathlib import Path
import imageio as iio

app = FastAPI()
model=tf.keras.models.load_model('./classifier_resnet_model.keras')
class_names = ['Ba-impetigo','VI-chickenpox','VI-shingles']

def read_image_file(data) -> Tuple[np.array, Tuple[int,int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((244,244), resample=Image.BICUBIC)
    image = np.array(img_resized)
    return image, img_resized.size

def pil_image_to_bytes(image):
    # Create a BytesIO object to store the bytes
    img_bytesio = io.BytesIO()

    # Save the PIL image to the BytesIO object in PNG format
    image.save(img_bytesio, format='PNG')

    # Get the byte-like object from BytesIO
    img_bytes = img_bytesio.getvalue()

    return img_bytes
# @app.post("/predict/{name}")
def predict(file):
    try:
        image, img_size = read_image_file(file)
        img_batch = np.expand_dims(image, 0)
        predictions= model.predict(img_batch)
        predicted_class=class_names[np.argmax(predictions[0])]
        confidence=np.max(predictions[0])
        return predicted_class, confidence
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))

@app.get("/getDirectory/{directory_path:path}")
async def getDirectory(directory_path: Path):
    imgData= Image.open(directory_path).tobytes()
    predicted_class, confidence = predict(imgData)
    return predicted_class, confidence

""" @app.get("/getPrediction")
async def getPrediction(file):
    filename=file
    predict(filename)
    return predicted_class, confidence """

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
    
