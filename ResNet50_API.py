# NOTE PLEASE USE http://localhost:8002/docs# to access the API

from PIL import Image
import numpy as np #  A library for numerical computing in Python.
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # ASGI server to run the FastAPI application.
import uvicorn
from io import BytesIO # A class for working with binary data in memory.
from PIL import Image # A library for image processing.
from typing import Tuple # A library for type hints.
import tensorflow as tf # A library for machine learning.
from fastapi.responses import HTMLResponse

app = FastAPI()

model=tf.keras.models.load_model('./classifier_resnet_model.keras')
class_names = ['BA-cellulitis','Ba-impetigo','FU-athlete-foot','FU-nail-fungus',
               'FU-ringworm','PA-cutaneous-larva-migrans','VI-chickenpox',
               'VI-shingles']

def read_image_file(data) -> Tuple[np.array, Tuple[int,int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((244,244), resample=Image.BICUBIC)
    image = np.array(img_resized)
    return image, img_resized.size

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    try:
        image, img_size = read_image_file(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions= model.predict(img_batch)
        predicted_class=class_names[np.argmax(predictions[0])]
        confidence=np.max(predictions[0])
        
        return {'class':predicted_class,'confidence': float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
    
