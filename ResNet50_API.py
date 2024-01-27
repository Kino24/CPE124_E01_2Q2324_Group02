from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import uvicorn
from io import BytesIO
from PIL import Image
from typing import Tuple
import tensorflow as tf

app = FastAPI()

model=tf.keras.models.load_model('./classifier_resnet_model.keras')
class_names = ['Ba-impetigo','VI-chickenpox',
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
    