import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pathlib import Path
import os
#API URL
reqURL='http://localhost:8002/getDirectory'
directory=os.path.dirname(os.path.realpath(__file__))
query='download (4).jpg'
files=(Path(directory))
fileOpen=files/query
response = requests.get(f"{reqURL}/{fileOpen}")
data=response.json()
classification=data.get('class')
prediction=data.get('confidence')

print(classification, prediction)