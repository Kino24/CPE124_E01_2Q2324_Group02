import requests

reqURL = 'http://127.0.0.1:8000/predict/'

def getPrediction(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            
            files = {'file': (image_path, image_file, 'image/jpeg')}
            response = requests.post(reqURL, files=files)
            data = response.json()
            return(data)
    except Exception as e:
        print ("File cannot be read. Please upload a proper image file!")