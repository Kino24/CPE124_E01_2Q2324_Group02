import requests
#API URL
reqURL='http://localhost:8002/get_prediction'
query='download (3).jpg'
classification, prediction = requests.get(reqURL,params=query)

print(classification.json, prediction.json)