from fastapi import FastAPI
from pydantic import BaseModel
import sys

sys.path.append('/home/ubuntu/proj/bvh_create/MotionBERT')
from infer_wild2 import load_model
from infer_wild2 import model_pred




class Opts(BaseModel):
    vid_path: str
    out_path: str

model = None
app = FastAPI()

# create a route
@app.get("/")
def index():
    return {"text": "3_match≈"}


# Register the function to run during startup
@app.on_event("startup")
def startup_event():
    global model_pos
    global testloader_params
    model_pos, testloader_params = load_model()




import subprocess

# Your FastAPI route handlers go here
from argparse import Namespace
@app.post("/predict")
def predict_sentiment(opts: Opts):
    data = opts.dict() 
    data = Namespace(**data)
    data.pixel = False
    data.clip_len = 243
    data.focus = None
    # Указываем путь к sh-скрипту
    script_path = '/home/ubuntu/proj/app/alphapose.sh'
    # Запускаем sh-скрипт alphapose
    subprocess.run(['sh', script_path], check=True)
    # Следующая команда будет выполнена только после завершения скрипта
    print("Скрипт выполнен. alphapose.")
    data.json_path = '/home/ubuntu/proj/data/alphapose/alphapose-results.json'
    #data.vid_path
    angle = model_pred(data, model_pos, testloader_params)
    return angle
    
    
