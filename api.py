# To start app type unicorn api:app --reload in bash

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from yaml import load

from utils.utils import get_model, text_label

app = FastAPI()

config = load(open('config.yaml')).get('model')
image_height, image_width = config['im_h'], config['im_w']
detection_threshold = config['det_threshold']
csv_path = Path(config['csv_path'])
model_path = config['path']

data = pd.read_csv(csv_path)
data = data.rename(columns={'supercategory_id': 'labels'})
data = data.rename(columns={'bbox': 'bboxes'})
data['labels'].astype('int64')
data = data.set_index('image_id')
mapping = data.groupby(['labels', 'supercategory']).size().reset_index().rename(columns={0: 'count'})

n_classes = mapping['labels'].nunique()

inference_model = get_model(n_classes, model_path)


@app.get("/")
def read_root():
    return JSONResponse(content={"content": {"Hello": "World"}})


@app.post("/search")
def search(file: UploadFile = File(...)):
    try:
        image = np.array(Image.open(file.file).convert("RGB").resize((image_height, image_width)))
    except Exception as e:
        return JSONResponse(content={"response": 404})

    device = torch.device('cpu')
    img = torch.tensor(image / 255.0, dtype=torch.float).to(device).permute(2, 0, 1)

    # getting inference on image
    outs = inference_model([img])

    output = [{k: v.to(device) for k, v in t.items()} for t in outs]

    logger.info(output)

    predicted = output[0]
    scores = predicted['scores']

    if len(scores) > 0:
        max_index = scores.argmax()
        max_score = scores[max_index]

        if max_score >= detection_threshold:
            predicted_label = int(predicted['labels'][max_index].to(device="cpu").numpy())
            pred_label_text = text_label(mapping, predicted_label)
            bbox = [float(i) for i in predicted['boxes'][max_index].detach().to(device="cpu").numpy()]

            return JSONResponse(content={"status": "Successfully",
                                         "label": predicted_label,
                                         'label_text': pred_label_text,
                                         'bbox': bbox})

    return JSONResponse(content={"status": "Failed", "label": '', 'label_text': '', 'bbox': ''})