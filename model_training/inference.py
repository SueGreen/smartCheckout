from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from yaml import load
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from utils.utils import get_model, text_label

def run_inference(filename):
    print(f'inference got file {filename}')
    # Parse config
    config_path = Path('./config.yaml')
    config = load(open(config_path))
    experiment_n = config.get('experiment_n')
    seed = config.get('seed')
    data_params = config.get('data')
    data_path = Path(data_params.get('inference_data_path'))
    image_height, image_width = int(data_params['im_h']), int(data_params['im_w'])
    detection_threshold = float(data_params['det_threshold'])
    csv_path = data_path / data_params['csv_path']

    data = pd.read_csv(csv_path)
    n_classes = data['labels'].nunique()
    mapping = data.groupby(['labels', 'supercategory']).size().reset_index().rename(columns={0: 'count'})
    training_params = config.get('training')
    max_lr = float(training_params.get('max_lr'))
    n_epochs = int(training_params.get('n_epochs'))
    model_config_params = config.get('model')
    inference_model_path = model_config_params.get('path')


    image = np.array(Image.open(filename).convert("RGB").resize((image_height, image_width)))
    device = torch.device('cuda')
    img = torch.tensor(image / 255.0, dtype=torch.float).to(device).permute(2, 0, 1).to(device)
    # getting inference on image
    inference_model = get_model(n_classes, inference_model_path, device=device)
    inference_model.eval()
    outs = inference_model([img])
    output = [{k: v.to(device) for k, v in t.items()} for t in outs]
    # logger.info(output)

    predicted = output[0]
    scores = predicted['scores']
    if len(scores) > 0:
        max_index = scores.argmax()
        max_score = scores[max_index]

        predicted_label = int(predicted['labels'][max_index].to(device="cpu").numpy())
        pred_label_text = text_label(mapping, predicted_label)
        bbox = [float(i) for i in predicted['boxes'][max_index].detach().to(device="cpu").numpy()]
        if max_score >= detection_threshold:
            return {"status": "Successful", "label": predicted_label, 'label_text': pred_label_text, 'bbox': bbox, 'certainty': round(max_score.item(), 2), 'decisive': True}
        else:
            return {"status": "Successful", "label": predicted_label, 'label_text': pred_label_text, 'bbox': bbox, 'certainty': round(max_score.item(), 2), 'decisive': False}

