# To start app type "uvicorn api:app --reload" in bash
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
from model_training.inference import run_inference

app = FastAPI()

@app.get("/")
def read_root():
    return JSONResponse(content={"content": {"Hello": "World"}})

@app.post("/predict/")
def predict_item(file: UploadFile = File(...)):
    return JSONResponse(content=run_inference(file.file))