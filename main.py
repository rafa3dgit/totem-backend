import os, io, base64, uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from rembg import remove
from openai import OpenAI
from PIL import Image
import qrcode

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FOTOS_DIR  = os.path.join(STATIC_DIR, "fotos")
QR_DIR     = os.path.join(STATIC_DIR, "qr")
SCENE_FILE = os.path.join(ASSETS_DIR, "ship_day_4k.jpg")

os.makedirs(FOTOS_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

client = OpenAI()  # usa OPENAI_API_KEY do ambiente

PROMPT = """ ... seu prompt ... """.strip()
