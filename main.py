import os
import io
import uuid
import base64

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rembg import remove
from openai import OpenAI
from PIL import Image
import qrcode

# ------------------------------------------------------------
# CONFIGURAÇÃO DE PASTAS
# ------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FOTOS_DIR  = os.path.join(STATIC_DIR, "fotos")
QR_DIR     = os.path.join(STATIC_DIR, "qr")

# Cenário onde a pessoa será inserida (navio, loja, etc.)
SCENE_FILE = os.path.join(ASSETS_DIR, "ship_day_4k.jpg")

# Garante que as pastas existem
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FOTOS_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

# ------------------------------------------------------------
# FASTAPI + STATIC
# ------------------------------------------------------------
app = FastAPI(
    title="Totem IA Backend",
    description="Backend FastAPI para totem (Unity) com OpenAI + rembg + QR Code",
    version="1.0.0",
)

# Servir arquivos estáticos (fotos finais e QRs)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cliente da OpenAI (usa OPENAI_API_KEY do ambiente)
client = OpenAI()

# ------------------------------------------------------------
# PROMPT PARA OPENAI (ajuste o texto se quiser outro estilo)
# ------------------------------------------------------------
PROMPT = """
Use a PRIMEIRA imagem como referência exata da pessoa:
- mantenha rosto, idade, expressão, cabelo, tom de pele e roupa
- não alterar logos, cores ou texto da roupa

Insira essa pessoa de corpo inteiro no cenário da SEGUNDA imagem.
- posição: em pé, centralizada, olhando para a câmera
- escala realista
- combinar iluminação e sombras com o cenário

Estilo fotorealista, alta nitidez, sem reestilização.
""".strip()

# ------------------------------------------------------------
# ROTAS DE TESTE / SAÚDE
# ------------------------------------------------------------
@app.get("/")
def root():
    """Rota simples para testar se a API está online."""
    return {"status": "ok", "message": "Totem IA backend rodando."}


@app.get("/ping")
def ping():
    """Rota de ping para health-check."""
    return {"msg": "pong"}


# ------------------------------------------------------------
# ROTA PRINCIPAL: /compose
# - Recebe a foto do Unity
# - Remove fundo (rembg)
# - Junta com cenário
# - Chama OpenAI Images
# - Salva imagem final
# - Gera QR code para a URL da imagem
# - Devolve final_url + qr_url para o Unity
# ------------------------------------------------------------
@app.post("/compose")
async def compose(request: Request, file: UploadFile = File(...)):
    try:
        # ----------------------------------------------------
        # 1) Ler bytes da foto enviada pelo Unity
        # ----------------------------------------------------
        raw_bytes = await file.read()
        try:
            person_raw = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        except Exception:
            return JSONResponse(
                {"detail": "Arquivo enviado não é uma imagem válida."},
                status_code=400
            )

        # ----------------------------------------------------
        # 2) Remover o fundo da pessoa (rembg)
        # ----------------------------------------------------
        try:
            person_cut = remove(person_raw)  # retorna um PIL Image também
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao remover fundo (rembg): {e}"},
                status_code=500
            )

        # ----------------------------------------------------
        # 3) Carregar o cenário (navio, etc.)
        # ----------------------------------------------------
        if not os.path.exists(SCENE_FILE):
            return JSONResponse(
                {"detail": f"Cenário não encontrado em {SCENE_FILE}"},
                status_code=500
            )

        try:
            bg = Image.open(SCENE_FILE).convert("RGBA")
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao abrir cenário: {e}"},
                status_code=500
            )

        # ----------------------------------------------------
        # 4) Redimensionar para 1024x1024 (mais leve para IA)
        # ----------------------------------------------------
        target_size = (1024, 1024)
        bg = bg.resize(target_size, Image.LANCZOS)

        # Redimensionar a pessoa para ~70% da altura da imagem
        max_h = int(target_size[1] * 0.7)
        scale = max_h / person_cut.height
        new_w = int(person_cut.width * scale)
        new_h = int(person_cut.height * scale)
        person_resized = person_cut.resize((new_w, new_h), Image.LANCZOS)

        # ----------------------------------------------------
        # 5) Converter pessoa + cenário para PNG (bytes) para a OpenAI
        # ----------------------------------------------------
        pbuf = io.BytesIO()
        person_resized.save(pbuf, "PNG")
        person_png = pbuf.getvalue()

        bbuf = io.BytesIO()
        bg.save(bbuf, "PNG")
        bg_png = bbuf.getvalue()

        # ----------------------------------------------------
        # 6) Chamar OpenAI Images (edição)
        # ----------------------------------------------------
        try:
            result = client.images.edit(
                model="gpt-image-1",
                image=[
                    ("person.png", person_png),
                    ("scene.png",  bg_png),
                ],
                prompt=PROMPT,
                size="1024x1024",
            )
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao chamar OpenAI Images: {e}"},
                status_code=500
            )

        try:
            final_b64 = result.data[0].b64_json
        except Exception:
            return JSONResponse(
                {"detail": "Resposta da OpenAI não contém imagem válida."},
                status_code=500
            )

        try:
            final_bytes = base64.b64decode(final_b64)
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao decodificar imagem da OpenAI: {e}"},
                status_code=500
            )

        # ----------------------------------------------------
        # 7) Salvar imagem final com ID único
        # ----------------------------------------------------
        img_id = str(uuid.uuid4())
        foto_filename = f"{img_id}.jpg"
        foto_path = os.path.join(FOTOS_DIR, foto_filename)

        try:
            final_img = Image.open(io.BytesIO(final_bytes)).convert("RGB")
            final_img.save(foto_path, "JPEG", quality=95)
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao salvar imagem final: {e}"},
                status_code=500
            )

        # ----------------------------------------------------
        # 8) Montar base_url a partir da requisição (funciona local e na nuvem)
        #     ex: http://127.0.0.1:8000 ou https://meuapp.onrender.com
        # ----------------------------------------------------
        base_url = str(request.base_url).rstrip("/")  # tira / extra do final

        # URL pública da foto final
        foto_url = f"{base_url}/static/fotos/{foto_filename}"

        # ----------------------------------------------------
        # 9) Gerar QR CODE apontando para foto_url
        # ----------------------------------------------------
        qr_img = qrcode.make(foto_url)
        qr_filename = f"{img_id}.png"
        qr_path = os.path.join(QR_DIR, qr_filename)

        try:
            qr_img.save(qr_path)
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao salvar QR code: {e}"},
                status_code=500
            )

        qr_url = f"{base_url}/static/qr/{qr_filename}"

        # ----------------------------------------------------
        # 10) Retornar URLs para o Unity
        # ----------------------------------------------------
        return {
            "final_url": foto_url,
            "qr_url": qr_url,
        }

    except Exception as e:
        # Erro inesperado
        return JSONResponse({"detail": f"Erro interno: {e}"}, status_code=500)
