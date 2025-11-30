import os
import io
import uuid
import base64

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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
    description="Backend FastAPI para totem (Unity) com OpenAI + QR Code",
    version="1.0.0",
)

# Servir arquivos estáticos (fotos finais e QRs)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cliente da OpenAI (usa OPENAI_API_KEY do ambiente)
client = OpenAI()

# ------------------------------------------------------------
# PROMPT PARA OPENAI (sem rembg, IA faz o recorte)
# ------------------------------------------------------------
PROMPT = """
Use a PRIMEIRA imagem como referência exata da pessoa:
- mantenha rosto, idade, expressão, cabelo, tom de pele e roupa
- não alterar logos, cores ou texto da roupa

Use a SEGUNDA imagem como cenário (um navio).
- recorte a pessoa da primeira imagem
- insira a pessoa em pé, em primeiro plano, centralizada, olhando para a câmera
- combine iluminação, sombras e cores com o cenário
- não estilizar como desenho; manter estilo fotográfico realista

Retorne uma única imagem final com a pessoa inserida no cenário do navio.
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
# - Redimensiona + prepara a imagem da pessoa
# - Carrega o cenário
# - Chama OpenAI Images com as duas imagens
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
            person_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return JSONResponse(
                {"detail": "Arquivo enviado não é uma imagem de imagem válida."},
                status_code=400
            )

        # ----------------------------------------------------
        # 2) Reduzir tamanho da foto da pessoa (pra aliviar memória)
        # ----------------------------------------------------
        max_dim = 1024
        w, h = person_img.size
        scale = min(max_dim / float(w), max_dim / float(h), 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            person_img = person_img.resize(new_size, Image.LANCZOS)

        # Converter para PNG bytes
        pbuf = io.BytesIO()
        person_img.save(pbuf, "PNG")
        person_png = pbuf.getvalue()

        # ----------------------------------------------------
        # 3) Carregar o cenário
        # ----------------------------------------------------
        if not os.path.exists(SCENE_FILE):
            return JSONResponse(
                {"detail": f"Cenário não encontrado em {SCENE_FILE}"},
                status_code=500
            )

        try:
            bg = Image.open(SCENE_FILE).convert("RGB")
        except Exception as e:
            return JSONResponse(
                {"detail": f"Erro ao abrir cenário: {e}"},
                status_code=500
            )

        # Redimensionar cenário para 1024x1024
        bg = bg.resize((1024, 1024), Image.LANCZOS)

        bbuf = io.BytesIO()
        bg.save(bbuf, "PNG")
        bg_png = bbuf.getvalue()

        # ----------------------------------------------------
        # 4) Chamar OpenAI Images (edição usando as duas imagens)
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
        # 5) Salvar imagem final com ID único
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
        # 6) Montar base_url a partir da requisição (funciona local e na nuvem)
        # ----------------------------------------------------
        base_url = str(request.base_url).rstrip("/")  # ex: https://seu-servico.onrender.com

        # URL pública da foto final
        foto_url = f"{base_url}/static/fotos/{foto_filename}"

        # ----------------------------------------------------
        # 7) Gerar QR CODE apontando para foto_url
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
        # 8) Retornar URLs para o Unity
        # ----------------------------------------------------
        return {
            "final_url": foto_url,
            "qr_url": qr_url,
        }

    except Exception as e:
        # Erro inesperado
        return JSONResponse({"detail": f"Erro interno: {e}"}, status_code=500)
