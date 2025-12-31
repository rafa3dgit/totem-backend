import os              # Módulo padrão do Python para lidar com caminhos, pastas e arquivos
import io              # Módulo para trabalhar com streams de bytes na memória (buffer)
import uuid            # Gera IDs únicos (usado para nomear arquivos sem repetir)

from fastapi import FastAPI, UploadFile, File, Request      # FastAPI: framework web; UploadFile/File para receber arquivos, Request para info da requisição
from fastapi.responses import JSONResponse                  # Resposta em JSON personalizada (permite setar status code)
from fastapi.staticfiles import StaticFiles                 # Para servir arquivos estáticos (imagens etc.)

from PIL import Image                                       # PIL (Pillow) para tratar imagens
import qrcode                                               # Biblioteca para gerar QR Codes
from io import BytesIO  # para abrir bytes como imagem PIL

from google import genai                                    # SDK oficial da Gemini API (google-genai) :contentReference[oaicite:2]{index=2}
from google.genai import types                              # Tipos auxiliares (config, etc.)

# ------------------------------------------------------------
# CONFIGURAÇÃO DE PASTAS
# ------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))     # Pega a pasta onde este arquivo main.py está
ASSETS_DIR = os.path.join(BASE_DIR, "assets")               # Pasta para armazenar recursos (ex: imagem de cenário, moldura)
STATIC_DIR = os.path.join(BASE_DIR, "static")               # Pasta de arquivos estáticos que serão expostos via HTTP
FOTOS_DIR  = os.path.join(STATIC_DIR, "fotos")              # Dentro de static, pasta para salvar as fotos finais
QR_DIR     = os.path.join(STATIC_DIR, "qr")                 # Dentro de static, pasta para salvar os QR Codes

# Cenário onde a(s) pessoa(s) será(ão) inserida(s) (navio, loja, etc.)
SCENE_FILE = os.path.join(ASSETS_DIR, "ship_day_4k.jpg")    # Caminho completo da imagem de cenário usada pela IA

# Moldura criada no Photoshop (PNG com transparência)
FRAME_FILE = os.path.join(ASSETS_DIR, "frame.png")          # Caminho completo da moldura (opcional)

# Garante que as pastas existem (se não existir, cria)
os.makedirs(ASSETS_DIR, exist_ok=True)                      # Cria a pasta assets se não existir
os.makedirs(STATIC_DIR, exist_ok=True)                      # Cria a pasta static se não existir
os.makedirs(FOTOS_DIR, exist_ok=True)                       # Cria a pasta static/fotos se não existir
os.makedirs(QR_DIR, exist_ok=True)                          # Cria a pasta static/qr se não existir

# ------------------------------------------------------------
# GEMINI API - CONFIG
# ------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash-image"                     # Modelo de imagem da Gemini API :contentReference[oaicite:3]{index=3}

# Lê a chave de API da variável de ambiente GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")                # Pega a key do ambiente (defina antes de rodar o servidor)

if not GEMINI_API_KEY:                                      # Se a chave não estiver configurada
    raise RuntimeError("GEMINI_API_KEY não configurada. Defina a variável de ambiente com sua chave da Gemini API.")

# Cria o cliente Gemini
client = genai.Client(api_key=GEMINI_API_KEY)               # Cliente da Gemini API (Google AI)

# ------------------------------------------------------------
# PROMPT PARA GEMINI (VÁRIAS PESSOAS + CENÁRIO)
# ------------------------------------------------------------
PROMPT = """
Use a PRIMEIRA imagem como referência da pessoa:

- estilizar como desenho 3D da Disney/Pixar
- não alterar logos, cores ou textos presentes na roupa

Use a SEGUNDA imagem como cenário.
- combinar iluminação, sombras e cores com o cenário
- estilizar como desenho 3D da Disney/Pixar

Retorne uma única imagem final com a pessoa etilizada como desenho 3D da Disney/Pixar  inserida no cenário do navio.
""".strip()                                                 # Remove espaços extras no começo/fim do texto

# ------------------------------------------------------------
# FASTAPI + STATIC
# ------------------------------------------------------------
app = FastAPI(                                              # Cria a aplicação FastAPI
    title="Totem IA Backend (Gemini)",                      # Título (aparece na documentação /docs)
    description="Backend FastAPI para totem (Unity) com Gemini + QR Code",  # Descrição da API
    version="2.0.0",                                        # Versão da API (agora 2.0 com Gemini)
)

# Servir arquivos estáticos (fotos finais e QRs)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # Tudo em STATIC_DIR fica acessível na rota /static

# ------------------------------------------------------------
# ROTAS DE TESTE / SAÚDE
# ------------------------------------------------------------
@app.get("/")                                               # Define a rota GET / (raiz) da API
def root():                                                 # Função que será executada quando alguém acessar GET /
    """Rota simples para testar se a API está online."""    # Docstring explicando a função
    return {"status": "ok", "backend": "gemini"}            # Retorna um JSON com status informativo

@app.get("/ping")                                           # Define rota GET /ping
def ping():                                                 # Função que responde ao ping
    """Rota de ping para health-check."""                   # Docstring
    return {"msg": "pong"}                                  # Retorna JSON simples, útil para ver se o servidor está vivo

# ------------------------------------------------------------
# FUNÇÃO AUXILIAR: CHAMAR GEMINI E PEGAR A IMAGEM FINAL
# ------------------------------------------------------------
def gerar_imagem_gemini(person_img: Image.Image, bg_img: Image.Image) -> Image.Image:
    """
    Envia PROMPT + foto da(s) pessoa(s) + cenário para a Gemini
    e devolve um objeto PIL.Image com a imagem final.
    """

    # Conteúdo enviado para a Gemini: texto + imagem de pessoas + imagem de cenário
    contents = [
        PROMPT,        # Texto com instruções
        person_img,    # Foto com as pessoas (PIL.Image)
        bg_img,        # Cenário (PIL.Image)
    ]

    try:
        # Chamada à Gemini API para gerar imagem a partir de texto + imagens
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]   # Queremos apenas imagem de saída
            ),
        )
    except Exception as e:
        # Se der erro na chamada à Gemini, propagamos como exceção
        raise RuntimeError(f"Erro ao chamar Gemini API: {e}")

    # Agora precisamos pegar os BYTES da imagem de dentro da resposta
    # A resposta costuma vir em response.candidates[x].content.parts[y].inline_data.data
    for candidate in getattr(response, "candidates", []):      # Percorre os candidatos retornados
        content = getattr(candidate, "content", None)
        if content is None:
            continue

        for part in getattr(content, "parts", []):             # Cada "part" pode ser texto ou imagem
            inline_data = getattr(part, "inline_data", None)   # Tenta acessar inline_data (imagem em bytes)
            if inline_data is None:
                continue

            data = getattr(inline_data, "data", None)          # Os bytes reais da imagem
            if not data:
                continue

            try:
                # Converte os bytes em uma imagem PIL
                img = Image.open(BytesIO(data))                # Abre os bytes como imagem
                img.load()                                     # Garante que a imagem seja totalmente carregada
                return img                                     # Devolve a imagem PIL.Image
            except Exception as e:
                raise RuntimeError(f"Erro ao abrir imagem retornada pela Gemini: {e}")

    # Se chegou aqui, não encontramos imagem na resposta
    raise RuntimeError("Resposta da Gemini não contém nenhuma imagem gerada.")


    return final_img                        # Devolve a imagem PIL.Image

# ------------------------------------------------------------
# ROTA PRINCIPAL: /compose
# - Recebe a foto do Unity
# - Redimensiona + prepara a imagem da(s) pessoa(s)
# - Carrega o cenário
# - Chama Gemini com as duas imagens
# - Aplica moldura (Photoshop) se existir
# - Salva imagem final
# - Gera QR code para a URL da imagem
# - Devolve final_url + qr_url para o Unity
# ------------------------------------------------------------
@app.post("/compose")                                       # Define a rota POST /compose
async def compose(request: Request, file: UploadFile = File(...)):  # Função assíncrona que recebe a requisição e um arquivo chamado "file"
    try:                                                    # Tenta executar todo o fluxo, se der erro cai no except geral
        # ----------------------------------------------------
        # 1) Ler bytes da foto enviada pelo Unity
        # ----------------------------------------------------
        raw_bytes = await file.read()                       # Lê os bytes do arquivo enviado pelo cliente
        try:                                                # Tenta abrir esses bytes como imagem
            person_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")  # Abre a imagem em memória e converte para RGB
        except Exception:                                   # Se não conseguir abrir como imagem
            return JSONResponse(                            # Retorna um erro 400 para o cliente
                {"detail": "Arquivo enviado não é uma imagem válida."},
                status_code=400
            )

        # ----------------------------------------------------
        # 2) Reduzir tamanho da foto da(s) pessoa(s) (pra aliviar memória)
        # ----------------------------------------------------
        max_dim = 1024                                      # Define tamanho máximo (em pixels) para largura ou altura
        w, h = person_img.size                              # Pega largura (w) e altura (h) atuais da imagem
        scale = min(max_dim / float(w), max_dim / float(h), 1.0)  # Calcula fator de escala para não ultrapassar 1024
        if scale < 1.0:                                     # Se a imagem for maior que 1024 em alguma dimensão
            new_size = (int(w * scale), int(h * scale))     # Calcula novo tamanho proporcional
            person_img = person_img.resize(new_size, Image.LANCZOS)  # Redimensiona a imagem com filtro de boa qualidade

        # ----------------------------------------------------
        # 3) Carregar o cenário
        # ----------------------------------------------------
        if not os.path.exists(SCENE_FILE):                  # Verifica se o arquivo de cenário existe
            return JSONResponse(                            # Se não existir, retorna erro 500
                {"detail": f"Cenário não encontrado em {SCENE_FILE}"},
                status_code=500
            )

        try:                                                # Tenta abrir o arquivo de cenário
            bg = Image.open(SCENE_FILE).convert("RGB")      # Abre a imagem de cenário e converte para RGB
        except Exception as e:                              # Se der erro ao abrir
            return JSONResponse(                            # Retorna erro 500 com mensagem de erro
                {"detail": f"Erro ao abrir cenário: {e}"},
                status_code=500
            )

        # Redimensionar cenário para 1024x1024
        bg = bg.resize((1024, 1024), Image.LANCZOS)         # Redimensiona a imagem de cenário para 1024x1024

        # ----------------------------------------------------
        # 4) Chamar Gemini para gerar imagem final
        # ----------------------------------------------------
        try:
            final_img = gerar_imagem_gemini(person_img, bg) # Chama função auxiliar que usa Gemini e devolve um PIL.Image
        except Exception as e:
            return JSONResponse(                            # Se der erro, retorna 500 com detalhe
                {"detail": str(e)},
                status_code=500
            )

        # ----------------------------------------------------
        # 5) Aplicar moldura (Photoshop), se existir, e salvar a imagem final
        # ----------------------------------------------------
        img_id = str(uuid.uuid4())                          # Gera um ID único para esta imagem
        foto_filename = f"{img_id}.jpg"                     # Nome do arquivo JPG final
        foto_path = os.path.join(FOTOS_DIR, foto_filename)  # Caminho completo onde será salvo

        try:                                                # Tenta montar a imagem final e salvar
            final_img = final_img.convert("RGBA")           # Garante que está em RGBA (pra poder mexer com alfa)

            # ------------------------------------------------
            # 5.1) Tentar carregar a moldura do Photoshop
            # ------------------------------------------------
            if os.path.exists(FRAME_FILE):                  # Verifica se o arquivo de moldura existe
                frame = Image.open(FRAME_FILE).convert("RGBA")  # Abre a moldura como RGBA (usa o alfa do PNG)

                # Garante que a imagem final tenha o mesmo tamanho da moldura
               # final_img = final_img.resize(frame.size, Image.LANCZOS)  # Redimensiona a imagem final para o tamanho da moldura

                # Cria uma nova imagem transparente para montar tudo
                composed = Image.new("RGBA", frame.size, (0, 0, 0, 0))  # Imagem vazia com transparência

                # Cola a foto final no fundo
                composed.paste(final_img, (25, 270))           # Coloca a foto como fundo (ocupa toda a área)

                # Cola a moldura por cima, usando o próprio alfa da moldura
                composed.paste(frame, (0, 0), frame)        # Coloca a moldura sobre a foto, respeitando a transparência

                # Converte de volta para RGB para salvar em JPG
                final_img = composed.convert("RGB")         # Remove o canal alfa para salvar como JPG
            else:
                # Se não tiver moldura, só converte para RGB normal
                final_img = final_img.convert("RGB")        # Converte a imagem final para RGB (sem alfa)

            # ------------------------------------------------
            # 5.2) Salvar em disco
            # ------------------------------------------------
            final_img.save(foto_path, "JPEG", quality=95)   # Salva a imagem (com ou sem moldura) em JPG, boa qualidade

        except Exception as e:                              # Se qualquer etapa acima falhar
            return JSONResponse(                            # Retorna erro 500
                {"detail": f"Erro ao salvar imagem final: {e}"},
                status_code=500
            )

        # ----------------------------------------------------
        # 6) Montar base_url a partir da requisição (funciona local e na nuvem)
        # ----------------------------------------------------
        base_url = str(request.base_url).rstrip("/")        # Pega a URL base da requisição (ex: https://meuapp.com) e tira a barra do final

        # URL pública da foto final
        foto_url = f"{base_url}/static/fotos/{foto_filename}"  # Monta a URL pública para acessar a foto final

        # ----------------------------------------------------
        # 7) Gerar QR CODE apontando para foto_url
        # ----------------------------------------------------
        qr_img = qrcode.make(foto_url)                      # Gera um QR Code que aponta para a URL da foto final
        qr_filename = f"{img_id}.png"                       # Nome do arquivo do QR Code (mesmo ID da foto)
        qr_path = os.path.join(QR_DIR, qr_filename)         # Caminho completo para salvar o QR

        try:                                                # Tenta salvar o QR Code
            qr_img.save(qr_path)                            # Salva a imagem do QR code em disco
        except Exception as e:                              # Se der erro ao salvar
            return JSONResponse(                            # Retorna erro 500
                {"detail": f"Erro ao salvar QR code: {e}"},
                status_code=500
            )

        qr_url = f"{base_url}/static/qr/{qr_filename}"      # Monta a URL pública do QR Code

        # ----------------------------------------------------
        # 8) Retornar URLs para o Unity
        # ----------------------------------------------------
        return {                                            # Retorna para o Unity um JSON com as duas URLs
            "final_url": foto_url,                          # URL da imagem final (com cenário + moldura)
            "qr_url": qr_url,                               # URL da imagem do QR Code
        }

    except Exception as e:                                  # Captura qualquer erro inesperado no fluxo
        # Erro inesperado
        return JSONResponse({"detail": f"Erro interno: {e}"}, status_code=500)  # Retorna erro 500 genérico
