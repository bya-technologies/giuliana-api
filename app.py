import os
import io
import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------- ENV VARS ----------
ORPHEUS_BASETEN_URL = os.getenv("ORPHEUS_BASETEN_URL")
ORPHEUS_BASETEN_API_KEY = os.getenv("ORPHEUS_BASETEN_API_KEY")

# ---------- APP ----------
app = FastAPI(
    title="Giuliana API",
    description="Backend for Giuliana AI concierge (TTS and Image)",
    version="1.0.0",
)

# ---------- MODELS ----------
class TTSRequest(BaseModel):
    text: str
    voice: str | None = "default"

# ---------- HEALTH ----------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "giuliana-api"}

# ---------- ORPHEUS TTS ----------
@app.post("/tts/orpheus")
async def tts_orpheus(req: TTSRequest):
    # (your existing Orpheus TTS code here)
    ...

# ---------- FLUX.1-SCHNELL IMAGE GENERATION ----------
@app.post("/image/flux-baseten")
async def generate_flux_image_baseten(req: dict):
    FLUX_BASETEN_URL = os.getenv("FLUX_BASETEN_URL")
    FLUX_BASETEN_API_KEY = os.getenv("FLUX_BASETEN_API_KEY")

    if not FLUX_BASETEN_URL or not FLUX_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Baseten FLUX model not configured. "
                   "Set FLUX_BASETEN_URL and FLUX_BASETEN_API_KEY.",
        )

    prompt = req.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' field.")

    headers = {
        "Authorization": f"Api-Key {FLUX_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt}

    try:
        r = requests.post(FLUX_BASETEN_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()

        data = r.json()
        image_b64 = data.get("data") or data.get("image")
        if not image_b64:
            raise HTTPException(status_code=500, detail=f"Unexpected Baseten response: {data}")

        image_bytes = base64.b64decode(image_b64)
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=flux_baseten.png"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseten FLUX error: {e}")
