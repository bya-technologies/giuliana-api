import os
import io
import base64

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ========== ENV VARS ==========
ORPHEUS_BASETEN_URL = os.getenv("ORPHEUS_BASETEN_URL")
ORPHEUS_BASETEN_API_KEY = os.getenv("ORPHEUS_BASETEN_API_KEY")

FLUX_BASETEN_URL = os.getenv("FLUX_BASETEN_URL")
FLUX_BASETEN_API_KEY = os.getenv("FLUX_BASETEN_API_KEY")

# ========== FASTAPI APP ==========
app = FastAPI(
    title="Giuliana API",
    description="Backend for Giuliana AI concierge (TTS + FLUX images via Baseten).",
    version="1.0.0",
)

# ========== REQUEST MODELS ==========
class TTSRequest(BaseModel):
    text: str
    voice: str | None = "default"


class FluxRequest(BaseModel):
    prompt: str


# ========== HEALTH CHECK ==========
@app.get("/health")
async def health():
    return {"status": "ok", "service": "giuliana-api"}


# ========== ORPHEUS TTS VIA BASETEN ==========
@app.post("/tts/orpheus")
async def tts_orpheus(req: TTSRequest):
    if not ORPHEUS_BASETEN_URL or not ORPHEUS_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Orpheus (Baseten) not configured. Set ORPHEUS_BASETEN_URL and ORPHEUS_BASETEN_API_KEY.",
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    headers = {
        "Authorization": f"Api-Key {ORPHEUS_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": req.text,
        "voice": req.voice or "default",
    }

    try:
        r = requests.post(ORPHEUS_BASETEN_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()

        # Try JSON base64 first; if not, assume raw audio bytes
        audio_bytes = None
        try:
            data = r.json()
            audio_b64 = data.get("audio") or data.get("data")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
        except ValueError:
            pass

        if audio_bytes is None:
            audio_bytes = r.content

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=orpheus.mp3"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orpheus TTS error: {e}")


# ========== FLUX.1-SCHNELL VIA BASETEN ==========
@app.post("/image/flux-baseten")
async def generate_flux_image_baseten(req: FluxRequest):
    if not FLUX_BASETEN_URL or not FLUX_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Baseten FLUX model not configured. Set FLUX_BASETEN_URL and FLUX_BASETEN_API_KEY.",
        )

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    headers = {
        "Authorization": f"Api-Key {FLUX_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"prompt": req.prompt}

    try:
        r = requests.post(FLUX_BASETEN_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()

        # Baseten FLUX deployments often return base64-encoded image in JSON.
        image_bytes = None
        try:
            data = r.json()
            image_b64 = data.get("data") or data.get("image")
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
        except ValueError:
            # Not JSON -> assume raw image bytes
            image_bytes = r.content

        if image_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="No image bytes returned from Baseten FLUX endpoint.",
            )

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=flux_baseten.png"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseten FLUX error: {e}")
