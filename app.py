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

# ========== FASTAPI APP ==========
app = FastAPI(
    title="Giuliana API",
    description="Backend for Giuliana AI concierge (Orpheus TTS).",
    version="1.0.0",
)

# ========== MODELS ==========
class TTSRequest(BaseModel):
    text: str
    voice: str | None = "default"


# ========== HEALTH CHECK ==========
@app.get("/health")
async def health():
    return {"status": "ok", "service": "giuliana-api"}


# ========== ORPHEUS VIA BASETEN ==========
@app.post("/tts/orpheus")
async def tts_orpheus(req: TTSRequest):
    """
    Call your Orpheus TTS deployment on Baseten and return MP3 audio.
    """
    if not ORPHEUS_BASETEN_URL or not ORPHEUS_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Orpheus (Baseten) is not configured. Set ORPHEUS_BASETEN_URL and ORPHEUS_BASETEN_API_KEY.",
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    headers = {
        "Authorization": f"Api-Key {ORPHEUS_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }

    # Adjust payload keys according to your Baseten Orpheus deployment schema
    payload = {
        "text": req.text,
        "voice": req.voice or "default",
    }

    try:
        r = requests.post(ORPHEUS_BASETEN_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()

        # Try JSON base64 first; if not, treat as raw audio bytes.
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
