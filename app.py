import os
import io

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")  # optional

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(
    title="Giuliana API",
    description="Backend for Giuliana AI concierge (TTS + utilities).",
    version="1.0.0",
)


class TTSRequest(BaseModel):
    text: str
    voice: str | None = "alloy"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "giuliana-api"}


@app.post("/tts/openai")
async def tts_openai(req: TTSRequest):
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Check OpenAI docs for the latest supported TTS-capable model name.
        result = openai_client.audio.speech.create(
            model="gpt-4.1-mini",  # replace with the recommended TTS/speech model from docs
            voice=req.voice or "alloy",
            input=req.text,
            format="mp3",
        )

        audio_bytes = result.read()
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=giuliana.mp3"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")


@app.post("/tts/minimax")
async def tts_minimax(req: TTSRequest):
    if not MINIMAX_API_KEY:
        raise HTTPException(
            status_code=501,
            detail="MINIMAX_API_KEY not set or Minimax TTS not implemented yet.",
        )

    # Placeholder until you wire Minimax
    return JSONResponse(
        {
            "status": "not_implemented",
            "message": "Minimax TTS integration placeholder.",
        },
        status_code=501,
    )
