import base64
import requests

ORPHEUS_BASETEN_URL = os.getenv("ORPHEUS_BASETEN_URL")
ORPHEUS_BASETEN_API_KEY = os.getenv("ORPHEUS_BASETEN_API_KEY")


@app.post("/tts/orpheus")
async def tts_orpheus(req: TTSRequest):
    if not ORPHEUS_BASETEN_URL or not ORPHEUS_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Orpheus (Baseten) is not configured.",
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        headers = {
            "Authorization": f"Api-Key {ORPHEUS_BASETEN_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": req.text,
            # adjust keys based on Baseten's Orpheus docs:
            "voice": req.voice or "default",
        }

        r = requests.post(ORPHEUS_BASETEN_URL, json=payload, headers=headers)
        r.raise_for_status()

        # Depending on Baseten setup:
        # 1) If response is raw audio bytes:
        # audio_bytes = r.content
        # 2) If it's JSON with base64:
        data = r.json()
        audio_b64 = data.get("audio") or data.get("data")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
        else:
            # fallback if it's raw bytes
            audio_bytes = r.content

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=orpheus.mp3"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orpheus TTS error: {e}")
