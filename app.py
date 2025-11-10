import os
import io
import base64
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel


# ========= ENV VARS =========
# Baseten - Orpheus TTS
ORPHEUS_BASETEN_URL = os.getenv("ORPHEUS_BASETEN_URL")
ORPHEUS_BASETEN_API_KEY = os.getenv("ORPHEUS_BASETEN_API_KEY")

# Baseten - FLUX.1-schnell (text -> image)
FLUX_BASETEN_URL = os.getenv("FLUX_BASETEN_URL")
FLUX_BASETEN_API_KEY = os.getenv("FLUX_BASETEN_API_KEY")

# Baseten - Stable Video Diffusion or any text/image -> video model (optional)
VIDEO_BASETEN_URL = os.getenv("VIDEO_BASETEN_URL")
VIDEO_BASETEN_API_KEY = os.getenv("VIDEO_BASETEN_API_KEY")

# Optional internal key to protect your API (if set)
GIULIANA_API_KEY = os.getenv("GIULIANA_API_KEY")


# ========= FASTAPI APP =========
app = FastAPI(
    title="Giuliana API",
    description="Backend for Giuliana AI concierge (TTS + Images + Demo Video via Baseten).",
    version="1.0.0",
)

# CORS (adjust origins to your domains when ready)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://luxestay.co", "https://byatechnologies.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= OPTIONAL AUTH MIDDLEWARE =========
@app.middleware("http")
async def verify_giuliana_key(request: Request, call_next):
    """
    If GIULIANA_API_KEY is set, require `x-giuliana-key` for all non-health endpoints.
    If not set, do nothing (open API, good for local/dev).
    """
    if GIULIANA_API_KEY and request.url.path not in ("/health", "/docs", "/openapi.json"):
        client_key = request.headers.get("x-giuliana-key")
        if client_key != GIULIANA_API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    response = await call_next(request)
    return response


# ========= REQUEST MODELS =========
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"


class FluxRequest(BaseModel):
    prompt: str


class VideoFromImageRequest(BaseModel):
    """
    For video pipelines that start from an already-generated image.
    image_b64: base64-encoded image (PNG/JPEG) to animate.
    """
    prompt: Optional[str] = None
    image_b64: str


class VideoDemoRequest(BaseModel):
    """
    High-level demo generator:
    category: "hotel", "bya_design_lab", "sports", "netstreaming", etc.
    prompt: optional custom text. If provided, overrides category template.
    """
    category: Optional[str] = None
    prompt: Optional[str] = None


# ========= HEALTH =========
@app.get("/health")
async def health():
    return {"status": "ok", "service": "giuliana-api"}


# ========= ORPHEUS TTS VIA BASETEN =========
@app.post("/tts/orpheus")
async def tts_orpheus(req: TTSRequest):
    if not ORPHEUS_BASETEN_URL or not ORPHEUS_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Orpheus (Baseten) not configured. "
                   "Set ORPHEUS_BASETEN_URL and ORPHEUS_BASETEN_API_KEY.",
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    headers = {
        "Authorization": f"Api-Key {ORPHEUS_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        # Adjust keys according to your Baseten Orpheus deployment schema.
        "text": req.text,
        "voice": req.voice or "default",
    }

    try:
        r = requests.post(ORPHEUS_BASETEN_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()

        # Try JSON base64 first; if not, assume raw audio bytes.
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


# ========= FLUX.1-SCHNELL VIA BASETEN (TEXT -> IMAGE) =========
@app.post("/image/flux-baseten")
async def generate_flux_image_baseten(req: FluxRequest):
    if not FLUX_BASETEN_URL or not FLUX_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Baseten FLUX model not configured. "
                   "Set FLUX_BASETEN_URL and FLUX_BASETEN_API_KEY.",
        )

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    headers = {
        "Authorization": f"Api-Key {FLUX_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": req.prompt,
        # Add extra parameters here if your Baseten deployment expects them.
    }

    try:
        r = requests.post(FLUX_BASETEN_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()

        # Baseten FLUX deployment may return:
        # - JSON with base64 image
        # - or raw bytes
        image_bytes = None
        try:
            data = r.json()
            image_b64 = data.get("data") or data.get("image")
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
        except ValueError:
            image_bytes = r.content

        if image_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="No image bytes returned from Baseten FLUX endpoint.",
            )

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=flux_schnell.png"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseten FLUX error: {e}")


# ========= VIDEO: FROM IMAGE VIA BASETEN (OPTIONAL) =========
@app.post("/video/from-image")
async def video_from_image(req: VideoFromImageRequest):
    """
    Generic image -> video endpoint using a Baseten deployment
    (e.g., Stable Video Diffusion). Adjust payload/fields to match your model.
    """
    if not VIDEO_BASETEN_URL or not VIDEO_BASETEN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Video model not configured. "
                   "Set VIDEO_BASETEN_URL and VIDEO_BASETEN_API_KEY.",
        )

    headers = {
        "Authorization": f"Api-Key {VIDEO_BASETEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "image": req.image_b64,
        # Add model-specific params here (num_frames, motion, etc.)
    }

    try:
        r = requests.post(VIDEO_BASETEN_URL, json=payload, headers=headers, timeout=300)
        r.raise_for_status()

        # Try JSON base64, else raw bytes.
        video_bytes = None
        try:
            data = r.json()
            video_b64 = data.get("video") or data.get("data")
            if video_b64:
                video_bytes = base64.b64decode(video_b64)
        except ValueError:
            video_bytes = r.content

        if video_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="No video bytes returned from Baseten video endpoint.",
            )

        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={"Content-Disposition": "inline; filename=bya_video.mp4"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation error: {e}")


# ========= VIDEO: HIGH-LEVEL DEMO ENDPOINT =========
@app.post("/video/demo")
async def video_demo(req: VideoDemoRequest):
    """
    High-level helper for demos: BYA Design Lab / Hotel / Sports / NetStreaming.
    This is a template: front-end can call /image/flux-baseten then /video/from-image.
    Here we just build a branded prompt you can log/use upstream.
    """

    # Choose a prompt template if none provided
    category = (req.category or "").lower()
    prompt = req.prompt

    if not prompt:
        if category == "hotel":
            prompt = (
                "Ultra-luxury hotel lobby in Bel Air, cinematic, warm golden light, "
                "BYA LuxeStay style, 8K concept art"
            )
        elif category == "sports":
            prompt = (
                "Dynamic football scene under stadium lights, pro camera angle, "
                "sports documentary style, motion blur"
            )
        elif category in ("netstreaming", "netflix", "streaming"):
            prompt = (
                "Cinematic streaming service hero banner with bold typography and "
                "vibrant gradient lighting, premium look"
            )
        elif category in ("bya_design_lab", "design_lab"):
            prompt = (
                "Futuristic architecture and interior design studio, clean lines, "
                "screens showing layouts, BYA Design Lab branding"
            )
        else:
            prompt = (
                "Luxury cinematic visual with modern aesthetics, suitable for BYA Technologies branding"
            )

    # For now, just return the resolved prompt.
    # Your frontend (or a future chain) can:
    # 1) Call /image/flux-baseten with this prompt
    # 2) Call /video/from-image with the resulting image
    return {
        "category": category or "custom",
        "prompt": prompt,
        "instructions": (
            "Use this prompt with /image/flux-baseten, then send the image_b64 "
            "to /video/from-image to generate a demo video."
        ),
    }
