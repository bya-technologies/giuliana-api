import os
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient


# ------------------------------
# Hugging Face Setup
# ------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

# Models
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"       # clause prediction
LEGAL_SUMMARY_MODEL = "legal-pegasus"                      # summarization model (replace with exact HF model ID)
LEGAL_NER_MODEL = "en_legal_ner_trf"                       # NER model (replace with exact HF model ID)


# ------------------------------
# FastAPI App
# ------------------------------

app = FastAPI(title="BYA Legal AI Enterprise API")


# ------------------------------
# Request / Response Models
# ------------------------------

class MaskRequest(BaseModel):
    text: str
    top_k: int = 5


class MaskResponseItem(BaseModel):
    sequence: str
    score: float
    token: int
    token_str: str


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 256


class SummaryResponse(BaseModel):
    summary_text: str


class NERRequest(BaseModel):
    text: str


class NEREntity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


class FullAnalysisRequest(BaseModel):
    text: str


class FullAnalysisResponse(BaseModel):
    summary: str
    entities: List[NEREntity]
    clause_suggestions: List[MaskResponseItem]


# ------------------------------
# API Endpoints
# ------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "BYA Legal AI API"}


# ---------- 1️⃣ Legal-BERT Fill Mask ----------
@app.post("/legal/fill-mask", response_model=List[MaskResponseItem])
def fill_mask(req: MaskRequest):
    """
    Predict missing legal terms in clauses using Legal-BERT.
    Example: "This Agreement shall be [MASK] by both parties."
    """
    result = client.fill_mask(
        req.text,
        model=LEGAL_BERT_MODEL,
        top_k=req.top_k,
    )
    return result


# ---------- 2️⃣ Legal Summarization ----------
@app.post("/legal/summarize", response_model=SummaryResponse)
def summarize(req: SummarizeRequest):
    """
    Summarize long contracts or legal documents.
    """
    result = client.summarization(
        req.text,
        model=LEGAL_SUMMARY_MODEL,
        max_new_tokens=req.max_length,
    )

    if isinstance(result, list):
        result = result[0]

    return SummaryResponse(summary_text=result["summary_text"])


# ---------- 3️⃣ Legal Named Entity Recognition ----------
@app.post("/legal/entities", response_model=List[NEREntity])
def extract_entities(req: NERRequest):
    """
    Extract legal entities: PARTY, DATE, MONEY, LAW, LOCATION, etc.
    """
    entities = client.token_classification(
        req.text,
        model=LEGAL_NER_MODEL,
        aggregation_strategy="simple",
    )

    return [
        NEREntity(
            entity_group=e["entity_group"],
            word=e["word"],
            score=e["score"],
            start=e["start"],
            end=e["end"]
        )
        for e in entities
    ]


# ---------- 4️⃣ Full Legal Analysis (All-in-One) ----------
@app.post("/legal/analyze", response_model=FullAnalysisResponse)
def analyze(req: FullAnalysisRequest):
    """
    Enterprise endpoint:
    - Summarize document
    - Extract entities
    - Suggest clause correction
    """

    # 1. Summary
    summary_result = client.summarization(
        req.text,
        model=LEGAL_SUMMARY_MODEL,
        max_new_tokens=256,
    )
    if isinstance(summary_result, list):
        summary_result = summary_result[0]
    summary_text = summary_result["summary_text"]

    # 2. Entities
    raw_entities = client.token_classification(
        req.text,
        model=LEGAL_NER_MODEL,
        aggregation_strategy="simple",
    )
    entities = [
        NEREntity(
            entity_group=e["entity_group"],
            word=e["word"],
            score=e["score"],
            start=e["start"],
            end=e["end"]
        )
        for e in raw_entities
    ]

    # 3. Clause suggestions via Legal-BERT
    clause_prompt = (
        "This Agreement may be [MASK] by the Company at any time without cause."
    )
    suggestions = client.fill_mask(
        clause_prompt,
        model=LEGAL_BERT_MODEL,
        top_k=5,
    )
    clause_suggestions = [MaskResponseItem(**s) for s in suggestions]

    return FullAnalysisResponse(
        summary=summary_text,
        entities=entities,
        clause_suggestions=clause_suggestions
    )
