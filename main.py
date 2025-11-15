import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# ------------------------------
# Hugging Face setup
# ------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

# Model IDs exactly as shown on Hugging Face
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"      # core legal BERT
LEGAL_SUMMARY_MODEL = "nsi319/legal-pegasus"              # legal summarization
LEGAL_NER_MODEL = "opennyaiorg/en_legal_ner_trf"          # legal NER


# ------------------------------
# FastAPI app
# ------------------------------

app = FastAPI(title="BYA Legal AI Enterprise API")


# ------------------------------
# Pydantic models
# ------------------------------

class MaskRequest(BaseModel):
    text: str               # sentence containing [MASK]
    top_k: int = 5          # number of suggestions


class MaskResponseItem(BaseModel):
    sequence: str
    score: float
    token: int
    token_str: str


class SummarizeRequest(BaseModel):
    text: str               # long legal text
    max_length: int = 256   # max length of summary (tokens)


class SummaryResponse(BaseModel):
    summary_text: str


class NERRequest(BaseModel):
    text: str               # legal text


class NEREntity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int


class FullAnalysisRequest(BaseModel):
    text: str               # full contract or document


class FullAnalysisResponse(BaseModel):
    summary: str
    entities: List[NEREntity]
    clause_suggestions: List[MaskResponseItem]


# ------------------------------
# Routes
# ------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "BYA Legal AI API"}


# 1️⃣ Legal-BERT – fill mask
@app.post("/legal/fill-mask", response_model=List[MaskResponseItem])
def fill_mask(req: MaskRequest):
    """
    Predict missing legal terms in clauses using Legal-BERT.
    Example:
    {
      "text": "This Agreement shall be [MASK] by both parties.",
      "top_k": 5
    }
    """
    result = client.fill_mask(
        req.text,
        model=LEGAL_BERT_MODEL,
        top_k=req.top_k,
    )
    return result


# 2️⃣ Legal summarization – Pegasus
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

    # HF may return list[dict] or dict
    if isinstance(result, list):
        result = result[0]

    return SummaryResponse(summary_text=result["summary_text"])


# 3️⃣ Legal NER – entities / clauses
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
            end=e["end"],
        )
        for e in entities
    ]


# 4️⃣ Full enterprise analysis – all in one
@app.post("/legal/analyze", response_model=FullAnalysisResponse)
def analyze(req: FullAnalysisRequest):
    """
    High-level enterprise endpoint:
    - Summarize document
    - Extract entities
    - Generate generic clause suggestions
    """

    # --- summary ---
    summary_result = client.summarization(
        req.text,
        model=LEGAL_SUMMARY_MODEL,
        max_new_tokens=256,
    )
    if isinstance(summary_result, list):
        summary_result = summary_result[0]
    summary_text = summary_result["summary_text"]

    # --- entities ---
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
            end=e["end"],
        )
        for e in raw_entities
    ]

    # --- generic clause suggestions with Legal-BERT ---
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
        clause_suggestions=clause_suggestions,
    )
