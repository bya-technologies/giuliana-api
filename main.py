import os
from typing import List, Optional

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

# ------------------------------
# Model IDs
# ------------------------------

# Core legal models
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"      # core legal BERT (fill-mask)
LEGAL_SUMMARY_MODEL = "nsi319/legal-pegasus"              # legal summarization
LEGAL_NER_MODEL = "opennyaiorg/en_legal_ner_trf"          # legal NER

# Classification models
DEBERTA_ZS_MODEL = "microsoft/deberta-v3-base-mnli"       # contract / risk classification (EN)
MULTILINGUAL_DEBERTA = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # multilingual domain classify

# QA models
MULTILINGUAL_QA_MODEL = "deepset/xlm-roberta-large-squad2"        # multilingual QA

# Drafting / reasoning LLM (you can swap this to any HF text-generation model you prefer)
ADV_DRAFTING_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Embedding model placeholder (for future case law search)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Default label sets
DEFAULT_CONTRACT_LABELS = [
    "Lease Agreement",
    "Purchase Agreement",
    "Listing Agreement",
    "Property Management Agreement",
    "Non-Disclosure Agreement",
    "Service Agreement",
    "Employment Contract",
    "Other",
]

DEFAULT_RISK_LABELS = [
    "Low risk",
    "Medium risk",
    "High risk",
]

DEFAULT_DOMAIN_LABELS = [
    "Family law",
    "Civil law",
    "Contract law",
    "Real estate law",
    "Corporate law",
    "Criminal law",
    "Labor law",
    "Immigration law",
]

# ------------------------------
# FastAPI app
# ------------------------------

app = FastAPI(title="BYA Legal & Real Estate AI API")

# ------------------------------
# Pydantic models
# ------------------------------

# --- Core NLP ---

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


# --- Classification ---

class ContractTypeRequest(BaseModel):
    text: str
    labels: Optional[List[str]] = None   # optional custom label set


class ContractTypeResponse(BaseModel):
    label: str
    score: float
    all_labels: List[str]
    all_scores: List[float]


class RiskScoreRequest(BaseModel):
    text: str


class RiskScoreResponse(BaseModel):
    risk_label: str
    score: float
    all_labels: List[str]
    all_scores: List[float]


class DomainClassifyRequest(BaseModel):
    text: str
    labels: Optional[List[str]] = None


class DomainClassifyResponse(BaseModel):
    label: str
    score: float
    all_labels: List[str]
    all_scores: List[float]


# --- QA & Redaction ---

class LegalQARequest(BaseModel):
    question: str
    context: str   # full contract or legal text


class LegalQAResponse(BaseModel):
    answer: str
    score: float
    start: int
    end: int


class RedactRequest(BaseModel):
    text: str      # input legal document text


class RedactResponse(BaseModel):
    redacted_text: str
    entities: List[NEREntity]


# --- Case law & citation ---

class CaseLawSearchRequest(BaseModel):
    query: str
    jurisdiction: Optional[str] = None
    top_k: int = 5


class CaseLawItem(BaseModel):
    title: str
    citation: str
    summary: str
    jurisdiction: Optional[str] = None


class CaseLawSearchResponse(BaseModel):
    results: List[CaseLawItem]


class CiteRequest(BaseModel):
    case_name: str
    court: Optional[str] = None
    year: Optional[int] = None
    raw_reference: Optional[str] = None


class CiteResponse(BaseModel):
    citation: str


# --- Drafting ---

class NDADraftRequest(BaseModel):
    party_a: str
    party_b: str
    governing_law: str = "California"
    term_months: int = 24
    unilateral: bool = True
    purpose: Optional[str] = None


class DraftResponse(BaseModel):
    draft_text: str


class ServiceAgreementDraftRequest(BaseModel):
    client_name: str
    provider_name: str
    governing_law: str = "California"
    scope: str
    fee_structure: str
    term_description: str


# --- Review & compliance ---

class ReviewReportRequest(BaseModel):
    text: str
    jurisdiction: Optional[str] = None
    contract_type: Optional[str] = None


class ReviewIssue(BaseModel):
    title: str
    severity: str  # info / warning / high
    description: str


class ReviewReportResponse(BaseModel):
    summary: str
    key_clauses: List[str]
    red_flags: List[ReviewIssue]
    missing_clauses: List[str]
    recommendations: List[str]


class MissingClausesResponse(BaseModel):
    missing_clauses: List[str]


class PrivacyScoreRequest(BaseModel):
    text: str
    jurisdiction: Optional[str] = None


class PrivacyScoreResponse(BaseModel):
    score: int  # 0-100
    issues: List[str]
    strengths: List[str]


# --- Litigation ---

class LitigationSummaryRequest(BaseModel):
    text: str


class LitigationSummaryResponse(BaseModel):
    parties: List[str]
    issues: List[str]
    requested_relief: str
    timeline: Optional[str] = None
    summary: str


class DepositionQuestionsRequest(BaseModel):
    facts_summary: str
    witness_role: str  # e.g. "plaintiff", "treating physician"
    topics: Optional[List[str]] = None


class DepositionQuestionsResponse(BaseModel):
    questions: List[str]


# --- OCR / PDF ---

class OCRPdfRequest(BaseModel):
    file_url: str  # URL to PDF stored in S3/GDrive/etc.


class OCRPdfResponse(BaseModel):
    text: str
    num_pages: Optional[int] = None


# ------------------------------
# Helper functions
# ------------------------------

def llm_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Generic wrapper to call a text-generation model.
    You can later swap ADV_DRAFTING_MODEL to any HF LLM you like.
    """
    result = client.text_generation(
        prompt,
        model=ADV_DRAFTING_MODEL,
        max_new_tokens=max_new_tokens,
    )

    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    if isinstance(result, str):
        return result
    return str(result)


# ------------------------------
# Routes
# ------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "BYA Legal & Real Estate AI API"}


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
    # HF returns list[dict] – FastAPI will coerce to MaskResponseItem automatically
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
            score=float(e["score"]),
            start=int(e["start"]),
            end=int(e["end"]),
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

    # summary
    summary_result = client.summarization(
        req.text,
        model=LEGAL_SUMMARY_MODEL,
        max_new_tokens=256,
    )
    if isinstance(summary_result, list):
        summary_result = summary_result[0]
    summary_text = summary_result["summary_text"]

    # entities
    raw_entities = client.token_classification(
        req.text,
        model=LEGAL_NER_MODEL,
        aggregation_strategy="simple",
    )
    entities = [
        NEREntity(
            entity_group=e["entity_group"],
            word=e["word"],
            score=float(e["score"]),
            start=int(e["start"]),
            end=int(e["end"]),
        )
        for e in raw_entities
    ]

    # clause suggestions via Legal-BERT
    clause_prompt = "This Agreement may be [MASK] by the Company at any time without cause."
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


# 5️⃣ Contract Type Classification (Legal + Real Estate)
@app.post("/legal/contract-type", response_model=ContractTypeResponse)
def classify_contract_type(req: ContractTypeRequest):
    """
    Classify what type of contract this is using DeBERTa zero-shot classification.
    """

    labels = req.labels or DEFAULT_CONTRACT_LABELS

    result = client.zero_shot_classification(
        req.text,
        labels=labels,
        model=DEBERTA_ZS_MODEL,
        multi_label=False,
    )

    labels_out = result["labels"]
    scores_out = result["scores"]
    best_idx = int(scores_out
