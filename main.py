import os
from typing import List, Optional

import requests
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
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"          # core legal BERT (fill-mask)
LEGAL_SUMMARY_MODEL = "nsi319/legal-pegasus"                  # legal summarization
LEGAL_NER_MODEL = "opennyaiorg/en_legal_ner_trf"              # legal NER

# Classification models
DEBERTA_ZS_MODEL = "microsoft/deberta-v3-base-mnli"           # contract / risk classification (EN)
MULTILINGUAL_DEBERTA = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # multilingual domain classify

# QA models
MULTILINGUAL_QA_MODEL = "deepset/xlm-roberta-large-squad2"    # multilingual QA

# Drafting / reasoning LLM (swap to any other HF text-gen model if you like)
ADV_DRAFTING_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Embedding model placeholder (future vector search)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Real estate specific models
REAL_ESTATE_LISTING_MODEL = "interneuronai/real_estate_listing_analysis_bart"
FLOORPLAN_VLM_MODEL = "sabaridsnfuji/FloorPlanVisionAIAdaptor"   # currently stubbed

# Hospitality-specific models
REVIEW_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

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

app = FastAPI(title="BYA Legal, Real Estate & Hospitality AI API")

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


def fetch_image_bytes(url: str) -> bytes:
    """
    Simple helper to fetch image bytes from a URL for vision models.
    Currently used only for potential floorplan analysis (stubbed).
    """
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.content


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


# --- Real Estate listing / floorplan ---


class ListingAnalyzeRequest(BaseModel):
    text: str   # property listing text


class ListingAnalyzeResponse(BaseModel):
    category: str
    raw_label_id: int


class FloorPlanAnalyzeRequest(BaseModel):
    image_url: str
    notes: Optional[str] = None


class FloorPlanAnalyzeResponse(BaseModel):
    message: str
    # later you can add structured fields (rooms, area, etc.)


# --- Hospitality / Hotels ---


class ConciergeChatRequest(BaseModel):
    hotel_name: Optional[str] = None
    locale: Optional[str] = None        # e.g. "en-US", "fr-FR"
    guest_profile: Optional[str] = None # VIP, family, business traveler, etc.
    message: str                        # guest question or request


class ConciergeChatResponse(BaseModel):
    reply: str


class RateOption(BaseModel):
    label: str              # e.g. "Standard", "Aggressive", "Conservative"
    recommended_rate: str   # keep as string for now (e.g. "$325")
    explanation: str


class RateOptimizeRequest(BaseModel):
    hotel_name: Optional[str] = None
    date_range: str                 # e.g. "2025-08-01 to 2025-08-05"
    occupancy_level: Optional[str] = None  # e.g. "high", "medium", "low"
    notes: Optional[str] = None     # extra context (events, holidays, etc.)


class RateOptimizeResponse(BaseModel):
    primary_rate: RateOption
    alternatives: List[RateOption]


class HousekeepingScheduleRequest(BaseModel):
    date: str                       # e.g. "2025-08-01"
    rooms: List[str]                # list of room numbers
    vip_rooms: Optional[List[str]] = None
    early_checkin_rooms: Optional[List[str]] = None
    notes: Optional[str] = None


class HousekeepingTask(BaseModel):
    room_number: str
    priority: str                   # "VIP", "EARLY_CHECKIN", "NORMAL"
    estimated_minutes: int
    notes: Optional[str] = None


class HousekeepingScheduleResponse(BaseModel):
    tasks: List[HousekeepingTask]


class ReviewAnalyzeRequest(BaseModel):
    reviews: List[str]              # list of guest reviews


class ReviewSentimentSummary(BaseModel):
    average_sentiment: str          # "positive", "neutral", "negative"
    positives: List[str]
    negatives: List[str]
    themes: List[str]


class ReviewAnalyzeResponse(BaseModel):
    overall: ReviewSentimentSummary
    raw: List[dict]                 # raw sentiment results per review


class MaintenancePredictRequest(BaseModel):
    asset_name: str                 # e.g. "Chiller 3", "Elevator B"
    recent_issue_notes: Optional[str] = None
    usage_pattern: Optional[str] = None     # "heavy usage", "seasonal", etc.


class MaintenancePredictResponse(BaseModel):
    risk_level: str                 # "low", "medium", "high"
    summary: str
    recommended_actions: List[str]


class MenuOptimizeRequest(BaseModel):
    concept: str                    # e.g. "Mediterranean rooftop bar"
    location: str                   # city / region
    menu_items: List[str]           # list of existing dishes
    notes: Optional[str] = None     # dietary / target market / goals


class MenuOptimizeResponse(BaseModel):
    recommended_items: List[str]
    price_tips: List[str]
    pairing_ideas: List[str]


class GuestJourneyRequest(BaseModel):
    hotel_name: Optional[str] = None
    stay_length_nights: int
    guest_profile: str              # e.g. "honeymoon couple", "family with kids"
    preferences: Optional[str] = None   # spa, dining, nightlife, sightseeing


class GuestJourneyResponse(BaseModel):
    itinerary_text: str


# ------------------------------
# Routes
# ------------------------------


@app.get("/")
def root():
    return {"status": "ok", "service": "BYA Legal, Real Estate & Hospitality AI API"}


# 1️⃣ Legal-BERT – fill mask
@app.post("/legal/fill-mask", response_model=List[MaskResponseItem])
def fill_mask(req: MaskRequest):
    result = client.fill_mask(
        req.text,
        model=LEGAL_BERT_MODEL,
        top_k=req.top_k,
    )
    return result


# 2️⃣ Legal summarization – Pegasus
@app.post("/legal/summarize", response_model=SummaryResponse)
def summarize(req: SummarizeRequest):
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
    summary_result = client.summarization(
        req.text,
        model=LEGAL_SUMMARY_MODEL,
        max_new_tokens=256,
    )
    if isinstance(summary_result, list):
        summary_result = summary_result[0]
    summary_text = summary_result["summary_text"]

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
    labels = req.labels or DEFAULT_CONTRACT_LABELS
    result = client.zero_shot_classification(
        req.text,
        labels=labels,
        model=DEBERTA_ZS_MODEL,
        multi_label=False,
    )
    labels_out = result["labels"]
    scores_out = result["scores"]
    best_idx = int(scores_out.index(max(scores_out)))
    return ContractTypeResponse(
        label=labels_out[best_idx],
        score=float(scores_out[best_idx]),
        all_labels=labels_out,
        all_scores=[float(s) for s in scores_out],
    )


# 6️⃣ Real Estate Risk Scoring
@app.post("/realestate/risk-score", response_model=RiskScoreResponse)
def realestate_risk_score(req: RiskScoreRequest):
    result = client.zero_shot_classification(
        req.text,
        labels=DEFAULT_RISK_LABELS,
        model=DEBERTA_ZS_MODEL,
        multi_label=False,
    )
    labels_out = result["labels"]
    scores_out = result["scores"]
    best_idx = int(scores_out.index(max(scores_out)))
    return RiskScoreResponse(
        risk_label=labels_out[best_idx],
        score=float(scores_out[best_idx]),
        all_labels=labels_out,
        all_scores=[float(s) for s in scores_out],
    )


# 7️⃣ Multilingual Legal Domain Classification
@app.post("/legal/domain-classify", response_model=DomainClassifyResponse)
def legal_domain_classify(req: DomainClassifyRequest):
    labels = req.labels or DEFAULT_DOMAIN_LABELS
    result = client.zero_shot_classification(
        req.text,
        labels=labels,
        model=MULTILINGUAL_DEBERTA,
        multi_label=False,
    )
    labels_out = result["labels"]
    scores_out = result["scores"]
    best_idx = int(scores_out.index(max(scores_out)))
    return DomainClassifyResponse(
        label=labels_out[best_idx],
        score=float(scores_out[best_idx]),
        all_labels=labels_out,
        all_scores=[float(s) for s in scores_out],
    )


# 8️⃣ Multilingual Legal Question Answering
@app.post("/legal/qa-multilingual", response_model=LegalQAResponse)
def legal_qa_multilingual(req: LegalQARequest):
    result = client.question_answering(
        question=req.question,
        context=req.context,
        model=MULTILINGUAL_QA_MODEL,
    )
    return LegalQAResponse(
        answer=result["answer"],
        score=float(result["score"]),
        start=int(result["start"]),
        end=int(result["end"]),
    )


# 9️⃣ Legal Redaction / PII Masking
@app.post("/legal/redact", response_model=RedactResponse)
def legal_redact(req: RedactRequest):
    text = req.text
    entities = client.token_classification(
        text,
        model=LEGAL_NER_MODEL,
        aggregation_strategy="simple",
    )

    ent_objs: List[NEREntity] = [
        NEREntity(
            entity_group=e["entity_group"],
            word=e["word"],
            score=float(e["score"]),
            start=int(e["start"]),
            end=int(e["end"]),
        )
        for e in entities
    ]

    ent_sorted = sorted(entities, key=lambda e: int(e["start"]))
    redacted_parts: List[str] = []
    last_idx = 0
    for e in ent_sorted:
        s = int(e["start"])
        e_end = int(e["end"])
        label = e.get("entity_group") or e.get("entity", "ENTITY")
        redacted_parts.append(text[last_idx:s])
        redacted_parts.append(f"[{label}]")
        last_idx = e_end
    redacted_parts.append(text[last_idx:])
    redacted_text = "".join(redacted_parts)

    return RedactResponse(
        redacted_text=redacted_text,
        entities=ent_objs,
    )


# 🔟 Case law search (stub)
@app.post("/legal/search-caselaw", response_model=CaseLawSearchResponse)
def search_caselaw(req: CaseLawSearchRequest):
    demo_case = CaseLawItem(
        title="Example v. Example Co.",
        citation="123 F.3d 456 (9th Cir. 2024)",
        summary=f"Demo precedent related to: {req.query}",
        jurisdiction=req.jurisdiction or "N/A",
    )
    return CaseLawSearchResponse(results=[demo_case])


# 1️⃣1️⃣ Citation formatter
@app.post("/legal/cite", response_model=CiteResponse)
def format_citation(req: CiteRequest):
    parts: List[str] = [req.case_name]
    if req.court and req.year:
        parts.append(f"{req.court} ({req.year})")
    elif req.year:
        parts.append(f"({req.year})")
    if req.raw_reference:
        parts.append(req.raw_reference)
    citation = ", ".join(parts)
    return CiteResponse(citation=citation)


# 1️⃣2️⃣ NDA drafting
@app.post("/legal/draft/nda", response_model=DraftResponse)
def draft_nda(req: NDADraftRequest):
    purpose = req.purpose or "discussing a potential business relationship"
    unilateral_str = "unilateral" if req.unilateral else "mutual"
    prompt = f"""
You are an assistant for lawyers. Draft a {unilateral_str} Non-Disclosure Agreement
between "{req.party_a}" and "{req.party_b}" under the laws of {req.governing_law}.
The term of confidentiality is {req.term_months} months.
Purpose of the NDA: {purpose}.

Include standard sections and clearly mark this as a draft
for attorney review only.
"""
    draft = llm_generate(prompt, max_new_tokens=1200)
    return DraftResponse(draft_text=draft)


# 1️⃣3️⃣ Service agreement drafting
@app.post("/legal/draft/service-agreement", response_model=DraftResponse)
def draft_service_agreement(req: ServiceAgreementDraftRequest):
    prompt = f"""
You are an assistant for law firms. Draft a professional services agreement
between client "{req.client_name}" and provider "{req.provider_name}".
Governing law: {req.governing_law}.

Scope of services: {req.scope}
Fee structure: {req.fee_structure}
Term: {req.term_description}

Include clear limitation of liability, IP ownership, confidentiality,
termination, dispute resolution, and data privacy sections.
Label the output as a draft that must be reviewed and customized by an attorney.
"""
    draft = llm_generate(prompt, max_new_tokens=1400)
    return DraftResponse(draft_text=draft)


# 1️⃣4️⃣ Full legal review report (high-level)
@app.post("/legal/review/report", response_model=ReviewReportResponse)
def review_report(req: ReviewReportRequest):
    prompt = f"""
You are assisting a law firm. Analyze the following contract text
for a high-level review. Jurisdiction: {req.jurisdiction or "unspecified"}.
Contract type: {req.contract_type or "unspecified"}.

Contract text:
\"\"\"{req.text}\"\"\".

1) Provide a 3-5 sentence executive summary.
2) List the 5-10 most important clauses (titles only).
3) List major red flags with severity (info/warning/high) and short description.
4) List important clauses that appear to be missing.
5) Give concise recommendations for the reviewing attorney.

Return your answer as clear, structured bullet points.
"""
    raw = llm_generate(prompt, max_new_tokens=1600)
    return ReviewReportResponse(
        summary="See detailed analysis below.",
        key_clauses=[],
        red_flags=[],
        missing_clauses=[],
        recommendations=[raw],
    )


# 1️⃣5️⃣ Missing clauses (lighter endpoint)
@app.post("/legal/missing-clauses", response_model=MissingClausesResponse)
def missing_clauses(req: ReviewReportRequest):
    prompt = f"""
Given the following contract text, list important clauses that appear to be missing.
Only output a bullet list of clause names. Contract type: {req.contract_type or "unspecified"}.

Text:
\"\"\"{req.text}\"\"\".
"""
    raw = llm_generate(prompt, max_new_tokens=600)
    clauses = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
    return MissingClausesResponse(missing_clauses=clauses)


# 1️⃣6️⃣ Privacy / data protection score
@app.post("/legal/compliance/privacy-score", response_model=PrivacyScoreResponse)
def privacy_score(req: PrivacyScoreRequest):
    prompt = f"""
You are assisting a privacy lawyer. Read the following text and evaluate
how strong the privacy/data-protection language is.

Text:
\"\"\"{req.text}\"\"\".

Rate from 0 (no privacy protection) to 100 (very strong and detailed).
List main strengths and main issues in short bullet points.
Return clearly marked sections: SCORE, STRENGTHS, ISSUES.
"""
    raw = llm_generate(prompt, max_new_tokens=700)
    return PrivacyScoreResponse(
        score=70,
        issues=[raw],
        strengths=[],
    )


# 1️⃣7️⃣ Litigation summary
@app.post("/litigation/summary", response_model=LitigationSummaryResponse)
def litigation_summary(req: LitigationSummaryRequest):
    prompt = f"""
You assist litigation attorneys. Analyze the following pleading/motion text
and extract:
- Parties involved
- Key legal issues
- Relief requested
- Brief timeline of events
- 3-5 sentence summary in plain language.

Text:
\"\"\"{req.text}\"\"\".
"""
    raw = llm_generate(prompt, max_new_tokens=1000)
    return LitigationSummaryResponse(
        parties=[],
        issues=[],
        requested_relief="See analysis in summary field.",
        timeline=None,
        summary=raw,
    )


# 1️⃣8️⃣ Deposition question generator
@app.post("/litigation/deposition-questions", response_model=DepositionQuestionsResponse)
def deposition_questions(req: DepositionQuestionsRequest):
    topics_str = ", ".join(req.topics) if req.topics else "key disputed facts"
    prompt = f"""
You help litigators brainstorm deposition questions.
Case facts summary:
{req.facts_summary}

Witness role: {req.witness_role}
Focus topics: {topics_str}

Draft a list of focused, neutral-sounding deposition questions that
an attorney could consider using. Do NOT include commentary, only the questions.
"""
    raw = llm_generate(prompt, max_new_tokens=700)
    questions = [q.strip("-• ").strip() for q in raw.splitlines() if q.strip()]
    return DepositionQuestionsResponse(questions=questions)


# 1️⃣9️⃣ OCR PDF stub (to be replaced by real OCR)
@app.post("/ocr/pdf", response_model=OCRPdfResponse)
def ocr_pdf(req: OCRPdfRequest):
    demo_text = (
        "OCR not yet implemented. This is a placeholder for file: "
        f"{req.file_url}"
    )
    return OCRPdfResponse(text=demo_text, num_pages=None)


# 2️⃣0️⃣ Real Estate Listing Analysis (classification)
@app.post("/realestate/listing-analyze", response_model=ListingAnalyzeResponse)
def listing_analyze(req: ListingAnalyzeRequest):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = AutoModelForSequenceClassification.from_pretrained(REAL_ESTATE_LISTING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(REAL_ESTATE_LISTING_MODEL)

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = model(**inputs)
    preds = outputs.logits.argmax(-1)
    label_id = preds.item()

    return ListingAnalyzeResponse(
        category=str(label_id),
        raw_label_id=label_id,
    )


# 2️⃣1️⃣ Floorplan AI Stub (no heavy vision yet)
@app.post("/realestate/floorplan-analyze", response_model=FloorPlanAnalyzeResponse)
def floorplan_analyze(req: FloorPlanAnalyzeRequest):
    msg = (
        "Floorplan AI analysis is not fully implemented yet. "
        "Received image URL: " + req.image_url
    )
    if req.notes:
        msg += f" | Notes: {req.notes}"
    return FloorPlanAnalyzeResponse(message=msg)


# 2️⃣2️⃣ Hospitality – Giuliana Concierge Chat
@app.post("/hospitality/concierge-chat", response_model=ConciergeChatResponse)
def hospitality_concierge_chat(req: ConciergeChatRequest):
    hotel = req.hotel_name or "the hotel"
    locale = req.locale or "en-US"
    guest_profile = req.guest_profile or "guest"

    prompt = f"""
You are 'Giuliana', an ultra-luxury hotel concierge AI for {hotel}.
Respond in language/locale: {locale}.
The guest profile is: {guest_profile}.

Be warm, professional, and concise. Provide clear, actionable suggestions.
Do NOT mention that you are an AI model; behave like a concierge assistant.

Guest message:
\"\"\"{req.message}\"\"\".
"""
    reply = llm_generate(prompt, max_new_tokens=600)
    return ConciergeChatResponse(reply=reply)


# 2️⃣3️⃣ Hospitality – Rate Optimization (Advisory)
@app.post("/hospitality/rate-optimize", response_model=RateOptimizeResponse)
def hospitality_rate_optimize(req: RateOptimizeRequest):
    prompt = f"""
You are a revenue management expert for a luxury hotel.

Hotel: {req.hotel_name or "Unnamed Hotel"}
Date range: {req.date_range}
Occupancy level: {req.occupancy_level or "unspecified"}
Context notes: {req.notes or "none"}

Suggest:
1) A primary recommended rate strategy with an example nightly rate.
2) Two alternative strategies (e.g. more aggressive and more conservative),
   each with an example rate and short explanation.

Return the answer in this structured textual format:

PRIMARY:
- Label:
- Rate:
- Explanation:

ALTERNATIVES:
- Label:
- Rate:
- Explanation:
- Label:
- Rate:
- Explanation:
"""
    raw = llm_generate(prompt, max_new_tokens=800)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    primary_label = "Primary"
    primary_rate = "N/A"
    primary_expl = ""
    alternatives: List[RateOption] = []
    current_section = None
    current_alt: dict = {}

    for line in lines:
        upper = line.upper()
        if upper.startswith("PRIMARY"):
            current_section = "PRIMARY"
            continue
        if upper.startswith("ALTERNATIVES"):
            current_section = "ALTERNATIVES"
            continue

        if line.startswith("- Label:"):
            value = line.split(":", 1)[1].strip()
            if current_section == "PRIMARY":
                primary_label = value or primary_label
            elif current_section == "ALTERNATIVES":
                if current_alt:
                    alternatives.append(RateOption(
                        label=current_alt.get("label", "Alternative"),
                        recommended_rate=current_alt.get("rate", "N/A"),
                        explanation=current_alt.get("explanation", ""),
                    ))
                    current_alt = {}
                current_alt["label"] = value or "Alternative"

        elif line.startswith("- Rate:"):
            value = line.split(":", 1)[1].strip()
            if current_section == "PRIMARY":
                primary_rate = value or primary_rate
            elif current_section == "ALTERNATIVES":
                current_alt["rate"] = value or "N/A"

        elif line.startswith("- Explanation:"):
            value = line.split(":", 1)[1].strip()
            if current_section == "PRIMARY":
                primary_expl = value or primary_expl
            elif current_section == "ALTERNATIVES":
                current_alt["explanation"] = value or ""

    if current_alt:
        alternatives.append(RateOption(
            label=current_alt.get("label", "Alternative"),
            recommended_rate=current_alt.get("rate", "N/A"),
            explanation=current_alt.get("explanation", ""),
        ))

    primary = RateOption(
        label=primary_label,
        recommended_rate=primary_rate,
        explanation=primary_expl,
    )

    return RateOptimizeResponse(
        primary_rate=primary,
        alternatives=alternatives,
    )


# 2️⃣4️⃣ Hospitality – Housekeeping Schedule
@app.post("/hospitality/housekeeping/schedule", response_model=HousekeepingScheduleResponse)
def housekeeping_schedule(req: HousekeepingScheduleRequest):
    vip_set = set(req.vip_rooms or [])
    early_set = set(req.early_checkin_rooms or [])

    tasks: List[HousekeepingTask] = []
    for room in req.rooms:
        if room in vip_set:
            priority = "VIP"
            est = 35
            notes = "Prioritize VIP guest; ensure extra attention to detail."
        elif room in early_set:
            priority = "EARLY_CHECKIN"
            est = 30
            notes = "Early check-in expected; clean as early as possible."
        else:
            priority = "NORMAL"
            est = 25
            notes = None
        tasks.append(HousekeepingTask(
            room_number=room,
            priority=priority,
            estimated_minutes=est,
            notes=notes,
        ))

    return HousekeepingScheduleResponse(tasks=tasks)


# 2️⃣5️⃣ Hospitality – Guest Review Analysis
@app.post("/hospitality/reviews/analyze", response_model=ReviewAnalyzeResponse)
def reviews_analyze(req: ReviewAnalyzeRequest):
    sentiments: List[dict] = []
    pos_count = neg_count = neu_count = 0

    for rev in req.reviews:
        result = client.text_classification(
            rev,
            model=REVIEW_SENTIMENT_MODEL,
        )
        if isinstance(result, list) and result:
            best = max(result, key=lambda x: x.get("score", 0.0))
            label = best.get("label", "").lower()
            if "pos" in label:
                pos_count += 1
            elif "neg" in label:
                neg_count += 1
            else:
                neu_count += 1
            sentiments.append(best)
        else:
            sentiments.append({"label": "unknown", "score": 0.0})

    total = pos_count + neg_count + neu_count
    if total == 0:
        avg_sent = "unknown"
    else:
        if pos_count >= max(neg_count, neu_count):
            avg_sent = "positive"
        elif neg_count >= max(pos_count, neu_count):
            avg_sent = "negative"
        else:
            avg_sent = "neutral"

    positives: List[str] = []
    negatives: List[str] = []
    for rev, sent in zip(req.reviews, sentiments):
        label = (sent.get("label") or "").lower()
        if "pos" in label and len(positives) < 3:
            positives.append(rev)
        elif "neg" in label and len(negatives) < 3:
            negatives.append(rev)

    joined_reviews = "\n\n".join(req.reviews[:20])
    theme_prompt = f"""
You are a hospitality consultant. Read these guest reviews and list the 3–7
most important recurring themes (e.g. 'staff friendliness', 'room cleanliness',
'breakfast quality', 'location', etc.). Only output a bullet list.

Reviews:
\"\"\"{joined_reviews}\"\"\".
"""
    themes_raw = llm_generate(theme_prompt, max_new_tokens=400)
    themes = [
        line.strip("-• ").strip()
        for line in themes_raw.splitlines()
        if line.strip()
    ]

    overall = ReviewSentimentSummary(
        average_sentiment=avg_sent,
        positives=positives,
        negatives=negatives,
        themes=themes,
    )

    return ReviewAnalyzeResponse(
        overall=overall,
        raw=sentiments,
    )


# 2️⃣6️⃣ Hospitality – Maintenance & Engineering Prediction
@app.post("/hospitality/maintenance/predict", response_model=MaintenancePredictResponse)
def maintenance_predict(req: MaintenancePredictRequest):
    prompt = f"""
You are a chief engineer at a luxury hotel. Analyze the following asset and
recent maintenance notes, and provide a high-level risk assessment and
recommended actions.

Asset: {req.asset_name}
Usage pattern: {req.usage_pattern or "unspecified"}
Recent issues:
\"\"\"{req.recent_issue_notes or "none provided"}\"\"\".

1) Classify risk as low, medium, or high.
2) Provide a short 2–3 sentence summary.
3) List 3–7 recommended actions as bullets.

Return format:

RISK: <low/medium/high>
SUMMARY: <text>
ACTIONS:
- <action 1>
- <action 2>
...
"""
    raw = llm_generate(prompt, max_new_tokens=700)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    risk = "unknown"
    summary = ""
    actions: List[str] = []
    in_actions = False

    for line in lines:
        upper = line.upper()
        if upper.startswith("RISK:"):
            val = line.split(":", 1)[1].strip().lower()
            if "low" in val:
                risk = "low"
            elif "medium" in val:
                risk = "medium"
            elif "high" in val:
                risk = "high"
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif upper.startswith("ACTIONS"):
            in_actions = True
        elif in_actions and (line.startswith("-") or line.startswith("•")):
            actions.append(line.lstrip("-•").strip())

    if not actions:
        actions.append("Schedule a manual inspection and review recent logs.")

    return MaintenancePredictResponse(
        risk_level=risk,
        summary=summary or "See detailed recommendations above.",
        recommended_actions=actions,
    )


# 2️⃣7️⃣ Hospitality – Menu Optimization
@app.post("/hospitality/menu/optimize", response_model=MenuOptimizeResponse)
def menu_optimize(req: MenuOptimizeRequest):
    items_text = "\n".join(f"- {item}" for item in req.menu_items)
    prompt = f"""
You are an F&B director for a luxury hotel.

Property concept: {req.concept}
Location: {req.location}
Context notes: {req.notes or "none"}

Current menu items:
{items_text}

1) Suggest 3–10 recommended new or improved items suitable for this concept.
2) Provide 3–7 pricing tips (e.g. which items can be premium priced, bundling, etc.).
3) Give 3–7 pairing ideas (wine, cocktails, desserts, sides).

Return in this textual format:

RECOMMENDED_ITEMS:
- ...
PRICE_TIPS:
- ...
PAIRING_IDEAS:
- ...
"""
    raw = llm_generate(prompt, max_new_tokens=900)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    section = None
    rec_items: List[str] = []
    price_tips: List[str] = []
    pairings: List[str] = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("RECOMMENDED_ITEMS"):
            section = "REC"
            continue
        if upper.startswith("PRICE_TIPS"):
            section = "PRICE"
            continue
        if upper.startswith("PAIRING_IDEAS"):
            section = "PAIR"
            continue

        if line.startswith("-"):
            val = line.lstrip("-•").strip()
            if section == "REC":
                rec_items.append(val)
            elif section == "PRICE":
                price_tips.append(val)
            elif section == "PAIR":
                pairings.append(val)

    return MenuOptimizeResponse(
        recommended_items=rec_items,
        price_tips=price_tips,
        pairing_ideas=pairings,
    )


# 2️⃣8️⃣ Hospitality – Guest Journey / Itinerary
@app.post("/hospitality/guest-journey", response_model=GuestJourneyResponse)
def guest_journey(req: GuestJourneyRequest):
    prompt = f"""
You are a luxury hotel concierge creating a tailored stay itinerary.

Hotel: {req.hotel_name or "the hotel"}
Stay length (nights): {req.stay_length_nights}
Guest profile: {req.guest_profile}
Preferences: {req.preferences or "not specified"}

Draft a refined, elegant, and practical itinerary for the entire stay,
including suggestions for:
- Breakfast / dining
- Spa or wellness
- Local experiences
- On-property activities
- Evening options

Use clear day-by-day headings (e.g. "Day 1", "Day 2") and write in a
polished, hospitality-focused tone.
"""
    itinerary = llm_generate(prompt, max_new_tokens=1200)
    return GuestJourneyResponse(itinerary_text=itinerary)
