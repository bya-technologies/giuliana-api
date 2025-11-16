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

# Real estate specific models
REAL_ESTATE_LISTING_MODEL = "interneuronai/real_estate_listing_analysis_bart"
# (future) floor plan vision model – to be wired later
FLOORPLAN_VLM_MODEL = "sabaridsnfuji/FloorPlanVisionAIAdaptor"

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


# --- Real Estate Listing Analysis ---

class RealEstateListingRequest(BaseModel):
    text: str  # raw listing or property description text


class RealEstateListingResponse(BaseModel):
    label: str           # predicted category / property type
    score: float         # confidence score
    raw: List[dict]      # full raw output from the model


# --- Real Estate Floor Plan Analysis (stub) ---

class FloorPlanAnalyzeRequest(BaseModel):
    file_url: str           # URL to the floor plan image (S3, GDrive, etc.)
    notes: Optional[str] = None  # optional textual notes or description


class FloorPlanAnalyzeResponse(BaseModel):
    analysis_text: str      # human-readable analysis / placeholder


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
    """
    Quick risk assessment for real estate contracts & deals.
    """

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
    """
    Multilingual zero-shot classifier for legal text.
    Detects if content relates to family, civil, contract, real estate, etc.
    """

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
    """
    Answer a legal question based on the provided contract or document.
    Works across multiple languages.
    """

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
    """
    Redact detected legal entities (names, orgs, locations, etc.)
    from the input text using the LEGAL_NER_MODEL.
    """

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


# 🔟 Case law search (stub for now)
@app.post("/legal/search-caselaw", response_model=CaseLawSearchResponse)
def search_caselaw(req: CaseLawSearchRequest):
    """
    Stub case law search.
    In production, replace with a real vector DB / legal provider.
    """

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
    """
    Format a simple case citation string.
    """

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
    """
    Generate an NDA first draft (for lawyer review, not legal advice).
    """

    purpose = req.purpose or "discussing a potential business relationship"
    unilateral_str = "unilateral" if req.unilateral else "mutual"

    prompt = f"""
You are an assistant for lawyers. Draft a {unilateral_str} Non-Disclosure Agreement
between "{req.party_a}" and "{req.party_b}" under the laws of {req.governing_law}.
The term of confidentiality is {req.term_months} months.
Purpose of the NDA: {purpose}.

Include standard sections: definitions, confidential information, exclusions,
obligations, term and termination, remedies, governing law, and miscellaneous.
Write in clear professional legal English and clearly mark this as a draft
for attorney review only.
"""
    draft = llm_generate(prompt, max_new_tokens=1200)
    return DraftResponse(draft_text=draft)


# 1️⃣3️⃣ Service agreement drafting
@app.post("/legal/draft/service-agreement", response_model=DraftResponse)
def draft_service_agreement(req: ServiceAgreementDraftRequest):
    """
    Generate a service agreement draft for lawyer review.
    """

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
    """
    Generate a structured review for a contract:
    - summary
    - key clauses
    - red flags
    - missing clauses
    - recommendations
    """

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

Return your answer as clear, structured bullet points under the headings:
SUMMARY, KEY_CLAUSES, RED_FLAGS, MISSING_CLAUSES, RECOMMENDATIONS.
"""
    raw = llm_generate(prompt, max_new_tokens=1600)

    # Minimal placeholder: we keep everything in recommendations for now.
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
    """
    Suggest important clauses that appear to be missing.
    """

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
    """
    Heuristic privacy score for data-protection clauses (for attorney review).
    """

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

    # Placeholder: real parsing can be added later
    return PrivacyScoreResponse(
        score=70,
        issues=[raw],
        strengths=[],
    )


# 1️⃣7️⃣ Litigation summary
@app.post("/litigation/summary", response_model=LitigationSummaryResponse)
def litigation_summary(req: LitigationSummaryRequest):
    """
    Summarize a pleading or motion: parties, issues, requested relief, timeline.
    """

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
    """
    Suggest deposition questions based on case facts and witness role.
    """

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
    """
    Placeholder endpoint for PDF OCR.
    In production, fetch the file from file_url and run an OCR model.
    """

    demo_text = (
        "OCR not yet implemented. This is a placeholder for file: "
        f"{req.file_url}"
    )
    return OCRPdfResponse(text=demo_text, num_pages=None)


# 2️⃣0️⃣ Real Estate Listing Analysis
@app.post("/realestate/listing-analyze", response_model=RealEstateListingResponse)
def real_estate_listing_analyze(req: RealEstateListingRequest):
    """
    Analyze real estate listing or property description text.
    Uses a BART-based classifier to categorize the listing and surface its type.
    Example use cases:
    - Classify as apartment / office / retail / new building / etc.
    - Pre-screen listings for brokers, investors, or legal review.
    """

    response = client.text_classification(
        req.text,
        model=REAL_ESTATE_LISTING_MODEL,
    )

    # Hugging Face usually returns a list of {"label": "...", "score": float}
    if isinstance(response, list) and len(response) > 0:
        best = max(response, key=lambda x: x.get("score", 0.0))
    else:
        # fallback in case of unexpected format
        best = {"label": "UNKNOWN", "score": 0.0}

    return RealEstateListingResponse(
        label=best.get("label", "UNKNOWN"),
        score=float(best.get("score", 0.0)),
        raw=response if isinstance(response, list) else [best],
    )


# 2️⃣1️⃣ Real Estate Floor Plan Analysis (stub / future vision integration)
@app.post("/realestate/floorplan-analyze", response_model=FloorPlanAnalyzeResponse)
def floorplan_analyze(req: FloorPlanAnalyzeRequest):
    """
    Placeholder endpoint for floor plan analysis.
    In the next phase, this can be connected to a vision-language model
    (e.g., FloorPlanVisionAIAdaptor) to extract layout, room counts, etc.

    For now, it simply echoes a structured message that can be shown in the UI.
    """

    base_msg = (
        f"Floor plan analysis is in preview. Received file URL: {req.file_url}."
        " In the next release, this endpoint will extract layout details such as"
        " room count, approximate area, and key features using a vision model."
    )
    if req.notes:
        base_msg += f" Notes provided by user: {req.notes}"

    return FloorPlanAnalyzeResponse(analysis_text=base_msg)
