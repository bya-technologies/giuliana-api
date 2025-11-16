# === New model IDs for advanced features ===

ADV_DRAFTING_MODEL = "deepseek-ai/DeepSeek-R1"  # or another strong LLM you like
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # for search stubs

# === New Pydantic models ===

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


class OCRPdfRequest(BaseModel):
    file_url: str  # URL to PDF stored in S3/GDrive/etc.


class OCRPdfResponse(BaseModel):
    text: str
    num_pages: Optional[int] = None


# === Helper for generic text generation with an LLM ===

def llm_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Generic wrapper to call a text-generation model safely.
    You can later swap this to any HF-compatible LLM.
    """
    result = client.text_generation(
        prompt,
        model=ADV_DRAFTING_MODEL,
        max_new_tokens=max_new_tokens,
    )
    # some HF endpoints return dict, others str; normalize
    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    if isinstance(result, str):
        return result
    # fallback
    return str(result)


# === New endpoints ===

# 1) Case law search (stubbed – later connect to a vector DB or legal API)
@app.post("/legal/search-caselaw", response_model=CaseLawSearchResponse)
def search_caselaw(req: CaseLawSearchRequest):
    """
    Stub case law search.
    In production, replace with a real vector DB / legal provider.
    """

    # TODO: replace with real retrieval
    demo_case = CaseLawItem(
        title="Example v. Example Co.",
        citation="123 F.3d 456 (9th Cir. 2024)",
        summary=f"Demo precedent related to: {req.query}",
        jurisdiction=req.jurisdiction or "N/A",
    )

    return CaseLawSearchResponse(results=[demo_case])


# 2) Citation formatter
@app.post("/legal/cite", response_model=CiteResponse)
def format_citation(req: CiteRequest):
    """
    Format a simple case citation string.
    """

    parts = [req.case_name]
    if req.court and req.year:
        parts.append(f"{req.court} ({req.year})")
    elif req.year:
        parts.append(f"({req.year})")
    if req.raw_reference:
        parts.append(req.raw_reference)

    citation = ", ".join(parts)
    return CiteResponse(citation=citation)


# 3) NDA drafting
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


# 4) Service agreement drafting
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


# 5) Full legal review report (high-level)
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

Return your answer in a structured JSON-like format with sections:
SUMMARY, KEY_CLAUSES, RED_FLAGS, MISSING_CLAUSES, RECOMMENDATIONS.
"""
    raw = llm_generate(prompt, max_new_tokens=1600)

    # For now, we do a very simple parse: lawyer can read "raw".
    # To keep response predictable, we just wrap raw text into the fields.
    # Later you can post-process with a structured-output model.
    return ReviewReportResponse(
        summary="See LLM output below.",
        key_clauses=[raw[:400]],  # minimal placeholder
        red_flags=[],
        missing_clauses=[],
        recommendations=[raw],
    )


# 6) Missing clauses (lighter endpoint)
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


# 7) Privacy / data protection score
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
Return in a JSON-like format: SCORE, STRENGTHS, ISSUES.
"""
    raw = llm_generate(prompt, max_new_tokens=700)

    # Simple placeholder parsing
    return PrivacyScoreResponse(
        score=70,
        issues=[raw],
        strengths=[],
    )


# 8) Litigation summary
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
        requested_relief="See analysis in full_text field below.",
        timeline=None,
        summary=raw,
    )


# 9) Deposition question generator
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
an attorney could consider using. Do NOT include instructions or commentary,
only the questions.
"""
    raw = llm_generate(prompt, max_new_tokens=700)
    questions = [q.strip("-• ").strip() for q in raw.splitlines() if q.strip()]
    return DepositionQuestionsResponse(questions=questions)


# 10) OCR PDF stub (to be replaced by real OCR)
@app.post("/ocr/pdf", response_model=OCRPdfResponse)
def ocr_pdf(req: OCRPdfRequest):
    """
    Placeholder endpoint for PDF OCR.
    In production, fetch the file from file_url and run an OCR model.
    """

    # TODO: integrate real OCR pipeline (e.g., LayoutLM / Donut / external service)
    demo_text = f"OCR not yet implemented. This is a placeholder for file: {req.file_url}"
    return OCRPdfResponse(text=demo_text, num_pages=None)
