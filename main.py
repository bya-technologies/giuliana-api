# 8️⃣ Multilingual Legal Domain Classification
class DomainClassifyRequest(BaseModel):
    text: str
    labels: Optional[List[str]] = None


class DomainClassifyResponse(BaseModel):
    label: str
    score: float
    all_labels: List[str]
    all_scores: List[float]


MULTILINGUAL_DEBERTA = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@app.post("/legal/domain-classify", response_model=DomainClassifyResponse)
def legal_domain_classify(req: DomainClassifyRequest):
    """
    Multilingual zero-shot classifier for legal text.
    Detects if content relates to family, civil, contract, real estate, criminal law, etc.
    Works with Arabic, French, English, Spanish, German...
    """

    labels = req.labels or [
        "Family law",
        "Civil law",
        "Contract law",
        "Real estate law",
        "Corporate law",
        "Criminal law",
        "Labor law",
        "Immigration law"
    ]

    result = client.zero_shot_classification(
        req.text,
        labels=labels,
        model=MULTILINGUAL_DEBERTA,
        multi_label=False
    )

    labels_out = result["labels"]
    scores_out = result["scores"]
    best_idx = scores_out.index(max(scores_out))

    return DomainClassifyResponse(
        label=labels_out[best_idx],
        score=scores_out[best_idx],
        all_labels=labels_out,
        all_scores=scores_out
    )
