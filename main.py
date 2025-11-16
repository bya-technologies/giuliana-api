# 2️⃣2️⃣ Hospitality – Giuliana Concierge Chat
@app.post("/hospitality/concierge-chat", response_model=ConciergeChatResponse)
def hospitality_concierge_chat(req: ConciergeChatRequest):
    """
    Luxury-style concierge assistant for hotel guests.
    This endpoint generates a natural-language reply that front-desk or
    guest messaging systems can show directly to guests.
    """

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
    """
    Advisory endpoint to suggest room rate strategies.
    This does NOT fetch real pricing data; it provides AI-driven guidance for
    revenue managers to review.
    """

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

    # Minimal parsing to fit RateOption models
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


# 2️⃣4️⃣ Hospitality – Housekeeping Schedule (simple logic)
@app.post("/hospitality/housekeeping/schedule", response_model=HousekeepingScheduleResponse)
def housekeeping_schedule(req: HousekeepingScheduleRequest):
    """
    Simple AI-assisted housekeeping board generator.
    Uses rules + tags (VIP, early check-in) to prioritize rooms.
    """

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
    """
    Analyze guest reviews for sentiment and themes.
    Uses a sentiment model per review, plus an LLM to summarize themes.
    """

    sentiments: List[dict] = []
    pos_count = neg_count = neu_count = 0

    for rev in req.reviews:
        result = client.text_classification(
            rev,
            model=REVIEW_SENTIMENT_MODEL,
        )
        # Expect list[{'label': 'positive'|'neutral'|'negative', 'score': float}]
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

    # Derive average sentiment
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

    # Choose sample positives/negatives (for display in the dashboard)
    positives = []
    negatives = []
    for rev, sent in zip(req.reviews, sentiments):
        label = (sent.get("label") or "").lower()
        if "pos" in label and len(positives) < 3:
            positives.append(rev)
        elif "neg" in label and len(negatives) < 3:
            negatives.append(rev)

    # Ask LLM for themes
    joined_reviews = "\n\n".join(req.reviews[:20])  # limit a bit
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
    """
    High-level AI advisory for maintenance teams.
    Uses LLM reasoning to provide guidelines and recommended actions.
    """

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
            if any(x in val for x in ["low", "medium", "high"]):
                if "low" in val:
                    risk = "low"
                elif "medium" in val:
                    risk = "medium"
                elif "high" in val:
                    risk = "high"
                else:
                    risk = val
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
    """
    AI advisory for F&B teams to optimize menus, pricing, and pairings.
    """

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
    """
    Generate a high-level guest journey or itinerary for a luxury stay.
    Intended to be reviewed/edited by staff before sending to guests.
    """

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
