import re
from .ml import text_similarity
from .resume import extract_skills

ALIASES = {
    "js": "javascript", "ts": "typescript", "postgres": "postgresql",
    "ml": "machine learning", "genai": "generative ai", "llms": "llm",
    "sklearn": "scikit-learn", "react.js": "react",
}

def normalize(value: str) -> str:
    text = re.sub(r"[^a-z0-9+#. ]", " ", value.lower())
    phrases = {"natural language processing": "nlp", "large language models": "llm", "react js": "react", "fast api": "fastapi", "scikit learn": "scikit-learn", "nodejs": "node.js"}
    for phrase, canonical in phrases.items():
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", canonical, text)
    return " ".join(ALIASES.get(token, token) for token in text.split())

def score_match(profile: dict, job: dict) -> dict:
    candidate_text = " ".join([
        " ".join(profile.get("skills", [])),
        " ".join(profile.get("projects", [])),
        " ".join(profile.get("experience", [])),
        profile.get("education", ""), profile.get("resume_text", ""),
    ])
    candidate_skills = {normalize(str(skill)) for skill in profile.get("skills", [])}
    candidate_skills.update(normalize(skill) for skill in extract_skills(candidate_text))
    explicit_required = {normalize(str(skill)) for skill in job.get("required_skills", [])}
    preferred = {normalize(str(skill)) for skill in job.get("preferred_skills", [])}
    detected = {normalize(skill) for skill in extract_skills(" ".join([job.get("title", ""), job.get("description", "")]))}
    skill_targets = explicit_required or detected
    matched_required = sorted(explicit_required & candidate_skills)
    missing_required = sorted(explicit_required - candidate_skills)
    matched_detected = sorted(detected & candidate_skills)
    missing_detected = sorted(detected - candidate_skills)
    matched_preferred = sorted(preferred & candidate_skills)
    skill_score = len(skill_targets & candidate_skills) / len(skill_targets) if skill_targets else None
    preferred_score = len(matched_preferred) / len(preferred) if preferred else None

    role_text = " ".join([job.get("title", ""), job.get("description", "")])
    target_roles = " ".join(profile.get("preferences", {}).get("roles", []))
    role_similarity = max(text_similarity(candidate_text, job.get("title", "")), text_similarity(target_roles, job.get("title", "")))
    context_similarity = text_similarity(candidate_text, role_text)

    years = profile.get("preferences", {}).get("experience_years")
    min_years = job.get("experience_min")
    experience_ok = None if years is None or min_years is None else float(years) >= float(min_years)
    components = {}
    weights = {}
    if skill_score is not None:
        components["skill_coverage"] = {"score": round(skill_score * 100)}
        weights["skill_coverage"] = 45
    if preferred_score is not None:
        components["preferred_skills"] = {"score": round(preferred_score * 100)}
        weights["preferred_skills"] = 10
    components["role_similarity"] = {"score": round(role_similarity * 100)}
    components["description_similarity"] = {"score": round(context_similarity * 100)}
    weights["role_similarity"] = 15
    weights["description_similarity"] = 25
    if experience_ok is not None:
        components["experience_fit"] = {"score": 100 if experience_ok else 0}
        weights["experience_fit"] = 5
    total_weight = sum(weights.values())
    weighted_names = list(components)
    assigned_weight = 0.0
    for index, name in enumerate(weighted_names):
        part = components[name]
        part["weight"] = round(100.0 - assigned_weight, 1) if index == len(weighted_names) - 1 else round(weights[name] * 100 / total_weight, 1)
        assigned_weight += part["weight"]
        part["contribution"] = round(part["score"] * part["weight"] / 100, 1)
    score = round(sum(part["contribution"] for part in components.values()))
    eligible = False if missing_required or experience_ok is False else (True if explicit_required else None)
    review = years is None and min_years is not None
    eligibility = "Review experience requirement" if review else ("Required skills are missing" if missing_required else ("Meets listed requirements" if explicit_required and experience_ok is not False else "Review original listing; requirements were inferred"))
    reasons = [
        f"Matches {len(matched_required)} of {len(explicit_required)} structured required skills." if explicit_required else (f"Profile evidence mentions {len(matched_detected)} of {len(detected)} detected skills; confirm which are mandatory in the original listing." if detected else "No recognizable skills were extracted from this listing."),
        f"Profile text is {components['description_similarity']['score']}% similar to this role description using TF-IDF.",
    ]
    if experience_ok is True:
        reasons.append("Experience meets the listed minimum.")
    elif experience_ok is False:
        reasons.append("Listed experience is below the job's minimum; verify the original listing.")
    elif review:
        reasons.append("Experience is not listed in the profile, so this requirement needs review.")
    skill_source = "structured job requirements" if explicit_required else ("skills detected in job title/description" if detected else "no skill requirements detected")
    return {
        "score": score, "eligible": None if review else eligible, "eligibility": eligibility,
        "matching_required_skills": matched_required, "missing_required_skills": missing_required,
        "detected_job_skills": sorted(detected), "matching_detected_skills": matched_detected,
        "missing_detected_skills": missing_detected, "skill_evidence_source": skill_source,
        "matching_preferred_skills": matched_preferred, "experience_ok": experience_ok,
        "score_breakdown": components, "reasons": reasons,
        "recommendations": [f"Add verified evidence for {skill} if you have it, or build a project demonstrating it." for skill in (missing_required or missing_detected)],
        "score_confidence": "higher" if explicit_required and candidate_skills else ("medium" if detected and candidate_skills else "limited: insufficient profile or listing skills"),
        "education_context": "Education is shown as profile context; this listing has no structured education requirement.",
        "method": "NLP skill extraction + TF-IDF cosine similarity + available experience evidence",
    }
