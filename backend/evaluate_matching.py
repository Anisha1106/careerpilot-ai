"""Evaluate job rankings on labeled JSONL cases or the built-in synthetic smoke set."""

import argparse
import json
import math
from pathlib import Path

from app.matching import score_match


CASES = [
    {
        "name": "backend profile",
        "profile": {
            "skills": ["Python", "FastAPI", "PostgreSQL"],
            "projects": ["Built REST APIs with Python and FastAPI"],
            "experience": ["Backend internship"],
            "education": "Computer science",
            "resume_text": "",
            "preferences": {"roles": ["Backend Engineer"], "experience_years": 1},
        },
        "relevant_job": "backend-api",
        "jobs": [
            {"id": "frontend-ui", "title": "Frontend Engineer", "description": "Build React and TypeScript user interfaces.", "required_skills": ["React", "TypeScript"], "preferred_skills": [], "experience_min": 1},
            {"id": "backend-api", "title": "Backend Engineer", "description": "Build Python APIs with FastAPI and PostgreSQL.", "required_skills": ["Python", "FastAPI", "PostgreSQL"], "preferred_skills": [], "experience_min": 1},
            {"id": "data-analyst", "title": "Data Analyst", "description": "Analyze data with SQL, Excel, and Tableau.", "required_skills": ["SQL", "Tableau"], "preferred_skills": [], "experience_min": 0},
            {"id": "java-mobile", "title": "Android Developer", "description": "Develop Android applications using Java and Kotlin.", "required_skills": ["Java", "Kotlin"], "preferred_skills": [], "experience_min": 2},
        ],
    },
    {
        "name": "data profile",
        "profile": {
            "skills": ["Python", "SQL", "Pandas", "Tableau"],
            "projects": ["Analyzed sales data with Python and pandas"],
            "experience": [],
            "education": "Statistics",
            "resume_text": "",
            "preferences": {"roles": ["Data Analyst"], "experience_years": 0},
        },
        "relevant_job": "data-analyst",
        "jobs": [
            {"id": "backend-api", "title": "Backend Engineer", "description": "Create REST APIs with FastAPI and PostgreSQL.", "required_skills": ["FastAPI", "PostgreSQL"], "preferred_skills": [], "experience_min": 1},
            {"id": "data-analyst", "title": "Data Analyst", "description": "Use SQL, Python, pandas, and Tableau to report business trends.", "required_skills": ["SQL", "Python", "Pandas"], "preferred_skills": ["Tableau"], "experience_min": 0},
            {"id": "frontend-ui", "title": "Frontend Engineer", "description": "Build web interfaces with React, CSS, and TypeScript.", "required_skills": ["React", "CSS"], "preferred_skills": [], "experience_min": 1},
            {"id": "java-mobile", "title": "Android Developer", "description": "Build mobile applications in Java and Kotlin.", "required_skills": ["Java", "Kotlin"], "preferred_skills": [], "experience_min": 2},
        ],
    },
    {
        "name": "frontend profile",
        "profile": {
            "skills": ["JavaScript", "TypeScript", "React", "CSS"],
            "projects": ["Built an accessible React dashboard"],
            "experience": [],
            "education": "Information technology",
            "resume_text": "",
            "preferences": {"roles": ["Frontend Engineer"], "experience_years": 0},
        },
        "relevant_job": "frontend-ui",
        "jobs": [
            {"id": "data-analyst", "title": "Data Analyst", "description": "Analyze datasets with Python, SQL, and Tableau.", "required_skills": ["Python", "SQL"], "preferred_skills": [], "experience_min": 0},
            {"id": "java-mobile", "title": "Android Developer", "description": "Develop Android applications with Java and Kotlin.", "required_skills": ["Java", "Kotlin"], "preferred_skills": [], "experience_min": 1},
            {"id": "frontend-ui", "title": "Frontend Engineer", "description": "Build accessible web interfaces with React, TypeScript, JavaScript, and CSS.", "required_skills": ["React", "TypeScript", "CSS"], "preferred_skills": [], "experience_min": 0},
            {"id": "backend-api", "title": "Backend Engineer", "description": "Build Python services and PostgreSQL APIs.", "required_skills": ["Python", "PostgreSQL"], "preferred_skills": [], "experience_min": 2},
        ],
    },
    {
        "name": "machine learning profile",
        "profile": {
            "skills": ["Python", "scikit-learn", "Pandas", "Statistics"],
            "projects": ["Trained a classification model with scikit-learn"],
            "experience": [],
            "education": "Data science",
            "resume_text": "",
            "preferences": {"roles": ["Machine Learning Engineer"], "experience_years": 0},
        },
        "relevant_job": "ml-engineer",
        "jobs": [
            {"id": "frontend-ui", "title": "Frontend Engineer", "description": "Build React interfaces with JavaScript and CSS.", "required_skills": ["React", "JavaScript"], "preferred_skills": [], "experience_min": 1},
            {"id": "java-mobile", "title": "Android Developer", "description": "Build Android apps in Java and Kotlin.", "required_skills": ["Java", "Kotlin"], "preferred_skills": [], "experience_min": 1},
            {"id": "ml-engineer", "title": "Machine Learning Engineer", "description": "Develop machine learning models using Python, scikit-learn, pandas, and statistics.", "required_skills": ["Python", "scikit-learn", "Pandas"], "preferred_skills": ["Statistics"], "experience_min": 0},
            {"id": "backend-api", "title": "Backend Engineer", "description": "Build Python APIs using FastAPI and PostgreSQL.", "required_skills": ["Python", "FastAPI"], "preferred_skills": [], "experience_min": 2},
        ],
    },
]


def precision_at_k(ranked_ids, relevance, k):
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(relevance.get(item, 0) > 0 for item in ranked_ids[:k]) / k


def recall_at_k(ranked_ids, relevance, k):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_count = sum(value > 0 for value in relevance.values())
    if not relevant_count:
        return 0.0
    found = sum(relevance.get(item, 0) > 0 for item in ranked_ids[:k])
    return found / relevant_count


def reciprocal_rank(ranked_ids, relevance):
    return next(
        (1 / rank for rank, item in enumerate(ranked_ids, start=1) if relevance.get(item, 0) > 0),
        0.0,
    )


def ndcg_at_k(ranked_ids, relevance, k):
    if k <= 0:
        raise ValueError("k must be positive")

    def gain(value, rank):
        return (2 ** value - 1) / math.log2(rank + 1)

    actual = sum(gain(relevance.get(item, 0), rank) for rank, item in enumerate(ranked_ids[:k], start=1))
    ideal_values = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(gain(value, rank) for rank, value in enumerate(ideal_values, start=1))
    return actual / ideal if ideal else 0.0


def case_relevance(case):
    if "relevance" in case:
        relevance = case["relevance"]
        if not isinstance(relevance, dict):
            raise ValueError("relevance must be an object mapping job IDs to grades")
        for level in relevance.values():
            if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 3:
                raise ValueError("relevance grades must be integers from 0 to 3")
        return {str(job_id): level for job_id, level in relevance.items()}
    return {str(case["relevant_job"]): 1}


def validate_case(case):
    if not isinstance(case, dict) or not isinstance(case.get("profile"), dict):
        raise ValueError("Each case must contain a profile object")
    jobs = case.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Each case must contain a non-empty jobs array")
    job_ids = [str(job.get("id", "")) for job in jobs if isinstance(job, dict)]
    if len(job_ids) != len(jobs) or any(not job_id for job_id in job_ids):
        raise ValueError("Every job must be an object with a non-empty id")
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Job IDs must be unique within a case")
    relevance = case_relevance(case)
    if set(relevance) - set(job_ids):
        raise ValueError("Relevance labels must reference job IDs in the case")
    if not any(value > 0 for value in relevance.values()):
        raise ValueError("Each case must label at least one job as relevant")


def rank_case(case):
    scored_jobs = [
        (str(job["id"]), score_match(case["profile"], job)["score"])
        for job in case["jobs"]
    ]
    return [job_id for job_id, _ in sorted(scored_jobs, key=lambda item: item[1], reverse=True)]


def evaluate(cases, k=3, label="synthetic smoke set"):
    if not cases:
        raise ValueError("Dataset must contain at least one labeled case")

    metrics = {"precision_at_1": [], "precision_at_k": [], "recall_at_k": [], "mrr": [], "ndcg_at_k": []}
    for case in cases:
        validate_case(case)
        relevance = case_relevance(case)
        ranked_ids = rank_case(case)
        metrics["precision_at_1"].append(precision_at_k(ranked_ids, relevance, 1))
        metrics["precision_at_k"].append(precision_at_k(ranked_ids, relevance, k))
        metrics["recall_at_k"].append(recall_at_k(ranked_ids, relevance, k))
        metrics["mrr"].append(reciprocal_rank(ranked_ids, relevance))
        metrics["ndcg_at_k"].append(ndcg_at_k(ranked_ids, relevance, k))

    means = {name: sum(values) / len(values) for name, values in metrics.items()}
    print(f"Dataset: {label} ({len(cases)} cases)")
    print(f"Precision@1: {means['precision_at_1']:.3f}")
    print(f"Precision@{k}: {means['precision_at_k']:.3f}")
    print(f"Recall@{k}: {means['recall_at_k']:.3f}")
    print(f"MRR: {means['mrr']:.3f}")
    print(f"NDCG@{k}: {means['ndcg_at_k']:.3f}")
    return means


def load_jsonl(path):
    cases = []
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if line.strip():
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
                if not isinstance(case, dict):
                    raise ValueError(f"Line {line_number} must contain a JSON object")
                cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Path to a labeled JSONL evaluation dataset")
    parser.add_argument("--k", type=int, default=3, help="Cutoff for Precision, Recall, and NDCG (default: 3)")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    if args.dataset:
        cases = load_jsonl(args.dataset)
        label = f"{args.dataset} (externally labeled; verify provenance)"
    else:
        cases = CASES
        label = "built-in fictional smoke set; not real-world evidence"
    evaluate(cases, args.k, label)


if __name__ == "__main__":
    main()