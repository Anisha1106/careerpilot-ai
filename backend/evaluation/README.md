# Matching Evaluation Data

The repository includes only a small fictional smoke set in `evaluate_matching.py`. It is useful for checking code changes, but it is not a real-world benchmark and must not be reported as model accuracy.

## JSONL Format

Provide one JSON object per line. Each case contains one candidate profile, a set of candidate jobs, and human-assigned relevance grades keyed by job ID:

```json
{
  "case_id": "case-001",
  "profile": {
    "skills": ["Python", "SQL"],
    "projects": ["Built a data reporting tool"],
    "experience": [],
    "education": "Computer science",
    "resume_text": "",
    "preferences": {"roles": ["Data Analyst"], "experience_years": 0}
  },
  "jobs": [
    {
      "id": "job-a",
      "title": "Junior Data Analyst",
      "description": "Analyze data with Python and SQL.",
      "required_skills": ["Python", "SQL"],
      "preferred_skills": [],
      "experience_min": 0
    }
  ],
  "relevance": {"job-a": 2}
}
```

Use grades `0` (not relevant), `1` (weak), `2` (relevant), and `3` (strong). Include all judged jobs in `jobs`; labels omitted from `relevance` are treated as non-relevant. Every case must include at least one positive label.

## Data Collection

- Use candidate information only with informed consent. Remove names, contact details, addresses, and other direct identifiers before evaluation.
- Use job descriptions you are permitted to retain and process. Keep source and collection-date metadata in a separate audit sheet if needed.
- Have at least two reviewers independently label candidate/job relevance using a written rubric; resolve disagreements before producing the JSONL file.
- Keep tuning/development cases separate from a held-out test set. Do not tune scoring weights against the held-out results.
- Keep real candidate data out of Git. The evaluator accepts a local file path so private datasets do not need to be committed.

## Run

Place a private dataset at `backend/evaluation/held-out.jsonl`, then from the repository root run:

```sh
docker compose run --rm -v "$PWD/backend:/app" api python evaluate_matching.py --dataset evaluation/held-out.jsonl --k 3
```

The command reports Precision@1, Precision@K, Recall@K, MRR, and NDCG@K. Metrics describe only the supplied labeled cases; record dataset size, labeling method, split, and date alongside any result you publish.