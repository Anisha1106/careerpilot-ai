import os
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import psycopg
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from psycopg.types.json import Jsonb

from .ai import answer, chunks, cosine, draft, embed
from .db import DATABASE_URL, get_db
from .matching import normalize, score_match
from .ml import rank_texts
from .resume import analyze, extract, extract_skills
from .security import create_token, hash_password, token_user_id, verify_password

app = FastAPI(title="CareerPilot AI", version="2.0.0", description="Explainable job matching and evidence-based career guidance.")
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type"])

SCHEMA = [
"""
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS profiles (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    full_name TEXT NOT NULL DEFAULT '', education TEXT NOT NULL DEFAULT '', degree TEXT NOT NULL DEFAULT '',
    specialization TEXT NOT NULL DEFAULT '', graduation_year INTEGER,
    skills JSONB NOT NULL DEFAULT '[]', projects JSONB NOT NULL DEFAULT '[]',
    experience JSONB NOT NULL DEFAULT '[]', preferences JSONB NOT NULL DEFAULT '{}',
    resume_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS jobs (
    id BIGSERIAL PRIMARY KEY,
    owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL, company TEXT NOT NULL DEFAULT 'Not specified', location TEXT NOT NULL DEFAULT 'Not specified',
    description TEXT NOT NULL DEFAULT '', url TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT 'User supplied',
    source_checked_at TIMESTAMPTZ NOT NULL DEFAULT now(), required_skills JSONB NOT NULL DEFAULT '[]',
    preferred_skills JSONB NOT NULL DEFAULT '[]', experience_min REAL,
    employment_type TEXT NOT NULL DEFAULT 'Not specified', created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS applications (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'Saved', application_date TIMESTAMPTZ, recruiter TEXT NOT NULL DEFAULT '',
    interview_date TIMESTAMPTZ, follow_up_date TIMESTAMPTZ, notes TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(user_id, job_id)
);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source TEXT NOT NULL, content TEXT NOT NULL, embedding JSONB
);
""",
"CREATE INDEX IF NOT EXISTS knowledge_chunks_user_idx ON knowledge_chunks(user_id)",
"ALTER TABLE users ALTER COLUMN created_at SET DEFAULT now()",
"ALTER TABLE profiles ALTER COLUMN full_name SET DEFAULT ''",
"ALTER TABLE profiles ALTER COLUMN education SET DEFAULT ''",
"ALTER TABLE profiles ALTER COLUMN degree SET DEFAULT ''",
"ALTER TABLE profiles ALTER COLUMN specialization SET DEFAULT ''",
"ALTER TABLE profiles ALTER COLUMN skills SET DEFAULT '[]'::jsonb",
"ALTER TABLE profiles ALTER COLUMN projects SET DEFAULT '[]'::jsonb",
"ALTER TABLE profiles ALTER COLUMN experience SET DEFAULT '[]'::jsonb",
"ALTER TABLE profiles ALTER COLUMN preferences SET DEFAULT '{}'::jsonb",
"ALTER TABLE profiles ALTER COLUMN resume_text SET DEFAULT ''",
"ALTER TABLE jobs ALTER COLUMN created_at SET DEFAULT now()",
"CREATE INDEX IF NOT EXISTS jobs_owner_idx ON jobs(owner_id)",
"CREATE INDEX IF NOT EXISTS applications_user_idx ON applications(user_id)",
]

@app.on_event("startup")
def startup():
    with psycopg.connect(DATABASE_URL) as connection:
        for statement in SCHEMA:
            connection.execute(statement)

def current_user(authorization: str | None = Header(default=None), db=Depends(get_db)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in to continue")
    user_id = token_user_id(authorization.split(" ", 1)[1])
    user = db.execute("SELECT id, email FROM users WHERE id = %s", (user_id,)).fetchone() if user_id else None
    if not user:
        raise HTTPException(401, "Invalid or expired session")
    return user

async def read_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Send a valid JSON object")
    if not isinstance(body, dict):
        raise HTTPException(400, "Send a JSON object")
    return body

def require_text(body: dict, key: str, maximum: int = 4000, required: bool = False) -> str:
    value = body.get(key, "")
    if value is None:
        value = ""
    if not isinstance(value, str) or len(value) > maximum or (required and not value.strip()):
        raise HTTPException(422, f"{key} must be a text value up to {maximum} characters" + (" and cannot be empty" if required else ""))
    return value.strip()

def list_of_strings(value, field):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise HTTPException(422, f"{field} must be a list of text values")
    return [item.strip() for item in value if item.strip()][:100]

def parse_date(value):
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise HTTPException(422, "Dates must be ISO-formatted text")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, "Dates must be ISO-formatted text")

def profile_data(profile):
    profile = dict(profile)
    profile["completion"] = profile_completion(profile)
    return profile

def profile_completion(profile):
    checks = [bool(profile.get("full_name")), bool(profile.get("education") or profile.get("degree")), bool(profile.get("skills")), bool(profile.get("projects")), bool(profile.get("preferences", {}).get("roles")), bool(profile.get("preferences", {}).get("locations"))]
    return round(100 * sum(checks) / len(checks))

def get_profile(db, user_id):
    profile = db.execute("SELECT * FROM profiles WHERE user_id = %s", (user_id,)).fetchone()
    if not profile:
        raise HTTPException(404, "Candidate profile not found")
    return profile

def job_data(job):
    return dict(job)

def index_knowledge(db, user_id, source, content):
    db.execute("DELETE FROM knowledge_chunks WHERE user_id = %s AND source = %s", (user_id, source))
    for part in chunks(content):
        vector = embed(part)
        db.execute("INSERT INTO knowledge_chunks (user_id, source, content, embedding) VALUES (%s, %s, %s, %s)", (user_id, source, part, Jsonb(vector) if vector else None))

def search_saved_jobs(db, user_id, profile, query, location):
    rows = db.execute("SELECT * FROM jobs WHERE owner_id IS NULL OR owner_id=%s ORDER BY created_at DESC LIMIT 100", (user_id,)).fetchall()
    query = query.strip().lower()
    location = location.strip().lower()
    results = []
    for row in rows:
        searchable = f"{row['title']} {row['company']} {row['description']}".lower()
        if query and query not in searchable:
            continue
        if location and location not in (row["location"] or "").lower():
            continue
        results.append({**job_data(row), "match": score_match(profile, row)})
    results.sort(key=lambda result: result["match"]["score"], reverse=True)
    return results[:20]

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/auth/register")
async def register(request: Request, db=Depends(get_db)):
    body = await read_body(request)
    email = require_text(body, "email", 320, True).lower()
    password = require_text(body, "password", 128, True)
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(422, "Enter a valid email address")
    if len(password) < 10:
        raise HTTPException(422, "Password must be at least 10 characters")
    if db.execute("SELECT 1 FROM users WHERE email = %s", (email,)).fetchone():
        raise HTTPException(409, "An account with this email already exists")
    try:
        user = db.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, email", (email, hash_password(password))).fetchone()
        db.execute("INSERT INTO profiles (user_id) VALUES (%s)", (user["id"],))
        db.commit()
    except psycopg.errors.UniqueViolation:
        db.rollback()
        raise HTTPException(409, "An account with this email already exists")
    return {"token": create_token(user["id"]), "user": user}

@app.post("/api/auth/login")
async def login(request: Request, db=Depends(get_db)):
    body = await read_body(request)
    email = require_text(body, "email", 320, True).lower()
    password = require_text(body, "password", 128, True)
    user = db.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (email,)).fetchone()
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Email or password is incorrect")
    return {"token": create_token(user["id"]), "user": {"id": user["id"], "email": user["email"]}}

@app.get("/api/me")
def me(user=Depends(current_user)):
    return user

@app.get("/api/profile")
def get_my_profile(user=Depends(current_user), db=Depends(get_db)):
    return profile_data(get_profile(db, user["id"]))

@app.put("/api/profile")
async def update_profile(request: Request, user=Depends(current_user), db=Depends(get_db)):
    body = await read_body(request)
    existing = get_profile(db, user["id"])
    fields = ["full_name", "education", "degree", "specialization"]
    values = [require_text(body, field, 4000) for field in fields]
    graduation_year = body.get("graduation_year") or None
    if graduation_year is not None and (not isinstance(graduation_year, int) or graduation_year < 1950 or graduation_year > 2100):
        raise HTTPException(422, "graduation_year must be a valid year")
    skills = list_of_strings(body.get("skills", existing["skills"]), "skills")
    projects = list_of_strings(body.get("projects", existing["projects"]), "projects")
    experience = list_of_strings(body.get("experience", existing["experience"]), "experience")
    preferences = body.get("preferences", existing["preferences"])
    if not isinstance(preferences, dict):
        raise HTTPException(422, "preferences must be an object")
    db.execute("UPDATE profiles SET full_name=%s, education=%s, degree=%s, specialization=%s, graduation_year=%s, skills=%s, projects=%s, experience=%s, preferences=%s WHERE user_id=%s", (*values, graduation_year, Jsonb(skills), Jsonb(projects), Jsonb(experience), Jsonb(preferences), user["id"]))
    index_knowledge(db, user["id"], "Candidate profile", " ".join([*skills, *projects, *experience, *values]))
    db.commit()
    return profile_data(get_profile(db, user["id"]))

@app.post("/api/resume")
async def upload_resume(file: UploadFile = File(...), user=Depends(current_user), db=Depends(get_db)):
    raw = await file.read(10_000_001)
    if len(raw) > 10_000_000:
        raise HTTPException(413, "Resume must be under 10 MB")
    try:
        text = extract(file.filename or "", raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, "Could not parse this document")
    if not text:
        raise HTTPException(400, "No readable text was found in the resume")
    result = analyze(text)
    profile = get_profile(db, user["id"])
    skills = sorted(set(profile["skills"] or []) | set(result["skills"]))
    db.execute("UPDATE profiles SET resume_text=%s, skills=%s WHERE user_id=%s", (text, Jsonb(skills), user["id"]))
    index_knowledge(db, user["id"], "Resume", text)
    db.commit()
    result.pop("extracted_text", None)
    result["profile"] = profile_data(get_profile(db, user["id"]))
    return result

def validate_job_link(url):
    if url:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise HTTPException(400, "Job link must be a valid HTTPS URL")

@app.post("/api/jobs")
async def add_job(request: Request, user=Depends(current_user), db=Depends(get_db)):
    body = await read_body(request)
    title = require_text(body, "title", 250, True)
    company = require_text(body, "company", 250) or "Not specified"
    location = require_text(body, "location", 250) or "Not specified"
    description = require_text(body, "description", 30000)
    url = require_text(body, "url", 2000)
    validate_job_link(url)
    required = list_of_strings(body.get("required_skills", []), "required_skills")
    preferred = list_of_strings(body.get("preferred_skills", []), "preferred_skills")
    experience_min = body.get("experience_min")
    if experience_min is not None and (not isinstance(experience_min, (int, float)) or experience_min < 0):
        raise HTTPException(422, "experience_min must be a non-negative number")
    source = require_text(body, "source", 80) or "User supplied"
    if source not in {"User supplied", "Adzuna"}:
        raise HTTPException(422, "Unsupported listing source")
    checked_at = parse_date(body.get("source_checked_at")) or datetime.now(timezone.utc)
    existing = db.execute("SELECT * FROM jobs WHERE owner_id=%s AND url=%s AND url <> ''", (user["id"], url)).fetchone() if url else None
    if existing:
        return {**job_data(existing), "match": score_match(get_profile(db, user["id"]), existing), "already_saved": True}
    job = db.execute("INSERT INTO jobs (owner_id,title,company,location,description,url,source,source_checked_at,required_skills,preferred_skills,experience_min,employment_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *", (user["id"], title, company, location, description, url, source, checked_at, Jsonb(required), Jsonb(preferred), experience_min, require_text(body, "employment_type", 80) or "Not specified")).fetchone()
    content = f"Title: {title}. Company: {company}. Location: {location}. Required skills: {', '.join(required)}. Preferred skills: {', '.join(preferred)}. Description: {description}"
    index_knowledge(db, user["id"], f"Job {job['id']}: {title}", content)
    db.commit()
    profile = get_profile(db, user["id"])
    return {**job_data(job), "match": score_match(profile, job)}

@app.get("/api/jobs")
def list_jobs(q: str = "", location: str = "", user=Depends(current_user), db=Depends(get_db)):
    rows = db.execute("SELECT * FROM jobs WHERE owner_id IS NULL OR owner_id=%s ORDER BY created_at DESC LIMIT 100", (user["id"],)).fetchall()
    profile = get_profile(db, user["id"])
    results = []
    for row in rows:
        text = f"{row['title']} {row['company']} {row['description']}"
        if q and q.lower() not in text.lower():
            continue
        if location and location.lower() not in row["location"].lower():
            continue
        results.append({**job_data(row), "match": score_match(profile, row)})
    return results

@app.get("/api/jobs/discover")
async def discover(q: str = "", location: str = "", user=Depends(current_user), db=Depends(get_db)):
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    profile = get_profile(db, user["id"])
    search_query = q.strip() or (profile["preferences"].get("roles", [""])[0] if profile["preferences"].get("roles") else "entry-level machine learning engineer")
    if not app_id or not app_key:
        saved = search_saved_jobs(db, user["id"], profile, search_query, location)
        return {"available": False, "message": "Live search needs Adzuna credentials. Showing matching saved jobs instead." if saved else "Live search needs Adzuna credentials. Add a job manually, then search your saved jobs.", "jobs": saved, "search_query": search_query, "ranking_method": "Saved-job keyword search + TF-IDF candidate/profile match"}
    try:
        country = os.getenv("ADZUNA_COUNTRY", "in").lower()
        async with httpx.AsyncClient(timeout=12) as http:
            response = await http.get(f"https://api.adzuna.com/v1/api/jobs/{country}/search/1", params={"app_id": app_id, "app_key": app_key, "what": search_query, "where": location, "results_per_page": 20, "sort_by": "date"})
            response.raise_for_status()
        now = datetime.now(timezone.utc).isoformat()
        jobs = []
        for item in response.json().get("results", []):
            posted_at = item.get("created") or now
            job = {"title": item.get("title", "Role not specified"), "company": item.get("company", {}).get("display_name", "Not specified"), "location": item.get("location", {}).get("display_name", "Not specified"), "description": item.get("description", ""), "url": item.get("redirect_url", ""), "source": "Adzuna", "source_checked_at": now, "posted_at": posted_at, "required_skills": [], "preferred_skills": [], "experience_min": None, "employment_type": "Not specified"}
            jobs.append({**job, "match": score_match(profile, job)})
        jobs.sort(key=lambda item: item["match"]["score"], reverse=True)
        return {"available": True, "jobs": jobs, "search_query": search_query, "ranking_method": "NLP skill extraction + TF-IDF candidate/profile match"}
    except httpx.HTTPError:
        saved = search_saved_jobs(db, user["id"], profile, search_query, location)
        return {"available": False, "message": "Adzuna could not be reached. Showing matching saved jobs instead." if saved else "Adzuna could not be reached. Try again later or search your saved jobs.", "jobs": saved, "search_query": search_query, "ranking_method": "Saved-job keyword search + TF-IDF candidate/profile match"}

@app.get("/api/jobs/{job_id:int}")
def get_job(job_id: int, user=Depends(current_user), db=Depends(get_db)):
    job = db.execute("SELECT * FROM jobs WHERE id=%s AND (owner_id IS NULL OR owner_id=%s)", (job_id, user["id"])).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")
    return {**job_data(job), "match": score_match(get_profile(db, user["id"]), job)}

@app.post("/api/applications/{job_id}")
async def save_application(job_id: int, request: Request, user=Depends(current_user), db=Depends(get_db)):
    body = await read_body(request)
    job = db.execute("SELECT * FROM jobs WHERE id=%s AND (owner_id IS NULL OR owner_id=%s)", (job_id, user["id"])).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")
    status = body.get("status", "Saved")
    allowed = {"Saved", "Preparing", "Applied", "Assessment", "Interview", "Offer", "Rejected", "Withdrawn"}
    if status not in allowed:
        raise HTTPException(422, "Invalid application status")
    existing = db.execute("SELECT * FROM applications WHERE user_id=%s AND job_id=%s", (user["id"], job_id)).fetchone()
    application_date = parse_date(body.get("application_date", existing["application_date"] if existing else None))
    interview_date = parse_date(body.get("interview_date", existing["interview_date"] if existing else None))
    follow_up_date = parse_date(body.get("follow_up_date", existing["follow_up_date"] if existing else None))
    recruiter = require_text(body, "recruiter", 250)
    notes = require_text(body, "notes", 10000)
    if existing:
        recruiter = body.get("recruiter", existing["recruiter"])
        notes = body.get("notes", existing["notes"])
    if status == "Applied" and application_date is None:
        application_date = datetime.now(timezone.utc)
    item = db.execute("INSERT INTO applications (user_id,job_id,status,application_date,recruiter,interview_date,follow_up_date,notes,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now()) ON CONFLICT (user_id,job_id) DO UPDATE SET status=EXCLUDED.status, application_date=EXCLUDED.application_date, recruiter=EXCLUDED.recruiter, interview_date=EXCLUDED.interview_date, follow_up_date=EXCLUDED.follow_up_date, notes=EXCLUDED.notes, updated_at=now() RETURNING *", (user["id"], job_id, status, application_date, recruiter, interview_date, follow_up_date, notes)).fetchone()
    db.commit()
    return application_data(item, job)

def application_data(item, job):
    return {"id": item["id"], "job_id": job["id"], "title": job["title"], "company": job["company"], "url": job["url"], "status": item["status"], "application_date": item["application_date"], "recruiter": item["recruiter"], "interview_date": item["interview_date"], "follow_up_date": item["follow_up_date"], "notes": item["notes"], "updated_at": item["updated_at"]}

@app.get("/api/applications")
def list_applications(user=Depends(current_user), db=Depends(get_db)):
    rows = db.execute("SELECT a.*, j.title, j.company, j.url FROM applications a JOIN jobs j ON j.id=a.job_id WHERE a.user_id=%s ORDER BY a.updated_at DESC", (user["id"],)).fetchall()
    return [{"id": row["id"], "job_id": row["job_id"], "title": row["title"], "company": row["company"], "url": row["url"], **{key: row[key] for key in ["status", "application_date", "recruiter", "interview_date", "follow_up_date", "notes", "updated_at"]}} for row in rows]

@app.get("/api/dashboard")
def dashboard(user=Depends(current_user), db=Depends(get_db)):
    profile = get_profile(db, user["id"])
    jobs = db.execute("SELECT * FROM jobs WHERE owner_id IS NULL OR owner_id=%s ORDER BY created_at DESC LIMIT 30", (user["id"],)).fetchall()
    applications = db.execute("SELECT status FROM applications WHERE user_id=%s", (user["id"],)).fetchall()
    statuses = dict(Counter(row["status"] for row in applications))
    best = sorted([{**job_data(job), "match": score_match(profile, job)} for job in jobs], key=lambda item: item["match"]["score"], reverse=True)[:5]
    return {"profile_completion": profile_completion(profile), "resume_uploaded": bool(profile["resume_text"]), "total_applications": len(applications), "applications_by_status": statuses, "interviews": statuses.get("Interview", 0), "offers": statuses.get("Offer", 0), "best_matches": best}

@app.post("/api/assistant")
async def assistant(request: Request, user=Depends(current_user), db=Depends(get_db)):
    body = await read_body(request)
    question = require_text(body, "question", 2000, True)
    rows = db.execute("SELECT source,content,embedding FROM knowledge_chunks WHERE user_id=%s", (user["id"],)).fetchall()
    if not rows:
        profile = get_profile(db, user["id"])
        content = " ".join([*profile["skills"], *profile["projects"], *profile["experience"], profile["education"], profile["resume_text"]])
        rows = [{"source": "Candidate profile and resume", "content": part, "embedding": None} for part in chunks(content)]
    query_embedding = embed(question)
    if query_embedding and rows and all(row["embedding"] for row in rows):
        ranked = sorted([(cosine(query_embedding, row["embedding"]), row) for row in rows], key=lambda item: item[0], reverse=True)[:5]
        method = "PyTorch cosine similarity"
    else:
        ranked_indexes = rank_texts(question, [row["content"] for row in rows])
        ranked = [(score, rows[index]) for index, score in ranked_indexes[:5]]
        method = "Local TF-IDF cosine similarity"
    contexts = [{"source": row["source"], "content": row["content"]} for score, row in ranked if score > 0]
    evidence = [{"source": row["source"], "similarity": round(float(score), 3)} for score, row in ranked if score > 0]
    return {**answer(question, contexts), "retrieval_method": method, "retrieved_sources": evidence}

@app.post("/api/drafts")
async def create_draft(request: Request, user=Depends(current_user), db=Depends(get_db)):
    body = await read_body(request)
    job_id = body.get("job_id")
    if not isinstance(job_id, int):
        raise HTTPException(422, "job_id must be an integer")
    kind = body.get("kind", "cover letter")
    allowed = {"cover letter", "recruiter message", "referral request", "LinkedIn connection message", "application answer", "resume summary", "resume tailoring"}
    if kind not in allowed:
        raise HTTPException(422, "Choose a supported draft type")
    job = db.execute("SELECT * FROM jobs WHERE id=%s AND (owner_id IS NULL OR owner_id=%s)", (job_id, user["id"])).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")
    profile = get_profile(db, user["id"])
    return {"kind": kind, "job_id": job["id"], "draft": draft(kind, profile, job_data(job)), "verified_information": {"skills": profile["skills"], "projects": profile["projects"], "experience": profile["experience"]}, "review_required": True}

@app.post("/api/github/analyze")
async def github_analyze(request: Request, user=Depends(current_user)):
    body = await read_body(request)
    url = require_text(body, "profile_url", 300, True)
    parsed = urlparse(url)
    username = parsed.path.strip("/").split("/")[0] if parsed.netloc.lower() in {"github.com", "www.github.com"} else ""
    if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", username):
        raise HTTPException(400, "Provide a public github.com/username URL")
    headers = {"Accept": "application/vnd.github+json"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.getenv("GITHUB_TOKEN")
    async with httpx.AsyncClient(timeout=12) as http:
        user_response = await http.get(f"https://api.github.com/users/{username}", headers=headers)
        repo_response = await http.get(f"https://api.github.com/users/{username}/repos", params={"per_page": 30, "sort": "updated"}, headers=headers)
    if user_response.status_code == 404:
        raise HTTPException(404, "Public GitHub user not found")
    if user_response.status_code == 403 or repo_response.status_code == 403:
        raise HTTPException(429, "GitHub rate limit reached. Try again later.")
    data, repos = user_response.json(), repo_response.json()
    return {"username": username, "public_repos": data.get("public_repos", 0), "followers": data.get("followers", 0), "repositories": [{"name": repo["name"], "description": repo.get("description"), "language": repo.get("language"), "stars": repo.get("stargazers_count", 0), "updated_at": repo.get("updated_at"), "url": repo.get("html_url")} for repo in repos], "observations": ["These insights use public repository names, languages, descriptions, stars, and update dates only.", "Repository README quality, test coverage, and code quality are not inspected."]}

@app.post("/api/linkedin/suggestions")
def linkedin_suggestions(user=Depends(current_user), db=Depends(get_db)):
    profile = get_profile(db, user["id"])
    roles = profile["preferences"].get("roles", [])
    role = roles[0] if roles else "your target role"
    skills = ", ".join(profile["skills"][:6]) or "[verified relevant skills]"
    return {"headline": f"Early-career {role} | {skills}", "about": f"I am building my career in {role}, with projects and skills involving {skills}. [Add a verified project example and outcome.]", "review_required": True, "note": "Suggestions are derived from your profile. Review and publish manually; CareerPilot does not access or edit LinkedIn."}

@app.get("/api/gaps/{job_id}")
def gaps(job_id: int, user=Depends(current_user), db=Depends(get_db)):
    job = db.execute("SELECT * FROM jobs WHERE id=%s AND (owner_id IS NULL OR owner_id=%s)", (job_id, user["id"])).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")
    profile = get_profile(db, user["id"])
    match = score_match(profile, job)
    gaps_found = [{"skill": skill, "priority": "High", "reason": "Recorded as a structured required skill on this saved listing."} for skill in match["missing_required_skills"]]
    if not match["missing_required_skills"]:
        gaps_found = [{"skill": skill, "priority": "Review", "reason": "Detected in the description, but not marked as a verified mandatory requirement."} for skill in match["missing_detected_skills"]]
    return {"job": job["title"], "current_skills": profile["skills"], "gaps": gaps_found, "progress": match["score"], "basis": "High priority means a structured requirement. Review means NLP found a mention in the description; confirm it in the original listing."}

@app.post("/api/portfolio/blueprint")
async def portfolio_blueprint(request: Request, user=Depends(current_user), db=Depends(get_db)):
    body = await read_body(request)
    github_url = require_text(body, "github_url", 300)
    github_skills = set()
    github_note = ""
    repo_count = 0
    if github_url:
        parsed = urlparse(github_url)
        username = parsed.path.strip("/").split("/")[0] if parsed.netloc.lower() in {"github.com", "www.github.com"} else ""
        if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", username):
            raise HTTPException(422, "Provide a public github.com/username URL, or leave the field empty")
        headers = {"Accept": "application/vnd.github+json"}
        if os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = "Bearer " + os.getenv("GITHUB_TOKEN")
        try:
            async with httpx.AsyncClient(timeout=8) as http:
                response = await http.get(f"https://api.github.com/users/{username}/repos", params={"per_page": 30, "sort": "updated"}, headers=headers)
            if response.status_code == 200:
                repos = response.json()
                repo_count = len(repos)
                github_skills = {normalize(skill) for skill in extract_skills(" ".join(f"{repo.get('language', '')} {repo.get('description') or ''} {repo.get('name', '')}" for repo in repos))}
            else:
                github_note = "Could not read public repository metadata; the project suggestion uses resume and job evidence only."
        except httpx.HTTPError:
            github_note = "Could not reach GitHub; the project suggestion uses resume and job evidence only."
    profile = get_profile(db, user["id"])
    jobs = db.execute("SELECT * FROM jobs WHERE owner_id IS NULL OR owner_id=%s ORDER BY created_at DESC LIMIT 50", (user["id"],)).fetchall()
    if not jobs:
        return {"available": False, "message": "Save a few real target jobs first. The project plan will be based on skills those employers actually request.", "github_note": github_note}
    skills = {normalize(skill) for skill in profile["skills"]}
    skills.update(normalize(skill) for skill in extract_skills(" ".join([profile["resume_text"], *profile["projects"], *profile["experience"]])))
    skills.update(github_skills)
    counts = Counter()
    examples = {}
    structured_counts = Counter()
    for job in jobs:
        structured = {normalize(skill) for skill in job["required_skills"] or []}
        detected = structured or {normalize(skill) for skill in extract_skills(" ".join([job["title"], job["description"]]))}
        for skill in detected - skills:
            counts[skill] += 1
            examples.setdefault(skill, []).append(job["title"])
            if skill in structured:
                structured_counts[skill] += 1
    if not counts:
        return {"available": False, "message": "Your saved listings do not show a repeated skill gap after checking your profile and public GitHub metadata. Save more target jobs or add more profile evidence.", "github_note": github_note}
    target_skill, frequency = counts.most_common(1)[0]
    target_roles = list(dict.fromkeys(examples[target_skill]))[:3]
    role_label = target_roles[0]
    return {
        "available": True,
        "title": f"A small {target_skill} project for {role_label}",
        "why_this_project": f"{target_skill} is not currently evidenced in your profile and appears in {frequency} of your {len(jobs)} saved job descriptions.",
        "evidence": [{"skill": target_skill, "jobs": target_roles, "frequency": frequency, "structured_requirement_count": structured_counts[target_skill], "mention_only_count": frequency - structured_counts[target_skill]}],
        "portfolio_context": {"public_repositories_reviewed": repo_count, "skills_detected_in_repo_metadata": sorted(github_skills), "note": github_note or "GitHub review uses repository names, language labels, and descriptions only; it does not inspect private repositories or run code."},
        "problem": f"Build a small, documented portfolio project that demonstrates {target_skill} in a context relevant to {role_label}. Choose a real problem you can explain and verify.",
        "milestones": ["Write a one-paragraph problem statement and define a tiny, testable scope.", f"Build one working feature that demonstrates {target_skill}; reuse skills you already list: {', '.join(profile['skills'][:5]) or 'your current skills'}.", "Add a clear README, setup steps, sample input/output, and a short limitations section."],
        "acceptance_criteria": ["A reviewer can run or inspect the project from the README.", f"The README shows concrete evidence of {target_skill}.", "Every result and metric in the README is reproducible and verified."],
        "scope_guardrail": "Description mentions are not necessarily mandatory requirements. Verify them on the source listing. This brief is a project idea, not a claim that the project has been built; use tools you can explain.",
    }
