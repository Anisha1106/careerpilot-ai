import re
import pymupdf as fitz
from docx import Document
from io import BytesIO

SKILL_TERMS = {
    "python": ("python",),
    "java": ("java",),
    "javascript": ("javascript", "js"),
    "typescript": ("typescript", "ts"),
    "react": ("react", "react.js", "react js"),
    "fastapi": ("fastapi", "fast api"),
    "django": ("django",),
    "flask": ("flask",),
    "sql": ("sql",),
    "postgresql": ("postgresql", "postgres", "psql"),
    "mysql": ("mysql",),
    "machine learning": ("machine learning", "ml"),
    "deep learning": ("deep learning",),
    "nlp": ("nlp", "natural language processing"),
    "generative ai": ("generative ai", "genai", "gen ai"),
    "llm": ("llm", "llms", "large language model", "large language models"),
    "rag": ("rag", "retrieval augmented generation"),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "docker": ("docker",),
    "git": ("git",),
    "github": ("github",),
    "html": ("html",),
    "css": ("css",),
    "node.js": ("node.js", "node js", "nodejs"),
    "aws": ("aws", "amazon web services"),
    "azure": ("azure",),
    "tableau": ("tableau",),
    "power bi": ("power bi",),
    "tensorflow": ("tensorflow",),
    "pytorch": ("pytorch",),
    "transformers": ("transformers",),
    "hugging face": ("hugging face", "huggingface"),
    "rest api": ("rest api", "restful api"),
    "linux": ("linux",),
    "mongodb": ("mongodb", "mongo db"),
    "redis": ("redis",),
    "kubernetes": ("kubernetes", "k8s"),
    "airflow": ("airflow",),
    "spark": ("spark", "apache spark"),
    "computer vision": ("computer vision",),
    "statistics": ("statistics",),
    "data analysis": ("data analysis",),
    "data visualization": ("data visualization",),
    "scipy": ("scipy",),
    "pydantic": ("pydantic",),
}

_SKILL_PATTERNS = sorted(
    ((term, canonical) for canonical, terms in SKILL_TERMS.items() for term in terms),
    key=lambda item: len(item[0]),
    reverse=True,
)

def extract_skills(text: str) -> list[str]:
    found = set()
    for term, canonical in _SKILL_PATTERNS:
        pattern = r"(?<![\w])" + re.escape(term) + r"(?![\w])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(canonical)
    return sorted(found)

def extract(filename: str, raw: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=raw, filetype="pdf") as document:
            text = "\n".join(page.get_text() for page in document)
    elif filename.lower().endswith(".docx"):
        document = Document(BytesIO(raw))
        blocks = [paragraph.text for paragraph in document.paragraphs]
        blocks.extend(cell.text for table in document.tables for row in table.rows for cell in row.cells)
        text = "\n".join(blocks)
    else:
        raise ValueError("Upload a PDF or DOCX resume")
    return re.sub(r"[ \t]+", " ", text).strip()

def analyze(text: str) -> dict:
    lower = text.lower()
    found = extract_skills(text)
    sections = [name for name in ["education", "experience", "projects", "skills", "certifications"] if re.search(r"\b" + name + r"\b", lower)]
    recommendations = []
    if "projects" not in sections: recommendations.append("Add a clearly labeled Projects section with outcomes and technologies.")
    if "experience" not in sections: recommendations.append("Add internship, work, volunteer, or relevant practical experience if applicable.")
    if not re.search(r"\b(achieved|improved|built|reduced|increased|developed|automated|delivered)\b", lower): recommendations.append("Use evidence based action statements; quantify impact only when you can verify the figures.")
    return {"skills": found, "sections_detected": sections, "recommendations": recommendations, "word_count": len(text.split()), "extracted_text": text[:30000]}
