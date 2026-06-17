"""Project constants and JD interpretation.

This file is the source of truth for the first real rule-based scorer. The
weights and keyword groups encode the Senior AI Engineer JD in a transparent
way, so later scoring changes can be reviewed instead of hidden in ad hoc code.
"""

from pathlib import Path
from datetime import date

DEFAULT_CANDIDATES_PATH = Path("candidates.jsonl")
DEFAULT_OUTPUT_PATH = Path("submission.csv")
TOP_N = 100
PANDAS_CHUNK_SIZE = 50000
DEFAULT_LOADER = "jsonl"
REFERENCE_DATE = date(2026, 6, 17)
ROLE_REQUIRES_HYBRID_OR_ONSITE = True

PYTHON_SKILL_NAMES = {"python"}
IDEAL_EXPERIENCE_YEARS = 7.0
MIN_TARGET_EXPERIENCE_YEARS = 5.0
MAX_TARGET_EXPERIENCE_YEARS = 9.0

STRONG_TITLES = [
    "search engineer",
    "recommendation systems engineer",
    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "applied ml engineer",
    "senior data scientist",
    "data scientist",
    "software engineer machine learning",
    "backend engineer",
    "data engineer",
]

NEGATIVE_TITLES = [
    "marketing manager",
    "sales executive",
    "hr manager",
    "accountant",
    "graphic designer",
    "content writer",
    "civil engineer",
    "mechanical engineer",
    "customer support",
]

CORE_AI_ML_TERMS = [
    "machine learning",
    "ml",
    "ai",
    "artificial intelligence",
    "nlp",
    "llm",
    "rag",
    "retrieval",
    "ranking",
    "recommendation",
    "search",
    "embeddings",
    "vector",
    "python",
]

HARD_RELEVANCE_TERMS = [
    "ranking",
    "retrieval",
    "search",
    "recommendation",
    "matching",
    "machine learning",
    "ml",
    "embeddings",
    "semantic search",
    "vector search",
    "relevance",
    "candidate matching",
    "job matching",
    "production ml",
]

GENAI_APP_ONLY_TERMS = [
    "langchain",
    "openai",
    "openai api",
    "chatgpt",
    "chatgpt chatbot",
    "prompt engineering",
    "llm chatbot",
    "chatbot",
    "gpt wrapper",
    "rag chatbot",
    "built a demo",
    "tutorial",
]

RESEARCH_ONLY_TERMS = [
    "academic lab",
    "academic research",
    "research assistant",
    "research intern",
    "research scientist",
    "research lab",
    "publication",
    "published paper",
    "papers",
    "conference paper",
    "phd",
    "phd researcher",
    "thesis",
]

PRODUCT_OR_PRODUCTION_TERMS = [
    "production",
    "deployed",
    "shipped",
    "launched",
    "served users",
    "a/b testing",
    "a/b test",
    "users",
    "real users",
    "latency",
    "scale",
    "product",
    "pipeline",
]

REAL_ML_RETRIEVAL_TERMS = [
    "retrieval",
    "ranking",
    "search relevance",
    "recommendation",
    "recommender",
    "matching system",
    "bm25",
    "elasticsearch",
    "opensearch",
    "faiss",
    "vector search",
    "semantic search",
    "hybrid search",
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "production ml",
]

TUTORIAL_ONLY_TERMS = [
    "langchain tutorial",
    "openai api tutorial",
    "chatgpt tutorial",
    "built a demo",
    "small rag side project",
    "prompt engineering",
]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision",
    "image classification",
    "object detection",
    "speech recognition",
    "tts",
    "robotics",
    "slam",
]

CODING_TERMS = [
    "python",
    "api",
    "backend",
    "implemented",
    "built",
    "developed",
    "coded",
    "designed",
    "deployed",
    "fastapi",
    "flask",
    "production",
    "production code",
    "pipeline",
    "code",
    "service",
    "microservice",
    "ranking system",
    "retrieval pipeline",
    "search system",
    "evaluation framework",
]

MANAGER_ARCHITECT_TERMS = [
    "architect",
    "solution architect",
    "principal engineer",
    "engineering manager",
    "tech lead",
    "project manager",
    "program manager",
    "director",
    "head of",
]

RETRIEVAL_TERMS = [
    "retrieval",
    "semantic search",
    "vector search",
    "hybrid search",
    "embeddings",
    "embedding",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "bm25",
]

RANKING_TERMS = [
    "ranking",
    "ranker",
    "recommendation",
    "recommender",
    "learning to rank",
    "relevance",
    "matching system",
    "candidate matching",
]

EVALUATION_TERMS = [
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "a/b test",
    "ab testing",
    "offline evaluation",
    "online evaluation",
    "relevance evaluation",
    "evaluation framework",
]

PRODUCTION_TERMS = [
    "production",
    "deployed",
    "shipped",
    "launched",
    "real users",
    "latency",
    "scale",
    "monitoring",
    "pipeline",
    "index refresh",
    "embedding drift",
    "regression",
]

PYTHON_RELATED_TERMS = [
    "python",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numpy",
]

PRODUCT_COMPANIES = [
    "cred",
    "swiggy",
    "zomato",
    "flipkart",
    "razorpay",
    "freshworks",
    "zerodha",
    "paytm",
    "pied piper",
    "hooli",
    "initech",
]

SERVICE_COMPANIES = [
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
]

FEATURE_WEIGHTS = {
    "title_fit": 0.14,
    "experience_fit": 0.12,
    "retrieval_fit": 0.18,
    "ranking_fit": 0.14,
    "production_fit": 0.14,
    "evaluation_fit": 0.10,
    "python_fit": 0.10,
    "product_company_fit": 0.08,
}

CORE_ASSESSMENT_SKILLS = [
    "python",
    "machine learning",
    "ml",
    "nlp",
    "search",
    "ranking",
    "recommendation",
    "retrieval",
]

SENIORITY_TRAP_TITLES = [
    "principal",
    "staff engineer",
    "architect",
    "director",
    "head of ai",
    "vp engineering",
    "vp of engineering",
]

IMPOSSIBLE_TECH_DURATION_LIMITS = {
    "langchain": 42,
    "rag": 48,
    "qlora": 42,
    "lora": 60,
    "fine-tuning llms": 48,
    "prompt engineering": 48,
    "chatgpt": 42,
    "openai": 60,
    "pinecone": 60,
    "qdrant": 60,
    "sentence transformers": 72,
    "gpt-4": 36,
    "gpt4": 36,
    "rag chatbot": 48,
}

# Optional extension point. Keep empty unless we have reliable source data for
# company founding years; do not guess founding years inside scoring logic.
COMPANY_FOUNDING_YEAR = {}
