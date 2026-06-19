"""Project constants and JD interpretation.

This file is the source of truth for the first real rule-based scorer. The
weights and keyword groups encode the Senior AI Engineer JD in a transparent
way, so later scoring changes can be reviewed instead of hidden in ad hoc code.
"""

from pathlib import Path
from datetime import date

DEFAULT_CANDIDATES_PATH = Path("candidates.jsonl")
DEFAULT_OUTPUT_PATH = Path("submission.csv")
DEFAULT_ELIGIBLE_PATH = Path("eligible_candidates.jsonl")
DEFAULT_REJECTED_PATH = Path("rejected_candidates.jsonl")
DEFAULT_TOP_2000_CSV_PATH = Path("top_2000_candidates.csv")
DEFAULT_TOP_2000_JSONL_PATH = Path("top_2000_candidates.jsonl")
DEFAULT_BM25_CACHE_PATH = Path("bm25_scores.npz")
DEFAULT_SECTION_EMBEDDINGS_PATH = Path("section_embeddings.npz")
DEFAULT_SECTION_EMBEDDINGS_DIR = Path(".")
DEFAULT_QUERY_EMBEDDINGS_PATH = Path("query_embeddings.npz")
DEFAULT_QDRANT_PATH = Path("qdrant_storage")
TOP_N = 100
TOP_2000_N = 2000
PANDAS_CHUNK_SIZE = 50000
DEFAULT_LOADER = "jsonl"
DEFAULT_DENSE_BACKEND = "numpy"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_SEQ_LENGTH = 160
EMBEDDING_LOCAL_FILES_ONLY = True
TEXT_BUILDER_VERSION = "v3_summary_no_education_dense"
EMBEDDING_CACHE_VERSION = TEXT_BUILDER_VERSION
QDRANT_COLLECTION_NAME = "redranker_section_vectors"
QDRANT_UPSERT_BATCH_SIZE = 1024
REFERENCE_DATE = date(2026, 6, 17)
ROLE_REQUIRES_HYBRID_OR_ONSITE = False

PREFERRED_OFFICE_LOCATIONS = [
    "pune",
    "noida",
]

RETRIEVAL_SECTION_WEIGHTS = {
    "title": 0.12,
    "summary": 0.18,
    "career_history": 0.50,
    "skills": 0.20,
}

RETRIEVAL_SECTION_MAX_CHARS = {
    "title": 200,
    "summary": 900,
    "career_history": 1600,
    "skills": 900,
}

SECTION_EMBEDDING_BATCH_SIZES = {
    "title": 512,
    "summary": 256,
    "career_history": 128,
    "skills": 256,
}

SECTION_EMBEDDING_MAX_SEQ_LENGTHS = {
    "title": 32,
    "summary": 128,
    "career_history": 160,
    "skills": 96,
}

HYBRID_DENSE_WEIGHT = 0.70
HYBRID_BM25_WEIGHT = 0.30

JD_SECTION_QUERIES = {
    "title": (
        "Senior AI Engineer search engineer recommendation systems engineer "
        "machine learning engineer applied ML engineer NLP engineer ranking engineer"
    ),
    "summary": (
        "Own the intelligence layer for candidate job matching. Strong product "
        "engineering mindset, async written communication, fast iteration, "
        "production ML, retrieval ranking recommendation systems, evaluation, "
        "shipper mindset."
    ),
    "career_history": (
        "Shipped end-to-end ranking search recommendation retrieval or matching "
        "systems to real users at meaningful scale. Built production ML services, "
        "hybrid retrieval, dense retrieval, BM25, vector search, reranking, A/B "
        "testing, NDCG, MRR, MAP, feedback loops, monitoring, latency."
    ),
    "skills": (
        "Python machine learning NLP information retrieval ranking search "
        "recommendation systems embeddings vector search semantic search BM25 "
        "FAISS Elasticsearch OpenSearch Qdrant Pinecone MLOps evaluation LLM "
        "reranking RAG fine-tuning prompt engineering."
    ),
    "education": (
        "Computer science machine learning artificial intelligence information "
        "retrieval data science software engineering mathematics statistics."
    ),
}

WELCOME_INDIA_LOCATIONS = [
    "hyderabad",
    "pune",
    "mumbai",
    "delhi",
    "delhi ncr",
    "ncr",
    "noida",
    "gurgaon",
    "gurugram",
]

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

AI_EXPLORER_ONLY_TERMS = [
    "generative ai explorer",
    "online courses",
    "taking online courses",
    "side projects",
    "experimenting with langchain",
    "exploring how llms",
    "emerging ai technologies",
    "grow my ai capabilities",
    "augment my work",
]

NON_TARGET_ROLE_TERMS = [
    "civil engineer",
    "mechanical engineer",
    "chemical engineer",
    "electrical engineer",
    "business analyst",
    "sales executive",
    "sales manager",
    "marketing",
    "content writer",
    "seo",
    "project manager",
    "operations manager",
    "operations management",
    "customer support",
    "support lead",
    "graphic designer",
    "brand design",
    "creative direction",
    "hr",
    "finance",
    "accountant",
    "procurement",
    "logistics",
    "warehouse",
    "fulfillment",
    "editorial",
]

BUSINESS_NON_TARGET_TERMS = [
    "business analyst",
    "sales",
    "sales executive",
    "marketing",
    "content",
    "content writer",
    "customer support",
    "support",
    "operations",
    "operations manager",
    "operations management",
    "project manager",
    "procurement",
    "logistics",
    "warehouse",
    "fulfillment",
    "seo",
    "graphic designer",
    "brand design",
    "editorial",
    "creative direction",
]

AI_KEYWORD_TERMS = [
    "rag",
    "langchain",
    "openai",
    "chatgpt",
    "llm",
    "llms",
    "embeddings",
    "vector search",
    "semantic search",
    "hybrid retrieval",
    "pinecone",
    "qdrant",
    "faiss",
    "sentence transformers",
    "hugging face transformers",
    "recommendation systems",
    "recommender systems",
    "information retrieval",
    "fine-tuning llms",
    "prompt engineering",
    "genai",
    "generative ai",
]

GENAI_EXPLORER_TERMS = [
    "recently excited",
    "exploring ai",
    "exploring genai",
    "ai enthusiast",
    "genai explorer",
    "building with llms",
    "online courses",
    "side projects",
    "experimenting with langchain",
    "openai api",
    "curious about ai tools",
    "ai tools could augment",
    "streamline workflows",
    "productivity",
    "content creation",
    "ai-assisted content",
    "drafting",
    "editing",
    "learning modern ml",
    "interested in transitioning",
    "self-directed ml projects",
    "not the core of my day",
]

AI_PRODUCTIVITY_USAGE_TERMS = [
    "chatgpt",
    "llm tools",
    "ai-assisted content",
    "productivity",
    "content creation",
    "ai-assisted content",
    "drafting",
    "editing",
    "seo",
    "research",
    "chatgpt usage",
    "using chatgpt",
    "ai tools",
    "augment my work",
    "streamline workflows",
]

REAL_AI_ENGINEERING_EVIDENCE_TERMS = [
    "built ranking",
    "ranking system",
    "search ranking",
    "search relevance",
    "retrieval system",
    "semantic search system",
    "vector search system",
    "hybrid retrieval",
    "hybrid search",
    "recommendation system",
    "recommender system",
    "candidate matching",
    "job matching",
    "bm25",
    "elasticsearch",
    "opensearch",
    "solr",
    "faiss index",
    "qdrant",
    "pinecone",
    "milvus",
    "weaviate",
    "embeddings",
    "embedding pipeline",
    "sentence transformer",
    "llm reranking",
    "reranker",
    "cross encoder",
    "learning to rank",
    "ndcg",
    "mrr",
    "map",
    "precision@k",
    "recall@k",
    "a/b testing",
    "relevance labels",
    "production ml",
    "deployed ml",
    "model serving",
    "ml pipeline",
    "feature pipeline",
    "python backend",
    "fastapi",
    "api latency",
    "monitoring",
    "scale",
]

REAL_CAREER_EVIDENCE_TERMS = REAL_AI_ENGINEERING_EVIDENCE_TERMS

STRONG_REAL_AI_CAREER_EVIDENCE_CATEGORIES = {
    "search_retrieval_ranking": [
        "ranking system",
        "search relevance",
        "retrieval system",
        "semantic search",
        "vector search",
        "hybrid retrieval",
        "bm25",
        "elasticsearch",
        "opensearch",
        "solr",
        "faiss",
        "qdrant",
        "pinecone",
        "milvus",
        "weaviate",
    ],
    "recommendation_matching": [
        "recommendation system",
        "recommender system",
        "candidate matching",
        "job matching",
        "marketplace matching",
        "personalization",
    ],
    "production_ml_system_ownership": [
        "production ml",
        "deployed ml",
        "model serving",
        "ml pipeline",
        "feature pipeline",
        "embedding pipeline",
        "latency",
        "monitoring",
        "scale",
        "backend api",
        "fastapi",
        "python service",
    ],
    "evaluation": [
        "ndcg",
        "mrr",
        "map",
        "precision@k",
        "recall@k",
        "a/b testing",
        "offline evaluation",
        "online evaluation",
        "relevance labels",
        "feedback loop",
    ],
}

REAL_AI_CAREER_ANCHOR_TERMS = [
    "built ranking",
    "ranking system",
    "search ranking",
    "search relevance",
    "retrieval system",
    "semantic search system",
    "vector search system",
    "hybrid retrieval",
    "hybrid search",
    "recommendation system",
    "recommender system",
    "candidate matching",
    "job matching",
    "marketplace matching",
    "personalization",
    "bm25",
    "elasticsearch",
    "opensearch",
    "solr",
    "faiss index",
    "qdrant",
    "pinecone",
    "milvus",
    "weaviate",
    "embedding pipeline",
    "sentence transformer",
    "llm reranking",
    "reranker",
    "cross encoder",
    "learning to rank",
    "production ml",
    "deployed ml",
    "model serving",
    "ml pipeline",
]

REAL_AI_EVALUATION_TERMS = [
    "ndcg",
    "mrr",
    "map",
    "precision@k",
    "recall@k",
    "a/b testing",
    "offline evaluation",
    "online evaluation",
    "relevance labels",
]

ADJACENT_TECHNICAL_ROLE_TERMS = [
    "software engineer",
    "backend engineer",
    "data engineer",
    "analytics engineer",
    "ml engineer",
    "machine learning engineer",
    "search engineer",
    "recommendation engineer",
    "data scientist",
]

PRE_LLM_ML_PRODUCTION_TERMS = [
    "machine learning",
    "ml model",
    "model deployment",
    "model serving",
    "prediction api",
    "predictive modeling",
    "churn prediction",
    "fraud detection",
    "scikit-learn",
    "sklearn",
    "xgboost",
    "lightgbm",
    "random forest",
    "gradient boosting",
    "pytorch",
    "tensorflow",
    "feature engineering",
    "feature pipeline",
    "recommendation",
    "recommender",
    "ranking",
    "search relevance",
    "retrieval",
    "nlp pipeline",
    "document classification",
    "sentiment analysis",
    "a/b testing",
    "offline evaluation",
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

RESEARCH_ROLE_TITLE_TERMS = [
    "research assistant",
    "research intern",
    "research scientist",
    "research engineer",
    "ai research engineer",
    "phd researcher",
    "doctoral researcher",
    "postdoctoral researcher",
    "postdoc",
]

RESEARCH_ENVIRONMENT_TERMS = [
    "academic lab",
    "research lab",
    "university lab",
    "doctoral lab",
    "phd lab",
    "academic research",
    "research-only",
    "paper-focused",
    "publication-focused",
    "thesis",
    "conference paper",
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
    "customer-facing",
    "serving",
    "live",
    "integrated",
    "api",
    "monitoring",
]

CULTURE_ASYNC_WRITING_TERMS = [
    "async",
    "asynchronous",
    "documentation",
    "documented",
    "design doc",
    "rfc",
    "technical writing",
    "wrote",
    "written",
    "docs",
    "runbook",
    "playbook",
]

CULTURE_OWNERSHIP_TERMS = [
    "owned",
    "owner",
    "ownership",
    "drove",
    "led",
    "took responsibility",
    "end-to-end",
    "on-call",
    "accountable",
]

CULTURE_OPEN_DECISION_TERMS = [
    "feedback",
    "trade-off",
    "tradeoff",
    "debated",
    "disagree",
    "stakeholder",
    "cross-functional",
    "decision",
    "review",
    "alignment",
]

CULTURE_FAST_AMBIGUITY_TERMS = [
    "startup",
    "0-to-1",
    "zero-to-one",
    "ambiguous",
    "fast-paced",
    "iterated",
    "iteration",
    "experiment",
    "experimentation",
    "hypothesis",
    "assumption",
    "mvp",
    "shipped",
    "launched",
]

APPLIED_ML_ROLE_TERMS = [
    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "applied scientist",
    "applied ml",
    "data scientist",
    "nlp engineer",
    "search engineer",
    "recommendation systems engineer",
    "recommender systems engineer",
    "ranking engineer",
    "ml platform",
    "production ml",
]

SHIPPED_RANKING_SYSTEM_TERMS = [
    "ranking system",
    "ranker",
    "search system",
    "search relevance",
    "recommendation system",
    "recommender system",
    "matching system",
    "candidate matching",
    "job matching",
    "retrieval pipeline",
    "semantic search",
    "hybrid search",
    "vector search",
]

MEANINGFUL_SCALE_TERMS = [
    "real users",
    "served users",
    "customer-facing",
    "production",
    "launched",
    "shipped",
    "latency",
    "scale",
    "10m",
    "millions",
    "high-traffic",
    "large-scale",
]

RETRIEVAL_JUDGMENT_TERMS = [
    "hybrid retrieval",
    "dense retrieval",
    "sparse retrieval",
    "bm25",
    "vector search",
    "semantic search",
    "reranking",
    "re-ranking",
    "cross-encoder",
    "bi-encoder",
    "ann",
    "faiss",
    "elasticsearch",
    "opensearch",
]

LLM_INTEGRATION_JUDGMENT_TERMS = [
    "fine-tuning",
    "fine tuning",
    "prompt",
    "prompting",
    "prompt engineering",
    "rag",
    "llm reranking",
    "llm re-ranking",
    "guardrails",
    "hallucination",
    "latency",
    "cost",
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
    "image moderation",
    "image recognition",
    "object detection",
    "opencv",
    "resnet",
    "cnn",
    "convolutional",
    "speech recognition",
    "voice ai",
    "voice recognition",
    "asr",
    "tts",
    "text to speech",
    "robotics",
    "slam",
    "ros",
]

NLP_IR_EXPOSURE_TERMS = [
    "nlp",
    "natural language processing",
    "information retrieval",
    "ir",
    "retrieval",
    "ranking",
    "ranker",
    "learning to rank",
    "search",
    "search relevance",
    "semantic search",
    "vector search",
    "hybrid search",
    "bm25",
    "elasticsearch",
    "opensearch",
    "faiss",
    "pinecone",
    "qdrant",
    "weaviate",
    "milvus",
    "embeddings",
    "document classification",
    "text classification",
    "text analytics",
    "sentiment analysis",
    "rag",
    "recommendation",
    "recommender",
    "matching system",
    "candidate matching",
    "job matching",
    "ndcg",
    "mrr",
    "map",
]

CLOSED_SOURCE_SYSTEM_TERMS = [
    "closed-source",
    "closed source",
    "proprietary",
    "internal",
    "enterprise",
    "client",
    "customer-facing",
    "saas",
    "b2b",
    "private",
    "in-house",
]

EXTERNAL_VALIDATION_TERMS = [
    "open source",
    "opensource",
    "github",
    "public repo",
    "public repository",
    "published paper",
    "conference paper",
    "arxiv",
    "speaker",
    "gave a talk",
    "tech talk",
    "conference talk",
    "meetup",
    "blog",
    "public project",
    "kaggle",
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

SENIOR_ENGINEER_ROLE_TERMS = [
    "senior engineer",
    "senior software engineer",
    "senior ai engineer",
    "senior ml engineer",
    "senior machine learning engineer",
    "senior backend engineer",
    "senior data engineer",
    "lead engineer",
    "lead ai engineer",
    "lead ml engineer",
    "lead machine learning engineer",
    "lead software engineer",
    "staff engineer",
    "staff ai engineer",
    "staff ml engineer",
    "staff machine learning engineer",
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
    "title_fit": 0.09,
    "experience_fit": 0.08,
    "retrieval_fit": 0.13,
    "ranking_fit": 0.11,
    "production_fit": 0.10,
    "evaluation_fit": 0.08,
    "python_fit": 0.07,
    "product_company_fit": 0.04,
    "culture_fit": 0.10,
    "ideal_recruiter_fit": 0.20,
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
