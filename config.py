import os
from pathlib import Path

from dotenv import load_dotenv


# Find the project directory containing this file.
BASE_DIR = Path(__file__).resolve().parent

# Load .env from the project directory.
load_dotenv(BASE_DIR / ".env")

# Read credentials from environment variables.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Prefer Groq when its key is configured.
if GROQ_API_KEY:
    API_KEY = GROQ_API_KEY
    BASE_URL = os.getenv(
        "API_BASE_URL",
        "https://api.groq.com/openai/v1",
    )
    DEFAULT_MODEL = "openai/gpt-oss-20b"
else:
    API_KEY = OPENAI_API_KEY
    BASE_URL = os.getenv("API_BASE_URL") or None
    DEFAULT_MODEL = "gpt-4.1-mini"

# Allow the model to be overridden in .env.
MODEL = os.getenv(
    "MODEL",
    os.getenv("GROQ_MODEL", DEFAULT_MODEL),
)

# Default approximate token budget for local conversation history.
MAX_HISTORY_TOKENS = int(
    os.getenv("MAX_HISTORY_TOKENS", "1000")
)

# Keep the system prompt separate from conversation history.
SYSTEM_PROMPT = (
    "You are a helpful Python tutor. "
    "Explain concepts in simple language with examples. "
    "Explain technical topics step by step."
)

# Fail early if the configuration is incomplete.
if not API_KEY:
    raise RuntimeError(
        "API key not found. Add GROQ_API_KEY or OPENAI_API_KEY to .env."
    )

if MAX_HISTORY_TOKENS < 1:
    raise RuntimeError("MAX_HISTORY_TOKENS must be greater than zero.")
