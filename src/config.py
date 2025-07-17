"""
Configuration settings for the Flash Report Generator.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import streamlit for secrets support
try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False

def get_config_value(key: str, default=None):
    """Get configuration value from environment or Streamlit secrets."""
    # First try environment variables
    value = os.getenv(key)
    if value:
        return value
    
    # Then try Streamlit secrets if available
    if _has_streamlit:
        try:
            return st.secrets.get(key, default)
        except:
            pass
    
    return default

# Base paths
ROOT_DIR = Path(__file__).parent.parent
TEMPLATE_PATH = ROOT_DIR / "templates"
VECTORSTORE_PATH = ROOT_DIR / "vectorstores"

# Document processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OpenAI settings
OPENAI_API_KEY = get_config_value("OPENAI_API_KEY")
OPENAI_MODEL = get_config_value("OPENAI_MODEL", "gpt-4")  # or gpt-3.5-turbo for faster, cheaper processing
OPENAI_EMBEDDING_MODEL = get_config_value("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# Vector store settings
COLLECTION_PREFIX = "flash_report_"

# Report generation
REPORT_TEMPLATE = TEMPLATE_PATH / "report_template.docx"
COMPANY_LOGO = TEMPLATE_PATH / "Images/Norstella_color_positive_RGB_(2).png"

# Evaluation metrics
RELEVANCE_THRESHOLD = 0.7  # Minimum relevance score for responses
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]  # ROUGE metric types to use

# Ensure required directories exist
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)

# Validate configuration
if not OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or Streamlit secrets."
    )

# Export all settings
__all__ = [
    "ROOT_DIR",
    "TEMPLATE_PATH",
    "VECTORSTORE_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "COLLECTION_PREFIX",
    "REPORT_TEMPLATE",
    "COMPANY_LOGO",
    "RELEVANCE_THRESHOLD",
    "ROUGE_TYPES",
] 