import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default model cache directory
CACHE_PATH = os.getenv("CACHE_PATH", str(Path.home() / ".cache" / "huggingface"))

# HuggingFace authentication token
HF_TOKEN = os.getenv("HF_TOKEN", None)
