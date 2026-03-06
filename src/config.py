from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embed_model: str
    data_dir: str
    chroma_dir: str

def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env or environment variables.")

    return Settings(
        openai_api_key=api_key,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        data_dir=os.getenv("DATA_DIR", "data"),
        chroma_dir=os.getenv("CHROMA_DIR", "data/index/chroma"),
    )