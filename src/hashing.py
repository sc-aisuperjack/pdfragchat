import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def short_hash(h: str, n: int = 12) -> str:
    return h[:n]