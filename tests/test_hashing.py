from src.hashing import sha256_bytes, short_hash

def test_sha256_and_short_hash():
    h = sha256_bytes(b"abc")
    assert len(h) == 64
    assert short_hash(h, 12) == h[:12]