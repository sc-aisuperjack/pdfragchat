import os
import pytest

@pytest.fixture
def has_openai_key():
    return bool(os.getenv("OPENAI_API_KEY"))