from common.text_utils import normalize_whitespace, split_text


def test_normalize_whitespace():
    assert normalize_whitespace("a   b\n c") == "a b c"


def test_split_text_basic():
    text = " ".join(["word"] * 600)
    chunks = list(split_text(text, chunk_size=200, chunk_overlap=50))
    assert len(chunks) >= 2
    assert all(len(chunk) > 0 for chunk in chunks)
