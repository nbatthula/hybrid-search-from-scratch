"""Turn text into a list of lowercase word tokens.

Rules, in order:
1. Lowercase the text.
2. Split on anything that is not a letter. Spaces and punctuation
   mark word breaks. Numbers are dropped.
3. Drop any token in STOPWORDS.

STOPWORDS is kept small on purpose, and it holds only words that
never appear in the three running-example docs: a, an, the, is, are,
and, or, of, in, on. None of the 11 vocab words is dropped, so the
vocab stays at 11 words and each doc still has 5 tokens.
"""

import re

STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "and", "or", "of", "in", "on"}
)

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text):
    """Return the list of tokens in text, in order.

    Example: tokenize("How to reset your password") gives
    ["how", "to", "reset", "your", "password"].
    """
    words = _TOKEN_RE.findall(text.lower())
    return [word for word in words if word not in STOPWORDS]
