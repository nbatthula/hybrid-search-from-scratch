"""Put src on sys.path so tests can import session01_lexical."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
