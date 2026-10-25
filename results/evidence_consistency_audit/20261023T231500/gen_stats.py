"""Historical entry delegates to the maintained read-only statistics generator.

Use --out <fresh-directory>. Old JSON/CSV evidence stays unchanged.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evidence_stats import counts, full_steady as steady, main


if __name__ == "__main__":
    main()
