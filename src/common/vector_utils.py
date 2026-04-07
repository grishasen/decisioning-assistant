from __future__ import annotations

from typing import Any

import numpy as np


def dot_score(left: Any, right: Any) -> float:
    """Signature: def dot_score(left: Any, right: Any) -> float.

    Return the dot product between two vectors as a float.
    """
    return float(np.dot(np.asarray(left), np.asarray(right)))
