from __future__ import annotations

from typing import Any

import numpy as np


def dot_score(left: Any, right: Any) -> float:
    """Return a scalar dot product as a Python float."""
    return float(np.dot(np.asarray(left), np.asarray(right)))
