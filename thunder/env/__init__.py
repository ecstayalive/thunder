from __future__ import annotations

import os

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()
