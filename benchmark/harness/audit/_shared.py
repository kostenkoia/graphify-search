"""Names two or more of the audit package's checks share."""

from __future__ import annotations

import re
import shutil

_WS = re.compile(r"\s+")
# inv: git is resolved from the harness's own PATH, never from the sandbox environment
_GIT = shutil.which("git") or "git"
