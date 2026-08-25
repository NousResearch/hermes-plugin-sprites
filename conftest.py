"""Test bootstrap: resolve hermes-agent imports and the plugin package.

The plugin's unit tests import ``agent.*`` and ``tools.*`` from a
hermes-agent checkout. Point HERMES_AGENT_REPO at one (defaults to
``~/.hermes/hermes-agent``); it must contain the pluggable terminal-backend
extension point (PR #94400).
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

_hermes_agent = Path(
    os.environ.get("HERMES_AGENT_REPO", Path.home() / ".hermes" / "hermes-agent")
).expanduser()

# hermes-agent first (agent/, tools/), then this repo (sprites_environment).
for p in (str(_hermes_agent), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

collect_ignore = ["__init__.py", "sprites_environment.py"]
