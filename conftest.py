"""Test bootstrap: resolve hermes-agent imports and the plugin package.

The plugin's unit tests import ``agent.*`` and ``tools.*`` from a
hermes-agent checkout, which must contain the pluggable terminal-backend
extension point (PR #94400). The checkout is chosen in this order:

1. ``HERMES_AGENT_REPO`` when set;
2. the current directory when it is a hermes-agent checkout. The README
   recipe runs ``scripts/run_tests.sh`` from the checkout, and that runner
   clears the environment, so without this the installed agent below would
   silently be tested instead of the checkout;
3. ``~/.hermes/hermes-agent`` (the installed agent).
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _hermes_agent_repo() -> Path:
    explicit = os.environ.get("HERMES_AGENT_REPO")
    if explicit:
        return Path(explicit).expanduser()
    cwd = Path.cwd()
    if (cwd / "agent" / "terminal_env_provider.py").is_file():
        return cwd
    return Path.home() / ".hermes" / "hermes-agent"


_hermes_agent = _hermes_agent_repo()

# hermes-agent first (agent/, tools/), then this repo (sprites_environment).
for p in (str(_hermes_agent), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

collect_ignore = ["__init__.py", "sprites_environment.py"]
