"""Sprites terminal backend plugin for Hermes Agent.

Registers ``terminal.backend: sprites`` — stateful cloud sandboxes on
Fly.io with checkpoint & restore, persistent by default and resumed by a
deterministic, profile-scoped name.

Install into ``~/.hermes/plugins/sprites/`` and enable via
``hermes plugins enable sprites``, then:

    hermes config set terminal.backend sprites
    # SPRITES_TOKEN in ~/.hermes/.env (get one with `sprite login`)

Optional settings (``plugins.entries.sprites.settings.*`` in config.yaml):
``name_prefix`` confines every Sprite name to ``<prefix>-*`` and ``scope``
(``session`` | ``instance``) chooses one Sprite per task id or one per
profile. See README.md.

Attribution: the Sprites environment was authored by Kyle McLaren
(@kylemclaren, Fly.io) as hermes-agent PR #30112 and hardened through the
#93523 salvage review (identity digests, DNS bounds, fail-closed profile
resolution, race-safe first-use create, bounded exec deadlines). Extracted
here when terminal backends became pluggable (hermes-agent PR #94400).
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Callable, Dict, Optional

from agent.terminal_env_provider import TerminalEnvironmentProvider

_SPRITES_SPEC = "sprites-py>=0.5.0,<0.6"

#: Manifest name; the settings namespace when no plugin context is available.
_PLUGIN_ID = "sprites"

#: Plugin settings this backend reads (``plugins.entries.<id>.settings.<key>``).
_SETTING_KEYS = ("name_prefix", "scope")


def _read_setting_from_config(key: str, default: Any = None) -> Any:
    """Read ``plugins.entries.sprites.settings.<key>`` straight from config.yaml.

    Used only when the provider was built without a plugin context (direct
    construction). Errors propagate: settings that cannot be read must not be
    treated as unset, or a configured prefix would be silently dropped.
    """
    from hermes_cli.config import load_config_readonly

    node: Any = load_config_readonly() or {}
    for part in ("plugins", "entries", _PLUGIN_ID, "settings", key):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _sdk_installed() -> bool:
    """True when the sprites SDK is importable.

    ``find_spec`` raises ValueError for an already-imported module whose
    ``__spec__`` is None (e.g. injected test doubles); treat presence in
    sys.modules as installed.
    """
    import sys

    if "sprites" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("sprites") is not None
    except (ImportError, ValueError):
        return False


def _get_token() -> Optional[str]:
    try:
        from agent.secret_scope import get_secret

        return get_secret("SPRITES_TOKEN") or get_secret("SPRITE_TOKEN")
    except Exception:
        return os.getenv("SPRITES_TOKEN") or os.getenv("SPRITE_TOKEN")


class SpritesProvider(TerminalEnvironmentProvider):
    """Sprites — stateful cloud sandboxes on Fly.io."""

    name = "sprites"
    display_name = "Sprites"
    is_remote = True
    is_container = True
    # A durable Sprite is resumed BY NAME; under container_persistent: false
    # a shared deterministic name would let two independent ephemeral runs
    # attach one live VM and delete it out from under each other (#82731).
    session_isolated_when_nonpersistent = True

    def __init__(self, get_config: Optional[Callable[[str, Any], Any]] = None):
        # ``get_config`` is ``PluginContext.get_config`` when registered by the
        # plugin loader: it reads this plugin's settings namespace live from
        # config.yaml on every call, so a changed setting applies to the next
        # environment without a restart.
        self._get_config = get_config or _read_setting_from_config

    def _naming_settings(self) -> tuple:
        """Validated ``(name_prefix, scope)``; raises ``ValueError`` if invalid."""
        if __package__:
            from .sprites_environment import resolve_naming_settings
        else:
            from sprites_environment import resolve_naming_settings
        raw = {key: self._get_config(key, None) for key in _SETTING_KEYS}
        return resolve_naming_settings(raw["name_prefix"], raw["scope"])

    @property
    def description(self) -> str:
        return (
            "Run commands in a Sprite — a stateful cloud sandbox on Fly.io "
            "with checkpoint & restore."
        )

    @property
    def env_description(self) -> str:
        return "a Sprite — a stateful cloud sandbox on Fly.io (Linux)"

    @property
    def cache_path_base(self) -> Optional[str]:
        # Hermes cache files are synced under the remote user's home.
        return "~/.hermes"

    @property
    def strip_env_keys(self) -> frozenset:
        return frozenset({"SPRITES_TOKEN", "SPRITE_TOKEN"})

    def is_available(self) -> bool:
        return (
            _sdk_installed()
            and bool(_get_token())
        )

    def check_requirements(self, config: Dict[str, Any]) -> bool:
        import logging

        logger = logging.getLogger(__name__)
        if not _sdk_installed():
            logger.error(
                "sprites-py is required for the Sprites terminal backend: "
                "pip install '%s'", _SPRITES_SPEC,
            )
            return False
        if not _get_token():
            logger.error(
                "Sprites backend requires SPRITES_TOKEN. Run `sprite login` "
                "and put the token in ~/.hermes/.env as SPRITES_TOKEN."
            )
            return False
        return True

    def probe(self):
        if not _sdk_installed():
            return (
                "needs_setup",
                f"sprites-py SDK not installed — pip install '{_SPRITES_SPEC}'.",
            )
        if _get_token():
            return ("ready", "")
        return ("needs_setup", "Set SPRITES_TOKEN to use the Sprites backend.")

    def setup_instructions(self):
        return [
            "Stateful cloud sandboxes on Fly.io, with checkpoint & restore.",
            "Sprites persist between sessions and are reused by task_id.",
            "Sign up at: https://sprites.dev",
            "Get a token with: sprite login  (or `sprite auth setup --token ...`)",
            "Tip: mint a Restricted Token with prefix=hermes to scope it to",
            "     hermes-* sprites only. Recommended for CI / shared use.",
            "Save it in ~/.hermes/.env as SPRITES_TOKEN.",
            "Note: Sprites allocates compute dynamically (up to 8 CPU / 16 GB RAM).",
        ]

    def doctor_checks(self):
        rows = []
        token = _get_token()
        rows.append((
            bool(token),
            "Sprites token",
            "(configured)" if token else "(required — run `sprite login`, save as SPRITES_TOKEN)",
        ))
        sdk_ok = _sdk_installed()
        rows.append((
            sdk_ok,
            "sprites-py SDK",
            "(installed)" if sdk_ok else f"(pip install '{_SPRITES_SPEC}')",
        ))
        persistent = os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"1", "true", "yes", "on"}
        rows.append((
            True,
            "Sprites persistence",
            "Sprite stays alive across sessions; its ext4 filesystem is the authoritative store"
            if persistent else "Sprite is deleted on cleanup (ephemeral)",
        ))
        try:
            prefix, scope = self._naming_settings()
        except Exception as e:
            rows.append((False, "Sprites naming", f"(invalid plugin settings: {e})"))
        else:
            rows.append((
                True,
                "Sprites naming",
                f"prefix {prefix + '-*' if prefix else 'hermes-* (default)'}, scope {scope}",
            ))
        return rows

    def create_environment(self, *, cwd, timeout, task_id="default",
                           image=None, container_config=None, **kwargs):
        if __package__:
            # Normal path: loaded by the Hermes plugin manager as a package.
            from .sprites_environment import SpritesEnvironment
        else:
            # Test / direct-import path (repo root on sys.path).
            from sprites_environment import SpritesEnvironment

        cc = container_config or {}
        name_prefix, scope = self._naming_settings()
        return SpritesEnvironment(
            cwd=cwd,
            timeout=timeout,
            persistent_filesystem=cc.get("container_persistent", True),
            task_id=task_id,
            name_prefix=name_prefix,
            scope=scope,
        )


def register(ctx):
    # Read settings through the plugin context so they come from this
    # plugin's own namespace (``plugins.entries.<plugin_id>.settings``).
    ctx.register_terminal_environment_provider(
        SpritesProvider(get_config=getattr(ctx, "get_config", None))
    )
