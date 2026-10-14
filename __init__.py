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
profile; ``image``, ``max_cpus``, ``max_ram_gb`` and ``max_disk_gb`` set the
image and sizing of newly created Sprites. See README.md.

Attribution: the Sprites environment was authored by Kyle McLaren
(@kylemclaren, Fly.io) as hermes-agent PR #30112 and hardened through the
#93523 salvage review (identity digests, DNS bounds, fail-closed profile
resolution, race-safe first-use create, bounded exec deadlines). Extracted
here when terminal backends became pluggable (hermes-agent PR #94400).
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional

from agent.terminal_env_provider import TerminalEnvironmentProvider

_SPRITES_SPEC = "sprites-py>=0.5.0,<0.6"

#: Manifest name; the default settings namespace.
_PLUGIN_ID = "sprites"

#: Plugin settings this backend reads (``plugins.entries.<id>.settings.<key>``).
_SETTING_KEYS = ("name_prefix", "scope", "image", "max_cpus", "max_ram_gb", "max_disk_gb")

#: Settings that only shape the create request (see ``resolve_create_settings``).
_CREATE_SETTING_KEYS = ("image", "max_cpus", "max_ram_gb", "max_disk_gb")


def _failed_read_marker() -> Optional[type]:
    """Core's marker class for a fallback config, wherever this Hermes defines it."""
    import importlib

    for module in ("hermes_cli.config_read_errors", "hermes_cli.config"):
        try:
            marker = getattr(importlib.import_module(module), "FailedConfigRead")
        except (ImportError, AttributeError):
            continue
        if isinstance(marker, type):
            return marker
    return None


def _verify_config_file_parses() -> None:
    """Parse config.yaml directly; raise if it exists but cannot be read or parsed.

    Only used on Hermes releases whose loader falls back silently, without a
    marker the plugin could check.
    """
    from hermes_cli.config import get_config_path

    path = get_config_path()
    if not path.exists():
        return
    try:
        from hermes_yaml import safe_load
    except ImportError:
        from yaml import safe_load
    safe_load(path.read_text(encoding="utf-8-sig"))


def _read_settings(plugin_id: str) -> Dict[str, Any]:
    """Raw plugin settings from ONE config snapshot, or raise if it is not trustworthy.

    When config.yaml cannot be parsed, core serves a fallback (the last good
    copy, or the defaults) instead of raising. Settings taken from it may be
    missing a ``name_prefix`` the operator just configured, so this refuses
    rather than return them. Every key comes from the same snapshot whose read
    status was checked, so a file repaired or broken between two reads cannot
    mix a fallback value into a "clean" result. Lookup order matches
    ``PluginContext.get_config``: ``settings`` first, then the legacy
    ``config`` subtree.
    """
    from hermes_cli.config import load_config_readonly

    marker = _failed_read_marker()
    if marker is None:
        try:
            _verify_config_file_parses()
        except Exception as e:
            raise _unreadable_config(e) from e
    config = load_config_readonly()
    # Check the marker before any normalisation: an empty fallback is falsy.
    if marker is not None and isinstance(config, marker):
        raise _unreadable_config(getattr(config, "read_error", None))

    entry: Any = config
    for part in ("plugins", "entries", plugin_id):
        entry = entry.get(part) if isinstance(entry, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    settings = entry.get("settings") if isinstance(entry.get("settings"), dict) else {}
    legacy = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    return {
        key: settings[key] if key in settings else legacy.get(key)
        for key in _SETTING_KEYS
    }


def _unreadable_config(error: Any) -> RuntimeError:
    return RuntimeError(
        "Sprites backend refused to start: config.yaml could not be read "
        f"({error}), so its plugin settings (name_prefix, scope, image, "
        "sizing) cannot be trusted. Fix config.yaml (`hermes config check`)."
    )


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

    def __init__(self, plugin_id: str = _PLUGIN_ID):
        # The settings namespace: ``plugins.entries.<plugin_id>.settings``.
        # ``register(ctx)`` passes the loader's id for this plugin. Settings
        # are re-read from config.yaml for every new environment, so a
        # changed setting applies without a restart.
        self._plugin_id = plugin_id

    def _naming_settings(self) -> tuple:
        """Validated ``(name_prefix, scope)``.

        Raises ``ValueError`` for an invalid value and ``RuntimeError`` when
        config.yaml could not be read (core then serves fallback values).
        """
        if __package__:
            from .sprites_environment import resolve_naming_settings
        else:
            from sprites_environment import resolve_naming_settings
        raw = _read_settings(self._plugin_id)
        return resolve_naming_settings(raw["name_prefix"], raw["scope"])

    def _settings(self) -> tuple:
        """Validated ``(name_prefix, scope, create_settings)`` from one config read.

        ``create_settings`` holds the raw ``image`` / ``max_*`` values that
        passed validation (``None`` when unset); ``SpritesEnvironment``
        re-validates them.
        """
        if __package__:
            from .sprites_environment import resolve_create_settings, resolve_naming_settings
        else:
            from sprites_environment import resolve_create_settings, resolve_naming_settings
        raw = _read_settings(self._plugin_id)
        prefix, scope = resolve_naming_settings(raw["name_prefix"], raw["scope"])
        create = {key: raw[key] for key in _CREATE_SETTING_KEYS}
        resolve_create_settings(**create)
        return prefix, scope, create

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
        try:
            _, _, create = self._settings()
        except Exception as e:
            rows.append((False, "Sprites create settings", f"(invalid plugin settings: {e})"))
        else:
            chosen = ", ".join(f"{k} {v}" for k, v in create.items() if v not in (None, ""))
            rows.append((
                True,
                "Sprites create settings",
                f"{chosen} (new Sprites only)" if chosen else "default image and sizing",
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
        name_prefix, scope, create = self._settings()
        # ``image`` from Hermes (the Docker backend's image setting) is NOT the
        # Sprite image: only the plugin's own ``image`` setting selects that.
        return SpritesEnvironment(
            cwd=cwd,
            timeout=timeout,
            persistent_filesystem=cc.get("container_persistent", True),
            task_id=task_id,
            name_prefix=name_prefix,
            scope=scope,
            **{k: v for k, v in create.items() if v is not None},
        )


def register(ctx):
    # Settings live in this plugin's own namespace,
    # ``plugins.entries.<plugin_id>.settings``. They are read directly rather
    # than through ``ctx.get_config`` because each key would then come from a
    # separate config read, with no way to tell whether that read was core's
    # fallback for an unparseable config.yaml.
    ctx.register_terminal_environment_provider(
        SpritesProvider(plugin_id=getattr(ctx, "plugin_id", None) or _PLUGIN_ID)
    )
