"""Unit tests for the Sprites cloud sandbox environment backend.

These exercise SpritesEnvironment against a mocked sprites-py SDK; no
network or token required. Live-API checks live under
tests/test_sprites_terminal_live.py.
"""

import json
import logging
import os
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest


# ---------------------------------------------------------------------------
# Mock sprites-py SDK
# ---------------------------------------------------------------------------

class _SpriteError(Exception):
    pass


class _NotFoundError(_SpriteError):
    """Mirror of sprites.exceptions.NotFoundError (a SpriteError subclass)."""


class _NetworkError(_SpriteError):
    pass


class _AuthenticationError(_SpriteError):
    pass


class _ExitError(Exception):
    """Mirror of sprites.exceptions.ExitError."""

    def __init__(self, message, exit_code, stdout=b"", stderr=b""):
        super().__init__(message)
        self._exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def exit_code(self):
        return self._exit_code


class _SpritesTimeoutError(Exception):
    pass


def _patch_sprites_imports(monkeypatch):
    """Inject a fake sprites SDK so SpritesEnvironment can import it."""
    sprites_mod = types.ModuleType("sprites")
    sprites_mod.SpritesClient = MagicMock(name="SpritesClient")

    exc_mod = types.ModuleType("sprites.exceptions")
    exc_mod.NotFoundError = _NotFoundError
    exc_mod.SpriteError = _SpriteError
    exc_mod.NetworkError = _NetworkError
    exc_mod.AuthenticationError = _AuthenticationError
    exc_mod.ExitError = _ExitError
    exc_mod.TimeoutError = _SpritesTimeoutError
    sprites_mod.exceptions = exc_mod

    monkeypatch.setitem(sys.modules, "sprites", sprites_mod)
    monkeypatch.setitem(sys.modules, "sprites.exceptions", exc_mod)
    return sprites_mod, exc_mod


def _make_sprite(name="hermes-default"):
    sprite = MagicMock()
    sprite.name = name

    # $HOME detection returns "/home/sprite" by default
    home_cmd = MagicMock()
    home_cmd.combined_output.return_value = b"/home/sprite\n"

    # init_session() bootstrap also goes through sprite.command(...).
    # combined_output() must succeed (return bytes) for snapshot_ready=True.
    bootstrap_cmd = MagicMock()
    bootstrap_cmd.combined_output.return_value = b"\n__HERMES_CWD_xxx__/home/sprite__HERMES_CWD_xxx__\n"

    sprite.command.side_effect = [home_cmd, bootstrap_cmd]
    sprite.filesystem.return_value = MagicMock()
    return sprite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Unit tests must not read the operator's profiles, tokens, or skills."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("SPRITES_TOKEN", raising=False)
    monkeypatch.delenv("SPRITE_TOKEN", raising=False)
    monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
    return home

@pytest.fixture()
def sprites_sdk(monkeypatch):
    return _patch_sprites_imports(monkeypatch)


@pytest.fixture()
def hermes_runtime(monkeypatch):
    """Keep host-file sync out of transport-mocked tests."""
    monkeypatch.setenv("SPRITES_TOKEN", "test-token")
    # Skip credential-file enumeration so init doesn't bring in real ~/.hermes state
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts", lambda: []
    )
    monkeypatch.setattr(
        "tools.credential_files.iter_skills_files", lambda **kw: []
    )
    monkeypatch.setattr(
        "tools.credential_files.iter_cache_files", lambda **kw: []
    )
    # Keep the base class from blocking forever on interrupt polling
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)


@pytest.fixture()
def make_env(sprites_sdk, hermes_runtime, monkeypatch):
    """Return a factory for SpritesEnvironment instances backed by the mocked SDK."""
    # Pin the profile identity to default (None) so Sprite names are
    # deterministic regardless of the test runner's HERMES_HOME.
    # Profile-scoping itself is covered explicitly in TestSpriteNaming.
    monkeypatch.setattr(
        "sprites_environment._resolve_profile_identity",
        lambda: None,
    )

    def _factory(get_side_effect=None, sprite=None, **kwargs):
        sprite = sprite or _make_sprite()

        mock_client = MagicMock()
        mock_client.create_sprite.return_value = sprite
        if get_side_effect is not None:
            mock_client.get_sprite.side_effect = get_side_effect
        else:
            mock_client.get_sprite.side_effect = _NotFoundError("not found")

        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(return_value=mock_client)

        from sprites_environment import SpritesEnvironment

        env = SpritesEnvironment(**kwargs)
        env._mock_client = mock_client
        env._mock_sprite = sprite
        return env

    return _factory


# ---------------------------------------------------------------------------
# Construction / token handling
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_missing_token_raises(self, sprites_sdk, monkeypatch):
        monkeypatch.delenv("SPRITES_TOKEN", raising=False)
        monkeypatch.delenv("SPRITE_TOKEN", raising=False)
        monkeypatch.setattr(
            "tools.lazy_deps.ensure", lambda *a, **k: None, raising=False
        )
        from sprites_environment import SpritesEnvironment

        with pytest.raises(ValueError, match="SPRITES_TOKEN"):
            SpritesEnvironment(task_id="x")

    def test_persistent_uses_get_first(self, make_env):
        existing = _make_sprite(name="hermes-mine")
        env = make_env(
            get_side_effect=lambda name: existing,
            sprite=existing,
            task_id="mine",
            persistent_filesystem=True,
        )
        env._mock_client.get_sprite.assert_called_once_with("hermes-mine")
        env._mock_client.create_sprite.assert_not_called()
        # The instance records the resolved name it will resume under.
        assert env._sprite_name == "hermes-mine"

    def test_creates_when_not_found(self, make_env):
        env = make_env(task_id="fresh", persistent_filesystem=True)
        env._mock_client.get_sprite.assert_called_once_with("hermes-fresh")
        env._mock_client.create_sprite.assert_called_once_with("hermes-fresh")
        assert env._sprite_name == "hermes-fresh"

    def test_no_size_kwargs_passed_to_create(self, make_env):
        """Compute sizing isn't honored yet — make sure we don't sneak it back in."""
        env = make_env(task_id="sizing")
        args, kwargs = env._mock_client.create_sprite.call_args
        assert args == ("hermes-sizing",)
        assert kwargs == {}


class TestEphemeralIsolation:
    """container_persistent: false — the #82731 contract on Sprites.

    A non-persistent sandbox must not survive or be shared across sessions.
    An ephemeral constructor therefore mints a unique name and only ever
    CREATES — it must never adopt a pre-existing Sprite (a peer's live VM,
    or the stale survivor of a crashed prior run).
    """

    def test_ephemeral_never_adopts(self, make_env):
        env = make_env(task_id="mine", persistent_filesystem=False)
        env._mock_client.get_sprite.assert_not_called()
        env._mock_client.create_sprite.assert_called_once()

    def test_ephemeral_names_are_unique_per_construction(self, make_env):
        a = make_env(task_id="mine", persistent_filesystem=False)
        b = make_env(task_id="mine", persistent_filesystem=False)
        assert a._sprite_name != b._sprite_name
        assert a._sprite_name.startswith("hermes-eph-mine-")
        assert b._sprite_name.startswith("hermes-eph-mine-")

    def test_ephemeral_names_are_dns_bounded(self):
        import re
        from sprites_environment import (
            _MAX_NAME_LEN,
            _ephemeral_sprite_name,
        )
        name = _ephemeral_sprite_name("session:agent:main:telegram:" + "x" * 60)
        assert len(name) <= _MAX_NAME_LEN
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)

    def test_session_isolation_covers_sprites(self, monkeypatch):
        """terminal_tool keys non-persistent sprites per session, not 'default'."""
        import __init__ as plugin_pkg
        import tools.terminal_tool as tt
        from agent import terminal_env_registry as reg

        reg._reset_for_tests()
        try:
            reg.register_provider(plugin_pkg.SpritesProvider())
            monkeypatch.setenv("TERMINAL_ENV", "sprites")
            monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
            monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
            # Docker-only paths (workspace mounts, container teardown) stay off.
            assert tt._docker_session_isolation_enabled() is False
            # An ordinary session task id no longer collapses onto the shared key.
            assert tt._resolve_container_task_id("session-abc123") != "default"
            # Persistent mode keeps the documented shared-Sprite contract.
            monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
            assert tt._resolve_container_task_id("session-abc123") == "default"
        finally:
            reg._reset_for_tests()

class TestPersistentCreateRace:
    """Cross-process first-use TOCTOU: GET 404 → CREATE loses to a peer."""

    def test_create_race_adopts_winner(self, make_env):
        winner = _make_sprite(name="hermes-fresh")
        gets = iter([_NotFoundError("404"), winner])

        def _get(name):
            result = next(gets)
            if isinstance(result, Exception):
                raise result
            return result

        env = make_env(
            get_side_effect=_get,
            sprite=winner,
            task_id="fresh",
            persistent_filesystem=True,
        )
        env._mock_client.create_sprite.side_effect = None
        # Constructor path: first get 404s, create raises (peer won), re-get
        # adopts the peer's Sprite.
        assert env._sprite is winner

    def test_create_race_wiring(self, sprites_sdk, monkeypatch):
        """Drive the constructor directly: create fails, re-GET adopts."""
        self._prep(monkeypatch)
        winner = _make_sprite(name="hermes-fresh")
        mock_client = MagicMock()
        gets = iter([_NotFoundError("404"), winner])
        mock_client.get_sprite.side_effect = (
            lambda name: (_ for _ in ()).throw(g) if isinstance(g := next(gets), Exception) else g
        )
        mock_client.create_sprite.side_effect = _SpriteError("name already exists")
        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(return_value=mock_client)

        from sprites_environment import SpritesEnvironment
        env = SpritesEnvironment(task_id="fresh", persistent_filesystem=True)
        assert env._sprite is winner
        assert mock_client.get_sprite.call_count == 2

    def test_genuine_create_error_not_masked(self, sprites_sdk, monkeypatch):
        """create fails AND the re-GET still 404s → the create error surfaces."""
        self._prep(monkeypatch)
        mock_client = MagicMock()
        mock_client.get_sprite.side_effect = _NotFoundError("404")
        mock_client.create_sprite.side_effect = _SpriteError("quota exceeded")
        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(return_value=mock_client)

        from sprites_environment import SpritesEnvironment
        with pytest.raises(_SpriteError, match="quota exceeded"):
            SpritesEnvironment(task_id="fresh", persistent_filesystem=True)

    @staticmethod
    def _prep(monkeypatch):
        monkeypatch.setenv("SPRITES_TOKEN", "test-token")
        monkeypatch.setattr(
            "tools.lazy_deps.ensure", lambda *a, **k: None, raising=False
        )
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts", lambda: []
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files", lambda **kw: []
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files", lambda **kw: []
        )
        monkeypatch.setattr(
            "tools.environments.base.is_interrupted", lambda: False
        )
        monkeypatch.setattr(
            "sprites_environment._resolve_profile_identity",
            lambda: None,
        )


class TestDeleteFailurePropagates:
    """_sprite_delete must honor FileSyncManager's rollback transaction.

    The manager commits a deletion (drops it from _synced_files, never
    retries) when the delete callback RETURNS; it rolls back and retries
    only when the callback RAISES. Swallowing a transient remote failure
    therefore leaves stale credential material in a durable Sprite forever.
    """

    def test_delete_exec_failure_raises(self, make_env):
        env = make_env(task_id="del", persistent_filesystem=True)
        cmd = MagicMock()
        cmd.combined_output.side_effect = _ExitError(
            "exit status 1", 1, b"rm: cannot remove: Read-only file system\n"
        )
        env._mock_sprite.command = MagicMock(return_value=cmd)
        with pytest.raises(RuntimeError, match="Read-only file system"):
            env._sprite_delete(["/home/sprite/.hermes/.env"])

    def test_delete_transport_failure_raises(self, make_env):
        env = make_env(task_id="del", persistent_filesystem=True)
        cmd = MagicMock()
        cmd.combined_output.side_effect = OSError("websocket dropped")
        env._mock_sprite.command = MagicMock(return_value=cmd)
        with pytest.raises(OSError, match="websocket dropped"):
            env._sprite_delete(["/home/sprite/.hermes/.env"])

    def test_sync_manager_rolls_back_and_retries_on_delete_failure(self, tmp_path):
        """End-to-end against the real FileSyncManager transaction."""
        from tools.environments.file_sync import FileSyncManager

        secret = tmp_path / "cred.env"
        secret.write_text("KEY=old")
        files = [(str(secret), "/remote/.hermes/cred.env")]

        deletes: list[list[str]] = []
        fail_next = {"flag": False}

        def _delete(paths):
            deletes.append(list(paths))
            if fail_next["flag"]:
                fail_next["flag"] = False
                raise OSError("transient remote failure")

        mgr = FileSyncManager(
            get_files_fn=lambda: list(files),
            upload_fn=lambda h, r: None,
            delete_fn=_delete,
        )
        mgr.sync(force=True)  # baseline: credential synced

        # Rotate the credential away locally; first delete attempt fails.
        files.clear()
        fail_next["flag"] = True
        mgr.sync(force=True)
        assert "/remote/.hermes/cred.env" in mgr._synced_files, (
            "failed deletion must not be committed"
        )
        # Next cycle retries and clears it.
        mgr.sync(force=True)
        assert "/remote/.hermes/cred.env" not in mgr._synced_files
        assert len(deletes) == 2


class TestSpriteNaming:
    """`_resolve_sprite_name`: deterministic, profile-scoped Sprite identity.

    A Sprite is resumed *by name*, so the name is the durable identity of a
    session's live sandbox. The name must (a) stay stable so resume works,
    (b) differ across independent Hermes profiles so they never resume into
    one another's live Sprite, and (c) be injective — no two distinct
    (profile, task) identities may map to one name. Named-profile names are
    pinned to exact literals (including the identity digest) so any change
    to the digest scheme — which silently orphans every live named-profile
    Sprite — fails loudly here.
    """

    @staticmethod
    def _set_profile(monkeypatch, name):
        """Point profile-identity resolution at profile *name* (None=default)."""
        monkeypatch.setattr(
            "sprites_environment._resolve_profile_identity",
            lambda: name,
        )

    def test_default_profile_keeps_legacy_name(self, monkeypatch):
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, None)
        # Backward compatible with Sprites created before profile scoping.
        assert _resolve_sprite_name("default") == "hermes-default"
        assert _resolve_sprite_name("mytask") == "hermes-mytask"

    def test_default_profile_resolution_from_paths(self, monkeypatch, tmp_path):
        """The real path-based resolver maps ~/.hermes and profiles/default → None."""
        import sprites_environment as sprites_mod
        root = tmp_path / ".hermes"
        for home in (root, root / "profiles" / "default"):
            home.mkdir(parents=True, exist_ok=True)
            monkeypatch.setattr(
                "agent.file_safety._hermes_home_path", lambda h=home: h
            )
            monkeypatch.setattr(
                "agent.file_safety._hermes_root_path", lambda r=root: r
            )
            assert sprites_mod._resolve_profile_identity() is None
        # A named profile dir resolves to its name.
        named = root / "profiles" / "work"
        named.mkdir(parents=True)
        monkeypatch.setattr("agent.file_safety._hermes_home_path", lambda: named)
        assert sprites_mod._resolve_profile_identity() == "work"
        # A custom HERMES_HOME outside the tree is its own identity, not default.
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        monkeypatch.setattr("agent.file_safety._hermes_home_path", lambda: outside)
        ident = sprites_mod._resolve_profile_identity()
        assert ident is not None and ident.startswith("home:")

    def test_named_profile_is_scoped(self, monkeypatch):
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "work")
        # Exact literals: profile slug + task slug + 6-hex digest over the
        # raw (profile, task) pair.
        assert _resolve_sprite_name("default") == "hermes-work-default-a092ad600654"
        assert _resolve_sprite_name("mytask") == "hermes-work-mytask-0bba1287573c"

    def test_independent_profiles_do_not_collide(self, monkeypatch):
        """Same task_id under two different profiles → distinct Sprites."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "alpha")
        a = _resolve_sprite_name("default")
        self._set_profile(monkeypatch, "beta")
        b = _resolve_sprite_name("default")
        assert a == "hermes-alpha-default-f763b5cbf547"
        assert b == "hermes-beta-default-8ab8e3ddf43d"
        assert a != b

    def test_component_boundaries_do_not_collide(self, monkeypatch):
        """(profile a-b, task c) and (profile a, task b-c) → distinct names."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "a-b")
        x = _resolve_sprite_name("c")
        self._set_profile(monkeypatch, "a")
        y = _resolve_sprite_name("b-c")
        assert x == "hermes-a-b-c-78208f2b509c"
        assert y == "hermes-a-b-c-590074485363"
        assert x != y

    def test_same_identity_resumes(self, monkeypatch):
        """Same (profile, task_id) is stable across calls → resume works."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "work")
        assert (
            _resolve_sprite_name("t")
            == _resolve_sprite_name("t")
            == "hermes-work-t-4825428fa506"
        )

    def test_names_are_dns_bounded(self, monkeypatch):
        """No generated name exceeds a DNS label; the digest survives intact."""
        import re
        from sprites_environment import (
            _MAX_NAME_LEN,
            _identity_digest,
            _resolve_sprite_name,
        )
        long_home = "home:/Users/someone/Library/Application Support/custom-hermes-home"
        long_task = "session:agent:main:telegram:" + "x" * 60
        cases = [
            (long_home, "default"),
            (long_home, long_task),
            ("work", long_task),
            (None, long_task),  # default profile, oversized session task
        ]
        for profile, task in cases:
            self._set_profile(monkeypatch, profile)
            name = _resolve_sprite_name(task)
            assert len(name) <= _MAX_NAME_LEN, (profile, task, name)
            assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
            # The authoritative digest is the untruncated tail.
            expected_digest = _identity_digest(profile or "", task or "")
            assert name.endswith(expected_digest), (name, expected_digest)

    def test_short_default_profile_names_stay_legacy(self, monkeypatch):
        """The DNS bound must not disturb historical default-profile names."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, None)
        assert _resolve_sprite_name("default") == "hermes-default"
        assert _resolve_sprite_name("mytask") == "hermes-mytask"

    def test_separator_forgery_does_not_collide(self, monkeypatch):
        """Length-prefixed digest: embedded separators cannot forge a boundary."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "a\x1fb")
        x = _resolve_sprite_name("c")
        self._set_profile(monkeypatch, "a")
        y = _resolve_sprite_name("b\x1fc")
        assert x != y

    def test_names_are_sanitized(self, monkeypatch):
        """Messy profile/task components collapse to a Sprite-safe slug.

        Pinned to the exact literal (display slugs + identity digest): a
        silent change to the collapse or digest scheme renames — and
        orphans — every live lossy-named Sprite, so it must fail here.
        """
        import re
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "Team.Prod")
        name = _resolve_sprite_name("sub agent_42")
        assert name == "hermes-team-prod-sub-agent-42-ca9aa4d02adc"
        # Only lowercase alnum + single interior hyphens (Fly/DNS-safe).
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)

    def test_clean_default_profile_tasks_are_unchanged(self, monkeypatch):
        """Default-profile clean task ids keep legacy names (resume works)."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, None)
        assert _resolve_sprite_name("mytask") == "hermes-mytask"

    def test_lossy_profile_slugs_do_not_collide(self, monkeypatch):
        """team_prod and team-prod are distinct profiles → distinct Sprites."""
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, "team_prod")
        a = _resolve_sprite_name("default")
        self._set_profile(monkeypatch, "team-prod")
        b = _resolve_sprite_name("default")
        assert a == "hermes-team-prod-default-e11db62b701d"
        assert b == "hermes-team-prod-default-264eccd5d608"
        assert a != b

    def test_empty_task_id_falls_back(self, monkeypatch):
        from sprites_environment import _resolve_sprite_name
        self._set_profile(monkeypatch, None)
        assert _resolve_sprite_name("") == "hermes-default"

    def test_profile_resolution_failure_fails_closed(self, monkeypatch):
        """A broken HOME/path resolver must not enter the default namespace."""
        import pytest
        import sprites_environment as sprites_mod

        def _boom():
            raise OSError("no home")

        # Break the underlying path resolution the identity is derived from —
        # the exact failure file_safety's own resolver would swallow into
        # "default" (the fail-open this guards against).
        monkeypatch.setattr("agent.file_safety._hermes_home_path", _boom)
        with pytest.raises(RuntimeError, match="refusing to fall back"):
            sprites_mod._resolve_sprite_name("x")


class TestDispatchWiring:
    """Wiring pins: config → terminal_tool dispatch → SpritesEnvironment kwargs.

    The class-level tests above prove SpritesEnvironment honors
    ``persistent_filesystem``; these prove the dispatch actually delivers it.
    A backend missing from terminal_tool's container_config builder gets
    ``container_config=None``, silently re-defaulting ``container_persistent:
    false`` back to persistent — i.e. ephemeral mode can never engage.
    """

    def test_terminal_tool_builds_container_config_for_sprites(self, monkeypatch):
        import __init__ as plugin_pkg
        import tools.terminal_tool as tt
        from agent import terminal_env_registry as reg

        reg._reset_for_tests()
        request_cleanup = reg._reset_for_tests
        try:
            provider = plugin_pkg.SpritesProvider()
            reg.register_provider(provider)
        except Exception:
            request_cleanup()
            raise

        captured = {}

        config = {
            "env_type": "sprites",
            "docker_image": "unused",
            "singularity_image": "unused",
            "modal_image": "unused",
            "daytona_image": "unused",
            "cwd": "/root",
            "host_cwd": None,
            "timeout": 180,
            "lifetime_seconds": 300,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": False,
            "docker_volumes": [],
            "docker_env": {},
            "docker_extra_args": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_run_as_host_user": False,
            "docker_forward_env": [],
            "modal_mode": "auto",
        }

        class _DummyEnv:
            cwd = "/root"

            def execute(self, *a, **k):
                return {"output": "", "exit_code": 0}

        def fake_create_environment(**kwargs):
            captured["container_config"] = kwargs.get("container_config")
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
        monkeypatch.setattr(provider, "create_environment", fake_create_environment)
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        try:
            tt.terminal_tool(command="pwd")
            cc = captured["container_config"]
            assert cc is not None, (
                "container_config=None silently discards container_persistent"
            )
            assert cc["container_persistent"] is False
        finally:
            request_cleanup()

    def test_provider_passes_persistence_and_task_id(self, monkeypatch):
        """The registered provider passes its configuration to the environment."""
        import sprites_environment as sprites_mod
        from agent import terminal_env_registry as reg

        import __init__ as plugin_pkg  # noqa: F401 — the plugin package

        captured = {}

        class _FakeSpritesEnv:
            def __init__(self, cwd, timeout, persistent_filesystem, task_id,
                         name_prefix=None, scope=None, **extra):
                captured["extra"] = extra
                captured.update(
                    cwd=cwd,
                    timeout=timeout,
                    persistent_filesystem=persistent_filesystem,
                    task_id=task_id,
                    name_prefix=name_prefix,
                    scope=scope,
                )

        monkeypatch.setattr(sprites_mod, "SpritesEnvironment", _FakeSpritesEnv)
        reg._reset_for_tests()
        try:
            reg.register_provider(plugin_pkg.SpritesProvider())

            reg.get_provider("sprites").create_environment(
                image="ignored",
                cwd="/root",
                timeout=60,
                container_config={"container_persistent": False},
                task_id="tid-ephemeral",
            )

            assert captured["persistent_filesystem"] is False
            assert captured["task_id"] == "tid-ephemeral"
            assert captured["cwd"] == "/root"
            # No plugin settings configured → today's naming.
            assert captured["name_prefix"] is None
            assert captured["scope"] == "session"
            # Hermes's own ``image`` (the Docker backend's) never reaches the
            # Sprite, and no create settings are passed when none are set.
            assert captured["extra"] == {}
        finally:
            reg._reset_for_tests()

    def test_no_base_url_kwarg(self, make_env, sprites_sdk):
        """SpritesClient is constructed without a base_url override (endpoint is fixed)."""
        make_env(task_id="urlcheck")
        sprites_mod, _ = sprites_sdk
        _, kwargs = sprites_mod.SpritesClient.call_args
        assert "base_url" not in kwargs

    @pytest.mark.parametrize("settings, expected_name", [
        (None, "hermes-default"),
        ({"name_prefix": "hc-abc123", "scope": "instance"}, "hc-abc123-default"),
    ])
    def test_discovered_plugin_executes(
        self, isolated_home, hermes_runtime, sprites_sdk, monkeypatch,
        settings, expected_name,
    ):
        """Exercise discovery, package imports and real terminal dispatch, mocking only transport.

        With settings, this also proves the loader-provided plugin context
        reads ``plugins.entries.sprites.settings`` from config.yaml.
        """
        from hermes_cli import plugins
        from agent import terminal_env_registry as reg
        import tools.terminal_tool as tt

        plugin_dir = isolated_home / "plugins" / "sprites"
        shutil.copytree(
            Path(__file__).resolve().parents[1],
            plugin_dir,
            ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "tests"),
        )
        # JSON is valid YAML; newer Hermes test environments ship no PyYAML.
        plugins_cfg = {"enabled": ["sprites"]}
        if settings is not None:
            plugins_cfg["entries"] = {"sprites": {"settings": settings}}
        (isolated_home / "config.yaml").write_text(json.dumps({
            "plugins": plugins_cfg,
            "terminal": {"backend": "sprites"},
        }))
        monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: isolated_home / "bundled")
        manager = plugins.get_plugin_manager()
        monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
        monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", False)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        sprite = _make_sprite()
        command = MagicMock()
        command.combined_output.return_value = b"from Sprite\n"
        sprite.command.side_effect = [*sprite.command.side_effect, command]
        client = sprites_sdk[0].SpritesClient.return_value
        client.get_sprite.side_effect = _NotFoundError("not found")
        client.create_sprite.return_value = sprite

        reg._reset_for_tests()
        try:
            manager.discover_and_load()
            provider = reg.get_provider("sprites")
            assert provider is not None
            # Verify the package-loaded provider, not the direct-import test shortcut.
            assert provider.__class__.__module__.startswith("hermes_plugins.")

            result = json.loads(tt.terminal_tool("printf 'from Sprite\\n'", task_id="plugin-smoke"))
            assert result["exit_code"] == 0, result
            assert "from Sprite" in result["output"]
            client.create_sprite.assert_called_once_with(expected_name)
        finally:
            for env in tt._active_environments.values():
                env.cleanup()
            manager.unload()
            reg._reset_for_tests()


# ---------------------------------------------------------------------------
# CWD / home detection
# ---------------------------------------------------------------------------

class TestCwdResolution:
    def test_default_cwd_rewrites_to_detected_home(self, make_env):
        env = make_env(task_id="cwd1")  # default cwd="/root"
        assert env.cwd == "/home/sprite"  # rewritten from "/root" → detected home

    def test_tilde_cwd_rewrites_to_detected_home(self, make_env):
        env = make_env(cwd="~", task_id="cwd2")
        assert env.cwd == "/home/sprite"

    def test_explicit_cwd_not_overridden(self, make_env):
        sprite = _make_sprite()
        env = make_env(sprite=sprite, cwd="/workspace", task_id="cwd3")
        assert env.cwd == "/workspace"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_leaves_sprite_alive(self, make_env):
        env = make_env(task_id="persist", persistent_filesystem=True)
        sprite = env._mock_sprite
        env.cleanup()
        sprite.delete.assert_not_called()

    def test_non_persistent_cleanup_deletes_sprite(self, make_env):
        env = make_env(task_id="ephem", persistent_filesystem=False)
        sprite = env._mock_sprite
        env.cleanup()
        sprite.delete.assert_called_once()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(task_id="idem", persistent_filesystem=True)
        env.cleanup()
        env.cleanup()  # second call must not raise

    def test_cleanup_closes_client(self, make_env):
        env = make_env(task_id="closeit", persistent_filesystem=True)
        env.cleanup()
        env._mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# _run_bash exit-code surfacing
# ---------------------------------------------------------------------------

class TestRunBashExitCodes:
    def test_zero_exit_returns_output(self, make_env):
        env = make_env(task_id="rb0")
        # Reset side_effect; new sprite.command() call should return a fresh Cmd.
        cmd = MagicMock()
        cmd.combined_output.return_value = b"hi\n"
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("echo hi", timeout=10)
        handle.wait()
        out = handle.stdout.read()
        assert out == "hi\n"
        assert handle.returncode == 0

    def test_nonzero_exit_surfaces_code_from_ExitError(self, make_env, sprites_sdk):
        env = make_env(task_id="rb7")
        _, exc_mod = sprites_sdk
        cmd = MagicMock()
        cmd.combined_output.side_effect = exc_mod.ExitError(
            "exit status 7", 7, b"before\n", b""
        )
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("exit 7", timeout=10)
        handle.wait()
        out = handle.stdout.read()
        assert "before" in out
        assert handle.returncode == 7

    @pytest.mark.parametrize("timeout_type", [_SpritesTimeoutError, TimeoutError], ids=["sdk", "sync-future"])
    def test_timeout_surfaces_124(self, make_env, sprites_sdk, timeout_type):
        env = make_env(task_id="rbto")
        cmd = MagicMock()
        cmd.combined_output.side_effect = timeout_type("deadline")
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("sleep 999", timeout=1)
        handle.wait()
        assert handle.returncode == 124


# ---------------------------------------------------------------------------
# File-sync push (upload_fn behavior)
# ---------------------------------------------------------------------------

class TestFileSyncPush:
    def test_upload_is_one_fs_write(self, make_env, tmp_path):
        """fs/write creates parents itself; no mkdir/stat (fs/list is gone on 2.0)."""
        env = make_env(task_id="fs")
        host_file = tmp_path / "secret.txt"
        host_file.write_bytes(b"hello")

        remote_path_obj = MagicMock()
        env._fs = MagicMock()
        env._fs.__truediv__.return_value = remote_path_obj

        env._sprite_upload(str(host_file), "/home/sprite/.hermes/new/dir/foo")
        env._fs.__truediv__.assert_called_once_with("/home/sprite/.hermes/new/dir/foo")
        # Same request sprites-py's write_bytes() defaults have always sent.
        remote_path_obj.write_bytes.assert_called_once_with(
            b"hello", mode=0o644, mkdir_parents=True
        )
        assert remote_path_obj.mock_calls == [
            call.write_bytes(b"hello", mode=0o644, mkdir_parents=True)
        ], "upload must not touch parent/mkdir/stat/exists"


class _RateLimitedSprite:
    """Sprite double that counts requests against a per-Sprite limit.

    Every ``fs/write`` and every exec counts, like the Sprites API's
    per-Sprite request limit (300 per minute). Writes land in ``writes``
    with their mode in ``modes``; an unpack exec extracts the written
    archive into ``files`` the way ``tar -xf ... -C /`` would.
    """

    def __init__(self, limit=300, name="hermes-default"):
        self.name = name
        self.limit = limit
        self.requests = 0
        self.writes: dict[str, bytes] = {}
        self.modes: dict[str, int] = {}
        self.files: dict[str, bytes] = {}
        self.unpacks = 0
        self.unpack_exit = 0         # non-zero: the unpack exec fails with it
        self.archive_left = False    # unpack works, archive removal does not
        self.unpack_transport_error = False
        self.fail_file_writes = False

    def _count(self):
        self.requests += 1
        if self.requests > self.limit:
            raise RuntimeError("HTTP 429: rate limited")

    # sprite.filesystem("/") / path -> object with write_bytes()
    def filesystem(self, _root):
        sprite = self

        class _Path:
            def __init__(self, path):
                self.path = path

            def write_bytes(self, data, mode=0o644, mkdir_parents=True):
                sprite._count()
                if sprite.fail_file_writes and ".sync-staging" not in self.path:
                    raise RuntimeError("fs/write failed")
                sprite.writes[self.path] = data
                sprite.modes[self.path] = mode

        class _Fs:
            def __truediv__(self, path):
                return _Path(path)

        return _Fs()

    def command(self, *args, timeout=None, **kwargs):
        import io
        import shlex
        import tarfile

        sprite = self
        script = args[-1]
        cmd = MagicMock()

        def _run():
            if script == "echo $HOME":
                return b"/home/sprite\n"
            sprite._count()
            if "tar -xf" in script:
                sprite.unpacks += 1
                if sprite.unpack_transport_error:
                    raise OSError("websocket dropped")
                if sprite.unpack_exit:
                    raise _ExitError("exit", sprite.unpack_exit, b"tar: write error\n")
                archive = shlex.split(script.split("tar -xf", 1)[1])[0]
                data = sprite.writes[archive]
                with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                    for member in tar.getmembers():
                        sprite.files["/" + member.name] = tar.extractfile(member).read()
                if sprite.archive_left:
                    raise _ExitError("exit", 86, b"rm: Permission denied\n")
                del sprite.writes[archive]
                return b""
            return b"\n__HERMES_CWD_xxx__/home/sprite__HERMES_CWD_xxx__\n"

        cmd.combined_output.side_effect = _run
        return cmd


class TestFileSyncBulkUpload:
    """Changed files travel as archives so a sync costs a few requests, not one per file."""

    @staticmethod
    def _host_tree(tmp_path, count):
        root = tmp_path / "skills"
        files = []
        for i in range(count):
            host = root / f"skill-{i:03d}" / "SKILL.md"
            host.parent.mkdir(parents=True, exist_ok=True)
            host.write_text(f"skill {i}\n")
            files.append((str(host), f"/home/sprite/.hermes/skills/skill-{i:03d}/SKILL.md"))
        return files

    @staticmethod
    def _env(make_env, monkeypatch, files, sprite, task="t"):
        import sprites_environment as se

        monkeypatch.setattr(se, "iter_sync_files", lambda _root: list(files))
        return make_env(task_id=task, sprite=sprite)

    @staticmethod
    def _archives(sprite):
        return [p for p in sprite.writes if ".sync-staging/" in p]

    def test_sync_of_more_files_than_the_request_limit_commits(
        self, make_env, monkeypatch, tmp_path
    ):
        """305 skill files (the count on a live Hermes Cloud instance) against a
        300-request limit: one request per file never finished a sync."""
        files = self._host_tree(tmp_path, 305)
        sprite = _RateLimitedSprite(limit=300)

        env = self._env(make_env, monkeypatch, files, sprite)

        assert set(env._sync_manager._synced_files) == {r for _, r in files}, (
            "initial sync must commit every file")
        for host, remote in files:
            assert sprite.files[remote] == Path(host).read_bytes()
        assert sprite.writes == {}, "the uploaded archive must be consumed"
        # One fs/write + one unpack exec for the whole tree, + init_session.
        assert sprite.requests <= 3

    def test_archive_is_root_only_in_a_staging_dir(self, make_env, monkeypatch, tmp_path):
        """The archive can carry credential files whose own directories are
        more restrictive than anything the archive inherits, so it is 0600."""
        files = self._host_tree(tmp_path, 2)
        sprite = _RateLimitedSprite(limit=10_000)
        sprite.unpack_exit = 2  # keep the archive in ``writes`` for inspection
        self._env(make_env, monkeypatch, files, sprite)
        (archive,) = self._archives(sprite)
        assert archive.startswith("/home/sprite/.hermes/.sync-staging/.sync-")
        assert archive.endswith(".tar")
        assert sprite.modes[archive] == 0o600

    @pytest.mark.parametrize("exit_code", [2, 127])
    def test_failed_unpack_falls_back_to_one_write_per_file(
        self, make_env, monkeypatch, tmp_path, caplog, exit_code
    ):
        """No tar (127), or tar refused (no sudo, full disk): the old path."""
        files = self._host_tree(tmp_path, 3)
        sprite = _RateLimitedSprite(limit=10_000)
        sprite.unpack_exit = exit_code

        with caplog.at_level("WARNING"):
            env = self._env(make_env, monkeypatch, files, sprite)
        assert set(env._sync_manager._synced_files) == {r for _, r in files}
        for host, remote in files:
            assert sprite.writes[remote] == Path(host).read_bytes()
            assert sprite.modes[remote] == 0o644
        assert "tar: write error" in caplog.text
        assert env._use_archives, "an unpack failure alone keeps archives in use"

    def test_failed_fallback_rolls_back_and_next_sync_retries(
        self, make_env, monkeypatch, tmp_path
    ):
        files = self._host_tree(tmp_path, 3)
        sprite = _RateLimitedSprite(limit=10_000)
        sprite.unpack_exit = 2
        sprite.fail_file_writes = True

        env = self._env(make_env, monkeypatch, files, sprite)
        assert env._sync_manager._synced_files == {}, "a failed sync must not commit"

        sprite.unpack_exit = 0
        sprite.fail_file_writes = False
        env._sync_manager.sync(force=True)
        assert set(env._sync_manager._synced_files) == {r for _, r in files}
        assert {r for _, r in files} <= set(sprite.files)

    def test_transport_failure_raises_without_fallback(self, make_env, monkeypatch, tmp_path):
        files = self._host_tree(tmp_path, 3)
        sprite = _RateLimitedSprite(limit=10_000)
        sprite.unpack_transport_error = True

        env = self._env(make_env, monkeypatch, files, sprite)
        assert env._sync_manager._synced_files == {}
        assert not any(r in sprite.writes for _, r in files), "no per-file fallback"

    def test_archive_that_cannot_be_removed_is_blanked_and_archives_stop(
        self, make_env, monkeypatch, tmp_path
    ):
        """A non-root user without sudo cannot delete the root-owned archive:
        empty it, and send this environment's later syncs file by file so
        archives do not pile up."""
        files = self._host_tree(tmp_path, 3)
        sprite = _RateLimitedSprite(limit=10_000)
        sprite.archive_left = True

        env = self._env(make_env, monkeypatch, files, sprite)
        assert set(env._sync_manager._synced_files) == {r for _, r in files}
        (archive,) = self._archives(sprite)
        assert sprite.writes[archive] == b"", "no credential bytes left behind"
        assert env._use_archives is False

        Path(files[0][0]).write_text("changed\n")
        unpacks = sprite.unpacks
        env._sync_manager.sync(force=True)
        assert sprite.unpacks == unpacks, "no further archives"
        assert sprite.writes[files[0][1]] == b"changed\n"
        assert len(self._archives(sprite)) == 1

    def test_later_batches_of_the_same_sync_skip_archives_once_one_is_left(
        self, make_env, monkeypatch, tmp_path
    ):
        import sprites_environment as se

        monkeypatch.setattr(se, "_ARCHIVE_MAX_BYTES", 1)  # one file per archive
        files = self._host_tree(tmp_path, 3)
        sprite = _RateLimitedSprite(limit=10_000)
        sprite.archive_left = True

        env = self._env(make_env, monkeypatch, files, sprite)
        assert set(env._sync_manager._synced_files) == {r for _, r in files}
        assert sprite.unpacks == 1
        assert len(self._archives(sprite)) == 1

    def test_archive_unpacks_with_real_tar(self, tmp_path):
        """The archive unpacks with tar to the remote paths, mode 0644."""
        import stat
        import subprocess

        from sprites_environment import _build_archive

        if shutil.which("tar") is None:
            pytest.skip("tar not available")
        src = tmp_path / "src"
        src.mkdir()
        names = ["plain.txt", "with space.txt", "it's \"quoted\".txt", "-dash", "$(touch pwned)"]
        files = []
        for name in names:
            host = src / name
            host.write_bytes(name.encode() * 3)
            host.chmod(0o600)
            files.append((str(host), f"/home/sprite/.hermes/sub dir/{name}"))
        archive = tmp_path / "a.tar"
        archive.write_bytes(_build_archive(files))
        dest = tmp_path / "root"
        dest.mkdir()
        subprocess.run(
            ["tar", "-xf", str(archive), "-C", str(dest), "--no-same-owner"],
            check=True)
        for host, remote in files:
            out = dest / remote.lstrip("/")
            assert out.read_bytes() == Path(host).read_bytes()
            assert stat.S_IMODE(out.stat().st_mode) == 0o644 & ~_umask()

    @pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
    @pytest.mark.parametrize(
        "tar_ok,sudo_ok,rm_ok,expect_rc",
        [
            (True, False, True, 0),     # root, or the archive was readable
            (False, True, True, 0),     # non-root with sudo
            (False, False, True, 1),    # non-root, sudo refused: non-zero
            (None, False, True, None),  # no tar at all: non-zero
            (True, False, False, 86),   # unpacked, archive not removable
        ],
    )
    def test_unpack_script(self, tmp_path, tar_ok, sudo_ok, rm_ok, expect_rc):
        """tar_ok=None: the Sprite has no tar. Real sh and find, stub tar/sudo/rm."""
        import os
        import subprocess

        from sprites_environment import _unpack_script

        if shutil.which("find") is None:
            pytest.skip("find not available")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "log"

        def stub(name, body):
            path = bindir / name
            path.write_text("#!/bin/sh\n" + body)
            path.chmod(0o755)

        if tar_ok is not None:
            stub("tar", f'echo "tar $*" >> {log}\nexit {0 if tar_ok else 2}\n')
        stub("sudo", f'echo "sudo $*" >> {log}\nexit {0 if sudo_ok else 1}\n')
        if rm_ok:
            (bindir / "rm").symlink_to(shutil.which("rm"))
        else:
            stub("rm", "exit 1\n")
        (bindir / "find").symlink_to(shutil.which("find"))

        staging = tmp_path / "it's staging"
        staging.mkdir()
        archive = staging / ".sync-current.tar"
        archive.write_bytes(b"x")
        stale = staging / ".sync-stale.tar"
        stale.write_bytes(b"old secret")
        os.utime(stale, (0, 0))
        fresh = staging / ".sync-other-live-sync.tar"
        fresh.write_bytes(b"in flight")

        env = {**os.environ, "PATH": str(bindir)}
        rc = subprocess.run(
            [shutil.which("sh"), "-c", _unpack_script(str(archive))], env=env,
            check=False).returncode
        if expect_rc is None:
            assert rc != 0
        else:
            assert rc == expect_rc
        assert fresh.exists(), "another sync's live archive must survive the sweep"
        if rm_ok:
            assert not archive.exists(), "archive must be removed whatever tar did"
            assert not stale.exists(), "a leftover from an earlier sync is swept"
        if tar_ok is None:
            return
        calls = log.read_text()
        assert calls.startswith("tar -xf ")
        assert ("sudo -n tar" in calls) == (not tar_ok)

    def test_archives_split_by_size(self, tmp_path, monkeypatch):
        import sprites_environment as se

        monkeypatch.setattr(se, "_ARCHIVE_MAX_BYTES", 10)
        files = []
        for name, size in [("a", 4), ("b", 4), ("c", 4), ("big", 25), ("d", 1)]:
            host = tmp_path / name
            host.write_bytes(b"x" * size)
            files.append((str(host), f"/r/{name}"))
        batches = se._archive_batches(files)
        assert [[r for _, r in b] for b in batches] == [
            ["/r/a", "/r/b"], ["/r/c"], ["/r/big"], ["/r/d"]]


def _umask() -> int:
    import os
    mask = os.umask(0)
    os.umask(mask)
    return mask


class TestFileSyncDelete:
    """Deletes go through one exec per batch: rm -f, then sudo -n rm -f."""

    @staticmethod
    def _env_with_cmd(make_env, output=b"", exc=None):
        env = make_env(task_id="fsdel")
        cmd = MagicMock()
        if exc is not None:
            cmd.combined_output.side_effect = exc
        else:
            cmd.combined_output.return_value = output
        env._mock_sprite.command = MagicMock(return_value=cmd)
        return env

    def test_one_exec_per_batch_with_sudo_fallback(self, make_env):
        import shlex
        env = self._env_with_cmd(make_env)
        paths = ["/home/sprite/.hermes/a", "/home/sprite/.hermes/b"]
        env._sprite_delete(paths)
        assert env._mock_sprite.command.call_count == 1
        args, kwargs = env._mock_sprite.command.call_args
        assert args[:2] == ("sh", "-c")
        quoted = " ".join(shlex.quote(p) for p in paths)
        assert args[2] == f"rm -f -- {quoted} || sudo -n rm -f -- {quoted}"
        assert kwargs == {"timeout": 60.0}  # bounded; no env on a normal Sprite

    def test_paths_with_spaces_and_quotes_are_quoted(self, make_env, tmp_path):
        import shlex
        import subprocess
        env = self._env_with_cmd(make_env)
        weird = [
            str(tmp_path / "with space.txt"),
            str(tmp_path / "it's \"quoted\".txt"),
            str(tmp_path / "-leading-dash"),
            str(tmp_path / "$(touch pwned)"),
        ]
        for p in weird:
            Path(p).write_text("x")
        env._sprite_delete(weird)
        script = env._mock_sprite.command.call_args[0][2]
        # Each path is exactly one shell word in both halves of the script.
        rm_part = script.split(" || ")[0]
        assert shlex.split(rm_part) == ["rm", "-f", "--", *weird]
        # And the script really removes exactly those files.
        subprocess.run(["sh", "-c", script], check=True, cwd=tmp_path)
        assert not any(Path(p).exists() for p in weird)
        assert not (tmp_path / "pwned").exists()

    def test_missing_file_is_success(self, make_env, tmp_path):
        import subprocess
        env = self._env_with_cmd(make_env)
        env._sprite_delete([str(tmp_path / "never-existed")])
        script = env._mock_sprite.command.call_args[0][2]
        # rm -f exits 0 for a missing path, so the exec succeeds.
        assert subprocess.run(["sh", "-c", script]).returncode == 0

    def test_nonzero_exit_raises(self, make_env):
        env = self._env_with_cmd(
            make_env, exc=_ExitError("exit status 1", 1, b"sudo: a password is required\n")
        )
        with pytest.raises(RuntimeError, match="exit 1.*password is required"):
            env._sprite_delete(["/home/sprite/.hermes/.env"])

    def test_sudo_fallback_runs_when_plain_rm_fails(self, make_env, tmp_path):
        """With rm failing, the script falls back to sudo -n (stubbed here)."""
        import os
        import subprocess
        env = self._env_with_cmd(make_env)
        env._sprite_delete(["/root-owned/file"])
        script = env._mock_sprite.command.call_args[0][2]
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "log"
        (bindir / "rm").write_text(f'#!/bin/sh\necho "rm $*" >> {log}\nexit 1\n')
        (bindir / "sudo").write_text(f'#!/bin/sh\necho "sudo $*" >> {log}\nexit 0\n')
        for f in ("rm", "sudo"):
            (bindir / f).chmod(0o755)
        envvars = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        assert subprocess.run(["sh", "-c", script], env=envvars).returncode == 0
        assert log.read_text().splitlines() == [
            "rm -f -- /root-owned/file",
            "sudo -n rm -f -- /root-owned/file",
        ]

    def test_empty_batch_makes_no_exec(self, make_env):
        env = self._env_with_cmd(make_env)
        env._sprite_delete([])
        env._mock_sprite.command.assert_not_called()

    @pytest.mark.parametrize("name", ["x" * 80, "it's \"q\" " * 8, "skïll-é-日本語" * 4])
    def test_huge_batch_is_split_by_encoded_size(self, make_env, name):
        import shlex
        from urllib.parse import quote
        env = self._env_with_cmd(make_env)
        paths = [f"/home/sprite/.hermes/skills/s{i:04d}/{name}" for i in range(200)]
        env._sprite_delete(paths)
        scripts = [c.args[2] for c in env._mock_sprite.command.call_args_list]
        assert len(scripts) > 1
        # The whole command (both rm halves), as it travels in the exec URL.
        assert all(len(quote(s, safe="")) <= 6000 for s in scripts)
        seen = [p for s in scripts for p in shlex.split(s.split(" || ")[0])[3:]]
        assert seen == paths

    def test_no_filesystem_delete_call(self, make_env):
        env = self._env_with_cmd(make_env)
        env._fs = MagicMock()
        env._sprite_delete(["/home/sprite/.hermes/a"])
        assert env._fs.mock_calls == []


# ---------------------------------------------------------------------------
# _stdin_mode wiring
# ---------------------------------------------------------------------------

class TestStdinMode:
    def test_stdin_mode_is_heredoc(self):
        """Ensures the base class will embed stdin via heredoc, not pipe.

        SDK calls don't accept a real stdin stream, so the backend declares
        ``_stdin_mode = "heredoc"`` and the base ``execute()`` wraps stdin
        into the command string before calling ``_run_bash``.
        """
        # Inspect the class without constructing — no SDK needed for this check
        import importlib

        # Stub the SDK so the module imports cleanly outside the make_env fixture
        sys.modules.setdefault("sprites", types.ModuleType("sprites"))
        sys.modules.setdefault("sprites.exceptions", types.ModuleType("sprites.exceptions"))

        # Force a clean import (in case earlier tests left it in sys.modules with
        # a different SDK mocked in)
        if "sprites_environment" in sys.modules:
            importlib.reload(sys.modules["sprites_environment"])
        from sprites_environment import SpritesEnvironment

        assert SpritesEnvironment._stdin_mode == "heredoc"


# ---------------------------------------------------------------------------
# Plugin settings: name_prefix and scope
# ---------------------------------------------------------------------------

_DNS_LABEL = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_LONG_TASK = "session:agent:main:telegram:dm:" + "x" * 80
_LONG_PREFIX = "hc-" + "a" * 37  # exactly the 40-char maximum


def _set_profile(monkeypatch, name):
    monkeypatch.setattr(
        "sprites_environment._resolve_profile_identity", lambda: name
    )


class TestNamingSettingsValidation:
    """Invalid settings fail closed; unset settings mean today's behaviour."""

    @pytest.mark.parametrize("prefix", [None, ""])
    @pytest.mark.parametrize("scope", [None, ""])
    def test_unset_means_legacy(self, prefix, scope):
        from sprites_environment import resolve_naming_settings
        assert resolve_naming_settings(prefix, scope) == (None, "session")

    @pytest.mark.parametrize("prefix", ["hc-abc123", "a", "hc-0", _LONG_PREFIX])
    def test_valid_prefixes(self, prefix):
        from sprites_environment import resolve_naming_settings
        assert resolve_naming_settings(prefix, "instance") == (prefix, "instance")

    @pytest.mark.parametrize("prefix", [
        "HC-abc", "hc_abc", "hc.abc", "-hc", "hc-", "hc--abc", "hc abc",
        "hç-abc", _LONG_PREFIX + "b", 123, True, ["hc"],
    ])
    def test_invalid_prefix_raises(self, prefix):
        from sprites_environment import resolve_naming_settings
        with pytest.raises(ValueError, match="name_prefix"):
            resolve_naming_settings(prefix, None)

    @pytest.mark.parametrize("scope", ["instnace", "Instance", "profile", 1, True])
    def test_invalid_scope_raises(self, scope):
        from sprites_environment import resolve_naming_settings
        with pytest.raises(ValueError, match="scope"):
            resolve_naming_settings(None, scope)


class TestPrefixedNames:
    def test_unset_prefix_is_byte_identical(self, monkeypatch):
        """Regression: no prefix → exactly the historical names."""
        from sprites_environment import _resolve_sprite_name
        _set_profile(monkeypatch, None)
        assert _resolve_sprite_name("default", None) == "hermes-default"
        assert _resolve_sprite_name("mytask", None) == "hermes-mytask"
        assert _resolve_sprite_name(_LONG_TASK, None) == _resolve_sprite_name(_LONG_TASK)
        _set_profile(monkeypatch, "work")
        assert _resolve_sprite_name("default", None) == "hermes-work-default-a092ad600654"

    def test_prefix_replaces_hermes_head(self, monkeypatch):
        from sprites_environment import _ephemeral_sprite_name, _resolve_sprite_name
        _set_profile(monkeypatch, None)
        assert _resolve_sprite_name("default", "hc-abc123") == "hc-abc123-default"
        assert _resolve_sprite_name("mytask", "hc-abc123") == "hc-abc123-mytask"
        assert _ephemeral_sprite_name("mine", "hc-abc123").startswith("hc-abc123-eph-mine-")

    def test_named_profile_stays_digest_scoped(self, monkeypatch):
        """The prefix swaps the head only; the identity digest is unchanged."""
        from sprites_environment import _resolve_sprite_name
        _set_profile(monkeypatch, "work")
        assert _resolve_sprite_name("default", "hc-abc123") == "hc-abc123-work-default-a092ad600654"
        _set_profile(monkeypatch, "play")
        assert _resolve_sprite_name("default", "hc-abc123") != "hc-abc123-work-default-a092ad600654"

    @pytest.mark.parametrize("profile", [None, "work", "home:/a/very/long/custom/hermes/home/path"])
    @pytest.mark.parametrize("task", ["default", "t", _LONG_TASK, "Sub Agent_42"])
    @pytest.mark.parametrize("prefix", ["h", "hc-abc123", _LONG_PREFIX])
    def test_dns_bound_and_prefix_intact(self, monkeypatch, profile, task, prefix):
        import re
        from sprites_environment import (
            _MAX_NAME_LEN, _ephemeral_sprite_name, _resolve_sprite_name,
        )
        _set_profile(monkeypatch, profile)
        for name in (_resolve_sprite_name(task, prefix), _ephemeral_sprite_name(task, prefix)):
            assert len(name) <= _MAX_NAME_LEN, name
            assert re.fullmatch(_DNS_LABEL, name), name
            assert name.startswith(f"{prefix}-"), name

    def test_long_prefix_keeps_full_digest(self, monkeypatch):
        from sprites_environment import _identity_digest, _resolve_sprite_name
        _set_profile(monkeypatch, "work")
        name = _resolve_sprite_name(_LONG_TASK, _LONG_PREFIX)
        assert name.endswith(_identity_digest("work", _LONG_TASK))


class TestPrefixAndScopeInEnvironment:
    def test_no_settings_unchanged(self, make_env):
        env = make_env(task_id="fresh")
        env._mock_client.get_sprite.assert_called_once_with("hermes-fresh")
        env._mock_client.create_sprite.assert_called_once_with("hermes-fresh")

    def test_prefix_persistent(self, make_env):
        env = make_env(task_id="fresh", name_prefix="hc-abc123",
                       sprite=_make_sprite("hc-abc123-fresh"))
        env._mock_client.get_sprite.assert_called_once_with("hc-abc123-fresh")
        env._mock_client.create_sprite.assert_called_once_with("hc-abc123-fresh")

    def test_prefix_ephemeral(self, make_env):
        env = make_env(task_id="mine", name_prefix="hc-abc123",
                       persistent_filesystem=False)
        (name,), _ = env._mock_client.create_sprite.call_args
        assert name.startswith("hc-abc123-eph-mine-")
        env._mock_client.get_sprite.assert_not_called()

    @pytest.mark.parametrize("bad", ["Bad_Prefix", "hc--x", 42])
    def test_invalid_prefix_fails_closed_before_any_api_call(self, make_env, sprites_sdk, bad):
        with pytest.raises(ValueError, match="name_prefix"):
            make_env(task_id="fresh", name_prefix=bad)
        sprites_sdk[0].SpritesClient.assert_not_called()

    def test_invalid_scope_fails_closed(self, make_env, sprites_sdk):
        with pytest.raises(ValueError, match="scope"):
            make_env(task_id="fresh", scope="instnace")
        sprites_sdk[0].SpritesClient.assert_not_called()

    def test_instance_scope_collapses_task_ids(self, make_env):
        names = set()
        for task in ("session:agent:main:telegram:dm:1", "cron-job-7", "default", _LONG_TASK):
            env = make_env(task_id=task, name_prefix="hc-abc123", scope="instance",
                           sprite=_make_sprite("hc-abc123-default"))
            (name,), _ = env._mock_client.get_sprite.call_args
            names.add(name)
        assert names == {"hc-abc123-default"}

    def test_instance_scope_without_prefix_uses_default_profile_name(self, make_env):
        env = make_env(task_id="session:agent:main:x", scope="instance")
        env._mock_client.get_sprite.assert_called_once_with("hermes-default")

    def test_instance_scope_named_profile_is_digest_scoped(self, make_env, monkeypatch):
        _set_profile(monkeypatch, "work")
        env = make_env(task_id="session:abc", name_prefix="hc-abc123", scope="instance",
                       sprite=_make_sprite("hc-abc123-work-default-a092ad600654"))
        env._mock_client.get_sprite.assert_called_once_with("hc-abc123-work-default-a092ad600654")

    def test_instance_scope_keeps_ephemeral_runs_unique(self, make_env):
        a = make_env(task_id="s1", name_prefix="hc-abc123", scope="instance",
                     persistent_filesystem=False)
        b = make_env(task_id="s1", name_prefix="hc-abc123", scope="instance",
                     persistent_filesystem=False)
        assert a._sprite_name != b._sprite_name
        assert a._sprite_name.startswith("hc-abc123-eph-")

    @pytest.mark.parametrize("persistent", [True, False])
    def test_prefix_guard_refuses_out_of_prefix_names(self, make_env, monkeypatch, persistent):
        """Even if name resolution regresses, nothing outside the prefix is touched."""
        monkeypatch.setattr("sprites_environment._resolve_sprite_name",
                            lambda task_id, prefix=None: "hermes-default")
        monkeypatch.setattr("sprites_environment._ephemeral_sprite_name",
                            lambda task_id, prefix=None: "hermes-eph-default-0123456789ab")
        with pytest.raises(RuntimeError, match="outside the configured name_prefix"):
            make_env(task_id="x", name_prefix="hc-abc123",
                     persistent_filesystem=persistent)
        client = sys.modules["sprites"].SpritesClient.return_value
        client.get_sprite.assert_not_called()
        client.create_sprite.assert_not_called()

    def test_prefix_guard_rejects_overlong_names(self):
        from sprites_environment import _require_within_prefix
        _require_within_prefix("hc-abc123-default", "hc-abc123")
        _require_within_prefix("anything", None)
        with pytest.raises(RuntimeError):
            _require_within_prefix("hc-abc123-" + "x" * 60, "hc-abc123")
        with pytest.raises(RuntimeError):
            _require_within_prefix("hc-abc1234-default", "hc-abc123")


class TestProviderSettings:
    """The provider reads plugins.entries.<id>.settings and passes them on."""

    @staticmethod
    def _capture(monkeypatch):
        import sprites_environment as sprites_mod
        captured = {}

        class _FakeSpritesEnv:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(sprites_mod, "SpritesEnvironment", _FakeSpritesEnv)
        return captured

    @staticmethod
    def _write_settings(home, settings, plugin_id="sprites", subtree="settings"):
        (home / "config.yaml").write_text(json.dumps({
            "plugins": {"entries": {plugin_id: {subtree: settings}}},
        }))

    def test_settings_from_config_yaml(self, monkeypatch, isolated_home):
        import __init__ as plugin_pkg
        captured = self._capture(monkeypatch)
        self._write_settings(isolated_home, {"name_prefix": "hc-file", "scope": "instance"})
        plugin_pkg.SpritesProvider().create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured["name_prefix"] == "hc-file"
        assert captured["scope"] == "instance"

    def test_legacy_config_subtree_is_honoured(self, monkeypatch, isolated_home):
        """Same lookup order as PluginContext.get_config (settings, then config)."""
        import __init__ as plugin_pkg
        captured = self._capture(monkeypatch)
        self._write_settings(isolated_home, {"name_prefix": "hc-legacy"}, subtree="config")
        plugin_pkg.SpritesProvider().create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured["name_prefix"] == "hc-legacy"

    def test_register_uses_loader_plugin_id(self, monkeypatch, isolated_home):
        import __init__ as plugin_pkg
        captured = self._capture(monkeypatch)
        self._write_settings(isolated_home, {"name_prefix": "hc-ctx"}, plugin_id="sprites-dev")
        registered = []

        class _Ctx:
            plugin_id = "sprites-dev"

            def register_terminal_environment_provider(self, provider):
                registered.append(provider)

        plugin_pkg.register(_Ctx())
        registered[0].create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured["name_prefix"] == "hc-ctx"
        assert captured["scope"] == "session"

    def test_invalid_setting_refuses_to_create(self, monkeypatch, isolated_home):
        import __init__ as plugin_pkg
        captured = self._capture(monkeypatch)
        self._write_settings(isolated_home, {"name_prefix": "Not Valid"})
        with pytest.raises(ValueError, match="name_prefix"):
            plugin_pkg.SpritesProvider().create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured == {}

    def test_doctor_reports_naming(self, isolated_home):
        import __init__ as plugin_pkg
        self._write_settings(isolated_home, {"name_prefix": "hc-abc123", "scope": "instance"})
        row = [r for r in plugin_pkg.SpritesProvider().doctor_checks() if r[1] == "Sprites naming"][0]
        assert row[0] is True and "hc-abc123-*" in row[2] and "instance" in row[2]
        self._write_settings(isolated_home, {"scope": "x"})
        row = [r for r in plugin_pkg.SpritesProvider().doctor_checks() if r[1] == "Sprites naming"][0]
        assert row[0] is False and "scope" in row[2]

    def test_manifest_declares_settings_for_plugins_hub(self):
        """plugin.yaml's config_schema renders both settings in the Plugins hub."""
        hub = pytest.importorskip("hermes_cli.plugins_settings")
        plugin_dir = Path(__file__).resolve().parents[1]
        fields = {f["key"]: f for f in hub.plugin_settings_fields("sprites", plugin_dir)}
        assert fields["name_prefix"]["type"] == "string"
        assert fields["scope"]["type"] == "enum"
        assert fields["scope"]["choices"] == ["session", "instance"]
        assert fields["scope"]["value"] == "session"

    _BROKEN = (
        "plugins:\n  entries:\n    sprites:\n      settings:\n"
        "        name_prefix: hc-abc123\n        scope: [instance\n"
    )

    def test_unreadable_config_refuses_to_create(self, monkeypatch, isolated_home):
        """Core serves fallback values for a malformed config.yaml; never name Sprites from them."""
        import __init__ as plugin_pkg
        if plugin_pkg._failed_read_marker() is None:
            pytest.skip("this Hermes has no fallback-config marker")
        captured = self._capture(monkeypatch)
        (isolated_home / "config.yaml").write_text(self._BROKEN)
        provider = plugin_pkg.SpritesProvider()
        with pytest.raises(RuntimeError, match="config.yaml could not be read"):
            provider.create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured == {}
        row = [r for r in provider.doctor_checks() if r[1] == "Sprites naming"][0]
        assert row[0] is False

    def test_config_broken_after_good_load_refuses(self, monkeypatch, isolated_home):
        """A last-known-good fallback (here: no prefix) must not be used either."""
        import __init__ as plugin_pkg
        if plugin_pkg._failed_read_marker() is None:
            pytest.skip("this Hermes has no fallback-config marker")
        captured = self._capture(monkeypatch)
        cfg = isolated_home / "config.yaml"
        cfg.write_text(json.dumps({"terminal": {"backend": "sprites"}}))
        provider = plugin_pkg.SpritesProvider()
        provider.create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured["name_prefix"] is None
        captured.clear()
        cfg.write_text("terminal: {backend: sprites}\n" + self._BROKEN)
        with pytest.raises(RuntimeError, match="config.yaml could not be read"):
            provider.create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured == {}

    def test_settings_come_from_one_checked_snapshot(self, monkeypatch):
        """A fallback read followed by a clean one must not yield the fallback's values."""
        import __init__ as plugin_pkg

        class _Fallback(dict):
            read_error = ValueError("bad yaml")

        reads = iter([
            _Fallback({}),  # fallback without the prefix
            {"plugins": {"entries": {"sprites": {"settings": {
                "name_prefix": "hc-new", "scope": "instance"}}}}},
        ])
        monkeypatch.setattr(plugin_pkg, "_failed_read_marker", lambda: _Fallback)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: next(reads))
        with pytest.raises(RuntimeError, match="config.yaml could not be read"):
            plugin_pkg.SpritesProvider()._naming_settings()

    def test_core_without_marker_parses_the_file(self, monkeypatch, isolated_home):
        """Older Hermes falls back silently; the plugin then checks the file itself."""
        import __init__ as plugin_pkg
        captured = self._capture(monkeypatch)
        monkeypatch.setattr(plugin_pkg, "_failed_read_marker", lambda: None)
        self._write_settings(isolated_home, {"name_prefix": "hc-old"})
        provider = plugin_pkg.SpritesProvider()
        provider.create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured["name_prefix"] == "hc-old"
        captured.clear()
        (isolated_home / "config.yaml").write_text(self._BROKEN)
        with pytest.raises(RuntimeError, match="config.yaml could not be read"):
            provider.create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured == {}


class TestSharedSpriteFileSync:
    """Environments resuming one Sprite must not leave an older file copy behind."""

    def test_concurrent_sync_keeps_newest_file(self, make_env, monkeypatch, tmp_path):
        import threading
        import sprites_environment as se

        host = tmp_path / "credential.json"
        host.write_bytes(b"old credential")
        remote_path = "/home/sprite/.hermes/credential.json"
        monkeypatch.setattr(se, "iter_sync_files", lambda _root: [(str(host), remote_path)])
        monkeypatch.setattr(
            "tools.environments.file_sync._credential_host_paths", lambda: set(), raising=False)

        remote = {}
        a_uploading = threading.Event()
        release_a = threading.Event()

        def _upload(self, host_path, rp):
            data = Path(host_path).read_bytes()
            if self._task_id == "a":
                a_uploading.set()
                assert release_a.wait(10)
            remote[rp] = data

        def _bulk_upload(self, files):
            for host_path, rp in files:
                _upload(self, host_path, rp)

        monkeypatch.setattr(se.SpritesEnvironment, "_sprite_upload", _upload)
        monkeypatch.setattr(se.SpritesEnvironment, "_sprite_bulk_upload", _bulk_upload)
        errors = []

        def _build(task):
            try:
                make_env(task_id=task, name_prefix="hc-abc123", scope="instance",
                         sprite=_make_sprite("hc-abc123-default"))
            except BaseException as e:  # surfaced below
                errors.append(e)

        a = threading.Thread(target=_build, args=("a",))
        a.start()
        assert a_uploading.wait(10)
        # A has read the old bytes and is mid-upload; the host file changes and
        # B starts its own first sync of the same Sprite.
        host.write_bytes(b"new credential version")
        b = threading.Thread(target=_build, args=("b",))
        b.start()
        b.join(1.0)
        release_a.set()
        a.join(10)
        b.join(10)
        assert not errors, errors
        assert remote[remote_path] == b"new credential version"


# ---------------------------------------------------------------------------
# Create-time settings: image, max_cpus, max_ram_gb, max_disk_gb
# ---------------------------------------------------------------------------

class TestCreateSettingsValidation:
    @pytest.mark.parametrize("kwargs", [{}, {"image": None}, {"image": "", "max_cpus": None}])
    def test_unset_means_no_config(self, kwargs):
        from sprites_environment import resolve_create_settings
        assert resolve_create_settings(**kwargs) == {}

    @pytest.mark.parametrize("image", [
        "ubuntu:24.04",
        "nousresearch/hermes-sandbox:v0.21.7-desktop",
        "ghcr.io/acme/agent-env:1.4",
        "localhost:5000/team/img:2026.10",
        "ubuntu@sha256:" + "a" * 64,
        "ubuntu:latest@sha256:" + "b" * 64,  # a digest pins even with a latest tag
        "ubuntu@sha512:" + "c" * 128,
    ])
    def test_valid_images(self, image):
        from sprites_environment import resolve_create_settings
        assert resolve_create_settings(image=image) == {"image_url": image}

    @pytest.mark.parametrize("image", [
        "ubuntu", "ubuntu:latest", "ghcr.io/acme/img:latest", "localhost:5000/img",
        "ubuntu:", "@sha256:abc", "ubuntu@abc", " ubuntu:24.04", "ubuntu :24.04",
        "ubuntu@sha256:", "ubuntu@sha256:not-a-digest", "ubuntu:24.04@", "ubuntu::24.04",
        "Ubuntu:24.04", "ubuntu:24.04@sha256:" + "a" * 64 + "@x", "/ubuntu:24.04",
        "acme//img:1", "ubuntu:-bad", "ubuntu:" + "a" * 129,
        "ubuntu@sha256:" + "a" * 32, "ubuntu@sha256:" + "a" * 63, "ubuntu@sha256:" + "a" * 65,
        "ubuntu@sha512:" + "a" * 64,
        42, True, ["ubuntu:24.04"],
    ])
    def test_invalid_images(self, image):
        from sprites_environment import resolve_create_settings
        with pytest.raises(ValueError, match="image"):
            resolve_create_settings(image=image)

    @pytest.mark.parametrize("key, value", [
        ("max_cpus", 1), ("max_cpus", 128), ("max_ram_gb", 1), ("max_ram_gb", 1024),
        ("max_disk_gb", 20), ("max_disk_gb", 1024),
    ])
    def test_valid_sizes(self, key, value):
        from sprites_environment import resolve_create_settings
        assert resolve_create_settings(**{key: value}) == {key: value}

    @pytest.mark.parametrize("key, value", [
        ("max_cpus", 0), ("max_cpus", 129), ("max_cpus", -1), ("max_cpus", 2.5),
        ("max_cpus", "2"), ("max_cpus", True), ("max_ram_gb", 0), ("max_ram_gb", 1025),
        ("max_disk_gb", 0), ("max_disk_gb", 2048), ("max_disk_gb", False),
    ])
    def test_invalid_sizes(self, key, value):
        from sprites_environment import resolve_create_settings
        with pytest.raises(ValueError, match=key):
            resolve_create_settings(**{key: value})

    def test_only_set_keys_are_returned(self):
        from sprites_environment import resolve_create_settings
        assert resolve_create_settings(max_ram_gb=4, max_cpus=2) == {
            "max_cpus": 2, "max_ram_gb": 4,
        }


def _create_response(status=201, name="hermes-fresh", text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.is_success = 200 <= status < 300
    resp.text = text
    resp.json.return_value = {"name": name}
    return resp


class TestCreateWithSettings:
    def test_no_settings_calls_sdk_create_unchanged(self, make_env):
        """No create settings → the SDK's plain create_sprite(name), nothing else."""
        env = make_env(task_id="fresh")
        assert env._mock_client.create_sprite.call_args_list == [call("hermes-fresh")]
        env._mock_client.http_client.post.assert_not_called()
        assert env._create_config == {}

    def test_no_settings_ephemeral_calls_sdk_create_unchanged(self, make_env):
        env = make_env(task_id="mine", persistent_filesystem=False)
        (name,), kwargs = env._mock_client.create_sprite.call_args
        assert kwargs == {} and name.startswith("hermes-eph-mine-")
        env._mock_client.http_client.post.assert_not_called()

    def _build(self, sprites_sdk, monkeypatch, response, get_side_effect=None, **kwargs):
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        sprite = _make_sprite("hermes-fresh")
        client = MagicMock()
        client.base_url = "https://api.sprites.dev"
        client.http_client.post.return_value = response
        state = {"n": 0}

        def _get(name):
            state["n"] += 1
            if state["n"] == 1:
                raise _NotFoundError("not found")
            return sprite

        client.get_sprite.side_effect = get_side_effect or _get
        sprites_sdk[0].SpritesClient = MagicMock(return_value=client)
        from sprites_environment import SpritesEnvironment
        env = SpritesEnvironment(task_id="fresh", **kwargs)
        return env, client

    def test_subset_config_in_json_body(self, sprites_sdk, hermes_runtime, monkeypatch):
        env, client = self._build(
            sprites_sdk, monkeypatch, _create_response(),
            max_cpus=2, max_disk_gb=20,
        )
        client.create_sprite.assert_not_called()
        (url,), kwargs = client.http_client.post.call_args
        assert url == "https://api.sprites.dev/v1/sprites"
        assert kwargs["json"] == {"name": "hermes-fresh",
                                  "config": {"max_cpus": 2, "max_disk_gb": 20}}
        assert kwargs["timeout"] == 240
        # The handle comes from the SDK (GET after create).
        assert client.get_sprite.call_args_list == [call("hermes-fresh"), call("hermes-fresh")]
        assert env._sprite.name == "hermes-fresh"

    def test_full_config_in_json_body(self, sprites_sdk, hermes_runtime, monkeypatch):
        env, client = self._build(
            sprites_sdk, monkeypatch, _create_response(),
            image="nousresearch/hermes-sandbox:v0.21.7-desktop",
            max_cpus=2, max_ram_gb=4, max_disk_gb=20,
        )
        assert client.http_client.post.call_args.kwargs["json"] == {
            "name": "hermes-fresh",
            "config": {"image_url": "nousresearch/hermes-sandbox:v0.21.7-desktop",
                       "max_cpus": 2, "max_ram_gb": 4, "max_disk_gb": 20},
        }

    def test_ephemeral_create_uses_config(self, sprites_sdk, hermes_runtime, monkeypatch):
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        client = MagicMock()
        client.base_url = "https://api.sprites.dev"
        client.http_client.post.return_value = _create_response()
        client.get_sprite.return_value = _make_sprite("hermes-eph")
        sprites_sdk[0].SpritesClient = MagicMock(return_value=client)
        from sprites_environment import SpritesEnvironment
        SpritesEnvironment(task_id="mine", persistent_filesystem=False, max_ram_gb=8)
        body = client.http_client.post.call_args.kwargs["json"]
        assert body["name"].startswith("hermes-eph-mine-")
        assert body["config"] == {"max_ram_gb": 8}
        client.create_sprite.assert_not_called()

    def test_existing_sprite_is_used_as_is(self, sprites_sdk, hermes_runtime, monkeypatch, caplog):
        """Settings never change an existing Sprite (no create, no PUT config)."""
        existing = _make_sprite("hermes-fresh")
        with caplog.at_level("DEBUG", logger="sprites_environment"):
            env, client = self._build(
                sprites_sdk, monkeypatch, _create_response(),
                get_side_effect=lambda name: existing, image="ubuntu:24.04",
            )
        client.http_client.post.assert_not_called()
        client.http_client.put.assert_not_called()
        client.create_sprite.assert_not_called()
        assert "apply only when a Sprite is created" in caplog.text

    def test_name_taken_adopts_winner(self, sprites_sdk, hermes_runtime, monkeypatch):
        """A 409 from the direct create keeps the race-adopt semantics."""
        env, client = self._build(
            sprites_sdk, monkeypatch, _create_response(409, text="name already exists"),
            max_cpus=2,
        )
        assert env._sprite.name == "hermes-fresh"
        assert client.get_sprite.call_count == 2

    @pytest.mark.parametrize("status, exc_type", [
        (400, _SpriteError), (401, _AuthenticationError), (403, _SpriteError),
        (404, _NotFoundError), (409, _SpriteError), (429, _SpriteError), (500, _SpriteError),
    ])
    def test_genuine_create_error_raises(self, sprites_sdk, hermes_runtime, monkeypatch,
                                         status, exc_type):
        """Same exception types as the SDK's create_sprite (_handle_response)."""
        with pytest.raises(_SpriteError, match="create sprite") as info:
            self._build(
                sprites_sdk, monkeypatch, _create_response(status, text="quota exceeded"),
                get_side_effect=_NotFoundError("not found"), max_cpus=2,
            )
        assert type(info.value) is exc_type

    def test_network_error_raises_sprite_error(self, sprites_sdk, hermes_runtime, monkeypatch):
        import httpx
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        client = MagicMock()
        client.base_url = "https://api.sprites.dev"
        client.http_client.post.side_effect = httpx.ConnectError("refused")
        client.get_sprite.side_effect = _NotFoundError("not found")
        sprites_sdk[0].SpritesClient = MagicMock(return_value=client)
        from sprites_environment import SpritesEnvironment
        with pytest.raises(_NetworkError, match="refused"):
            SpritesEnvironment(task_id="fresh", image="ubuntu:24.04")

    _STILL_CREATING = ("Sprite {name} is still being created (the first pull of image "
                       "ubuntu:24.04 can take a few minutes); retry the command shortly.")

    def _timeout_client(self, sprites_sdk, monkeypatch):
        import httpx
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        client = MagicMock()
        client.base_url = "https://api.sprites.dev"
        client.http_client.post.side_effect = httpx.ReadTimeout("timed out")
        client.get_sprite.side_effect = _NotFoundError("not found")
        sprites_sdk[0].SpritesClient = MagicMock(return_value=client)
        return client

    def test_persistent_create_timeout_is_retryable(self, sprites_sdk, hermes_runtime, monkeypatch):
        """A create that outlives the wait ends in one clear retry message."""
        client = self._timeout_client(sprites_sdk, monkeypatch)
        from sprites_environment import SpritesEnvironment, _CREATE_WITH_CONFIG_TIMEOUT
        assert _CREATE_WITH_CONFIG_TIMEOUT == 240  # under Hermes's 420 s tool deadline
        with pytest.raises(TimeoutError) as info:
            SpritesEnvironment(task_id="fresh", image="ubuntu:24.04")
        assert str(info.value) == self._STILL_CREATING.format(name="hermes-fresh")
        assert not isinstance(info.value, _SpriteError)
        # No re-GET of a half-created Sprite in the same call.
        assert client.get_sprite.call_args_list == [call("hermes-fresh")]

    def test_next_construction_adopts_finished_sprite(self, sprites_sdk, hermes_runtime, monkeypatch):
        """After a timed-out create, the retry's first GET resumes the Sprite."""
        client = self._timeout_client(sprites_sdk, monkeypatch)
        from sprites_environment import SpritesEnvironment
        with pytest.raises(TimeoutError):
            SpritesEnvironment(task_id="fresh", image="ubuntu:24.04")
        finished = _make_sprite("hermes-fresh")
        client.get_sprite.side_effect = lambda name: finished
        client.http_client.post.reset_mock()
        env = SpritesEnvironment(task_id="fresh", image="ubuntu:24.04")
        assert env._sprite is finished
        client.http_client.post.assert_not_called()

    def test_ephemeral_create_timeout_is_retryable_and_warns(
        self, sprites_sdk, hermes_runtime, monkeypatch, caplog,
    ):
        client = self._timeout_client(sprites_sdk, monkeypatch)
        from sprites_environment import SpritesEnvironment
        with caplog.at_level("WARNING", logger="sprites_environment"):
            with pytest.raises(TimeoutError) as info:
                SpritesEnvironment(task_id="mine", persistent_filesystem=False,
                                   image="ubuntu:24.04")
        name = client.http_client.post.call_args.kwargs["json"]["name"]
        assert str(info.value) == self._STILL_CREATING.format(name=name)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and name in warnings[0].getMessage()
        assert "orphaned" in warnings[0].getMessage()
        client.get_sprite.assert_not_called()

    def test_sizing_only_timeout_message_omits_image(self, sprites_sdk, hermes_runtime, monkeypatch):
        self._timeout_client(sprites_sdk, monkeypatch)
        from sprites_environment import SpritesEnvironment
        with pytest.raises(TimeoutError, match=r"^Sprite hermes-fresh is still being "
                                               r"created; retry the command shortly\.$"):
            SpritesEnvironment(task_id="fresh", max_cpus=2)

    def test_invalid_settings_fail_before_any_api_call(self, make_env, sprites_sdk):
        with pytest.raises(ValueError, match="latest"):
            make_env(task_id="fresh", image="ubuntu:latest")
        sprites_sdk[0].SpritesClient.assert_not_called()


# ---------------------------------------------------------------------------
# HOME fallback for images that run commands with $HOME unset
# ---------------------------------------------------------------------------

def _sprite_with_home(home: bytes, uid: bytes = b"0\n"):
    sprite = MagicMock()
    sprite.name = "hermes-default"
    outputs = {"echo $HOME": home, "id -u": uid}

    def _command(*args, **kwargs):
        cmd = MagicMock()
        cmd.combined_output.return_value = outputs.get(
            args[-1], b"\n__HERMES_CWD_xxx__/root__HERMES_CWD_xxx__\n"
        )
        return cmd

    sprite.command = MagicMock(side_effect=_command)
    sprite.filesystem.return_value = MagicMock()
    return sprite


class TestHomeFallback:
    def test_default_image_passes_no_env(self, make_env):
        env = make_env(task_id="h")
        assert env._remote_home == "/home/sprite"
        assert env._exec_env is None
        for c in env._mock_sprite.command.call_args_list:
            assert "env" not in c.kwargs
        cmd = MagicMock()
        cmd.combined_output.return_value = b"ok"
        env._mock_sprite.command = MagicMock(return_value=cmd)
        env._run_bash("true", timeout=5).wait()
        assert "env" not in env._mock_sprite.command.call_args.kwargs

    def test_empty_home_as_root_sets_home_env(self, make_env):
        sprite = _sprite_with_home(b"\n", b"0\n")
        env = make_env(task_id="h", sprite=sprite)
        assert env._remote_home == "/root"
        assert env.cwd == "/root"
        assert env._exec_env == {"HOME": "/root"}
        # init_session's bootstrap (after detection) already carries HOME.
        assert sprite.command.call_args_list[-1].kwargs.get("env") == {"HOME": "/root"}
        cmd = MagicMock()
        cmd.combined_output.return_value = b"ok"
        sprite.command = MagicMock(return_value=cmd)
        env._run_bash("true", timeout=5).wait()
        assert sprite.command.call_args.kwargs["env"] == {"HOME": "/root"}
        env._sprite_delete(["/root/.hermes/x"])
        assert sprite.command.call_args.kwargs["env"] == {"HOME": "/root"}

    def test_empty_home_non_root_keeps_previous_behaviour(self, make_env):
        sprite = _sprite_with_home(b"\n", b"1000\n")
        env = make_env(task_id="h", sprite=sprite)
        assert env._remote_home == "/root"
        assert env._exec_env is None

    def test_sync_target_follows_fallback_home(self, make_env):
        sprite = _sprite_with_home(b"", b"0\n")
        env = make_env(task_id="h", sprite=sprite)
        assert env._remote_home == "/root"


# ---------------------------------------------------------------------------
# Provider plumbing for the create settings
# ---------------------------------------------------------------------------

class TestProviderCreateSettings:
    def test_create_settings_from_config_yaml(self, monkeypatch, isolated_home):
        import __init__ as plugin_pkg
        captured = TestProviderSettings._capture(monkeypatch)
        TestProviderSettings._write_settings(isolated_home, {
            "image": "ubuntu:24.04", "max_cpus": 2, "max_ram_gb": 4, "max_disk_gb": 20,
        })
        plugin_pkg.SpritesProvider().create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured["image"] == "ubuntu:24.04"
        assert (captured["max_cpus"], captured["max_ram_gb"], captured["max_disk_gb"]) == (2, 4, 20)

    def test_unset_create_settings_are_not_passed(self, monkeypatch, isolated_home):
        import __init__ as plugin_pkg
        captured = TestProviderSettings._capture(monkeypatch)
        TestProviderSettings._write_settings(isolated_home, {"max_cpus": 2})
        plugin_pkg.SpritesProvider().create_environment(
            cwd="/root", timeout=60, task_id="t", image="hermes-docker-image:1"
        )
        assert captured["max_cpus"] == 2
        assert not {"image", "max_ram_gb", "max_disk_gb"} & set(captured)

    def test_invalid_create_setting_refuses_to_create(self, monkeypatch, isolated_home):
        import __init__ as plugin_pkg
        captured = TestProviderSettings._capture(monkeypatch)
        TestProviderSettings._write_settings(isolated_home, {"image": "ubuntu:latest"})
        with pytest.raises(ValueError, match="latest"):
            plugin_pkg.SpritesProvider().create_environment(cwd="/root", timeout=60, task_id="t")
        assert captured == {}

    def test_doctor_reports_create_settings(self, isolated_home):
        import __init__ as plugin_pkg

        def _row():
            return [r for r in plugin_pkg.SpritesProvider().doctor_checks()
                    if r[1] == "Sprites create settings"][0]

        assert _row() == (True, "Sprites create settings", "default image and sizing")
        TestProviderSettings._write_settings(isolated_home, {"image": "ubuntu:24.04", "max_cpus": 2})
        row = _row()
        assert row[0] is True and "ubuntu:24.04" in row[2] and "max_cpus 2" in row[2]
        TestProviderSettings._write_settings(isolated_home, {"max_cpus": 0})
        row = _row()
        assert row[0] is False and "max_cpus" in row[2]

    def test_manifest_declares_create_settings(self):
        hub = pytest.importorskip("hermes_cli.plugins_settings")
        plugin_dir = Path(__file__).resolve().parents[1]
        fields = {f["key"]: f for f in hub.plugin_settings_fields("sprites", plugin_dir)}
        assert fields["image"]["type"] == "string"
        for key in ("max_cpus", "max_ram_gb", "max_disk_gb"):
            assert fields[key]["type"] == "number"


# ---------------------------------------------------------------------------
# SPRITES_TOKEN_FILE: token read from a file, re-read on change and after 401
# ---------------------------------------------------------------------------

def _secrets_dir(tmp_path):
    """A dedicated token directory (everything in it stays out of file sync)."""
    path = tmp_path / "secrets"
    path.mkdir(exist_ok=True)
    return path


def _write_token(path, token):
    """Replace ``path`` atomically (temp + rename), as Hermes Cloud does."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(token)
    tmp.chmod(0o600)
    tmp.replace(path)


def _ok_cmd(output=b"ok\n"):
    cmd = MagicMock()
    cmd.combined_output.return_value = output
    return cmd


def _auth_cmd():
    cmd = MagicMock()
    cmd.combined_output.side_effect = _AuthenticationError("Authentication failed")
    return cmd


class _APIError401(_SpriteError):
    """Like sprites.exceptions.APIError for a 401 (websocket / fs paths)."""

    status_code = 401


class TestTokenFile:
    @pytest.fixture()
    def token_file(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets" / "sprites-token"
        path.parent.mkdir()
        _write_token(path, "file-token-1\n")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        return path

    def test_token_read_from_file_stripped(self, token_file):
        from sprites_environment import resolve_sprites_token
        assert resolve_sprites_token() == "file-token-1"

    def test_file_wins_over_env_token(self, token_file, monkeypatch):
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        from sprites_environment import resolve_sprites_token
        assert resolve_sprites_token() == "file-token-1"

    @pytest.mark.parametrize("content", [None, "", "  \n\t"], ids=["missing", "empty", "blank"])
    def test_missing_or_empty_file_errors_without_env_fallback(self, tmp_path, monkeypatch, content):
        path = _secrets_dir(tmp_path) / "sprites-token"
        if content is not None:
            path.write_text(content)
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        monkeypatch.setenv("SPRITE_TOKEN", "legacy-env-token")
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        with pytest.raises(SpritesTokenError, match="SPRITES_TOKEN_FILE") as info:
            resolve_sprites_token()
        assert "legacy-env-token" not in str(info.value)

    def test_unreadable_path_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(tmp_path))  # a directory
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        with pytest.raises(SpritesTokenError, match="SPRITES_TOKEN_FILE"):
            resolve_sprites_token()

    def test_unset_file_uses_env_token(self, monkeypatch):
        monkeypatch.setenv("SPRITES_TOKEN", "env-token")
        from sprites_environment import resolve_sprites_token
        assert resolve_sprites_token() == "env-token"
        monkeypatch.delenv("SPRITES_TOKEN")
        monkeypatch.setenv("SPRITE_TOKEN", "alt-token")
        assert resolve_sprites_token() == "alt-token"

    def test_unset_file_env_token_reaches_client(self, make_env, sprites_sdk):
        """hermes_runtime sets SPRITES_TOKEN only: the old path is unchanged."""
        make_env(task_id="legacy")
        sprites_mod, _ = sprites_sdk
        assert sprites_mod.SpritesClient.call_args.kwargs["token"] == "test-token"

    def test_environment_uses_file_token(self, make_env, sprites_sdk, token_file):
        make_env(task_id="tf")
        sprites_mod, _ = sprites_sdk
        assert sprites_mod.SpritesClient.call_args.kwargs["token"] == "file-token-1"

    def test_replaced_file_used_by_next_environment(self, make_env, sprites_sdk, token_file):
        make_env(task_id="tf1")
        sprites_mod, _ = sprites_sdk
        assert sprites_mod.SpritesClient.call_args.kwargs["token"] == "file-token-1"
        _write_token(token_file, "file-token-2")
        make_env(task_id="tf2")
        sprites_mod, _ = sprites_sdk
        assert sprites_mod.SpritesClient.call_args.kwargs["token"] == "file-token-2"

    def test_bad_file_refuses_environment_even_with_env_token(self, sprites_sdk, hermes_runtime,
                                                             tmp_path, monkeypatch):
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(tmp_path / "absent"))
        sprites_mod, _ = sprites_sdk
        from sprites_environment import SpritesEnvironment
        with pytest.raises(ValueError, match="SPRITES_TOKEN_FILE"):
            SpritesEnvironment(task_id="x")
        sprites_mod.SpritesClient.assert_not_called()

    def _swap_in_new_client(self, env, new_sprite):
        """Clients made after this return ``new_sprite`` from get_sprite."""
        new_client = MagicMock(name="new_client")
        new_client.get_sprite.return_value = new_sprite
        env._SpritesClient = MagicMock(return_value=new_client)
        return env._SpritesClient, new_client

    def test_401_rereads_file_and_retries_once_with_new_token(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="r401")
        old_client, old_sprite = env._mock_client, env._mock_sprite
        old_sprite.command = MagicMock(return_value=_auth_cmd())
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.command = MagicMock(return_value=_ok_cmd(b"fresh\n"))
        factory, new_client = self._swap_in_new_client(env, new_sprite)
        _write_token(token_file, "file-token-2")

        handle = env._run_bash("echo fresh", timeout=10)
        handle.wait()
        assert handle.stdout.read() == "fresh\n"
        assert handle.returncode == 0
        factory.assert_called_once()
        assert factory.call_args.kwargs["token"] == "file-token-2"
        new_client.get_sprite.assert_called_once_with(env._sprite_name)
        assert old_sprite.command.call_count == 1
        assert new_sprite.command.call_count == 1
        assert env._sprite is new_sprite and env._client is new_client
        old_client.close.assert_called_once()

        # Later calls keep using the new client without another re-read.
        handle = env._run_bash("echo again", timeout=10)
        handle.wait()
        assert new_sprite.command.call_count == 2
        factory.assert_called_once()

    def test_401_with_unchanged_token_is_not_retried(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="same")
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        factory, _ = self._swap_in_new_client(env, MagicMock())

        with pytest.raises(_AuthenticationError):
            env._exec_or_raise("true", 5, "probe")
        assert env._mock_sprite.command.call_count == 1
        factory.assert_not_called()

    def test_retry_happens_only_once(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="once")
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.command = MagicMock(return_value=_auth_cmd())
        factory, _ = self._swap_in_new_client(env, new_sprite)
        _write_token(token_file, "file-token-2")

        with pytest.raises(_AuthenticationError):
            env._exec_or_raise("true", 5, "probe")
        assert env._mock_sprite.command.call_count == 1
        assert new_sprite.command.call_count == 1
        factory.assert_called_once()

    def test_401_with_status_code_on_upload_is_retried(self, make_env, sprites_sdk, token_file, tmp_path):
        env = make_env(task_id="fs401")
        host = tmp_path / "f.txt"
        host.write_bytes(b"data")
        old_remote = MagicMock()
        old_remote.write_bytes.side_effect = _APIError401("API error (status 401)")
        env._fs = MagicMock()
        env._fs.__truediv__.return_value = old_remote
        new_remote = MagicMock()
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.filesystem.return_value.__truediv__.return_value = new_remote
        factory, _ = self._swap_in_new_client(env, new_sprite)
        _write_token(token_file, "file-token-2")

        env._sprite_upload(str(host), "/home/sprite/.hermes/f.txt")
        new_remote.write_bytes.assert_called_once_with(b"data", mode=0o644, mkdir_parents=True)
        assert factory.call_args.kwargs["token"] == "file-token-2"

    def test_401_when_file_turns_bad_is_not_retried(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="gone")
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        factory, _ = self._swap_in_new_client(env, MagicMock())
        token_file.unlink()
        with pytest.raises(_AuthenticationError):
            env._exec_or_raise("true", 5, "probe")
        factory.assert_not_called()

    def test_token_never_logged(self, make_env, sprites_sdk, token_file, caplog):
        import logging
        caplog.set_level(logging.DEBUG)
        env = make_env(task_id="quiet")
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.command = MagicMock(return_value=_ok_cmd())
        self._swap_in_new_client(env, new_sprite)
        _write_token(token_file, "file-token-2")
        env._exec_or_raise("true", 5, "probe")
        assert "file-token" not in caplog.text

    def test_401_detection(self, sprites_sdk):
        from sprites_environment import _is_auth_error
        assert _is_auth_error(_AuthenticationError("x"))
        assert _is_auth_error(_APIError401("x"))
        try:
            try:
                raise _APIError401("handshake 401")
            except _APIError401 as inner:
                raise _NetworkError("WebSocket command failed") from inner
        except _NetworkError as wrapped:
            assert _is_auth_error(wrapped)
        try:
            try:
                raise _APIError401("old")
            except _APIError401:
                raise _NetworkError("unrelated") from None
        except _NetworkError as suppressed:
            assert not _is_auth_error(suppressed)
        assert not _is_auth_error(_NetworkError("boom"))

    def test_non_auth_errors_are_not_retried(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="other")
        cmd = MagicMock()
        cmd.combined_output.side_effect = _NetworkError("boom")
        env._mock_sprite.command = MagicMock(return_value=cmd)
        factory, _ = self._swap_in_new_client(env, MagicMock())
        _write_token(token_file, "file-token-2")
        with pytest.raises(_NetworkError):
            env._exec_or_raise("true", 5, "probe")
        factory.assert_not_called()


class TestProviderTokenFile:
    def test_file_only_counts_as_configured(self, sprites_sdk, tmp_path, monkeypatch):
        import __init__ as plugin_pkg
        path = _secrets_dir(tmp_path) / "sprites-token"
        _write_token(path, "file-token-1")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        provider = plugin_pkg.SpritesProvider()
        assert provider.is_available() is True
        assert provider.probe() == ("ready", "")
        assert provider.check_requirements({}) is True
        row = [r for r in provider.doctor_checks() if r[1] == "Sprites token"][0]
        assert row[0] is True

    def test_bad_file_is_not_configured_and_names_the_variable(self, sprites_sdk, tmp_path, monkeypatch):
        import __init__ as plugin_pkg
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(tmp_path / "absent"))
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        provider = plugin_pkg.SpritesProvider()
        assert provider.is_available() is False
        assert provider.check_requirements({}) is False
        status, message = provider.probe()
        assert status == "needs_setup" and "SPRITES_TOKEN_FILE" in message

    def test_env_token_still_configures_when_file_unset(self, sprites_sdk, monkeypatch):
        import __init__ as plugin_pkg
        monkeypatch.setenv("SPRITES_TOKEN", "env-token")
        assert plugin_pkg.SpritesProvider().probe() == ("ready", "")

    def test_token_file_variable_stripped_from_subprocesses(self):
        import __init__ as plugin_pkg
        keys = plugin_pkg.SpritesProvider().strip_env_keys
        assert {"SPRITES_TOKEN", "SPRITE_TOKEN", "SPRITES_TOKEN_FILE"} <= keys

    def test_nas_detection_marker_present(self):
        """Hermes Cloud detects support with grep -rqs --include='*.py' SPRITES_TOKEN_FILE."""
        root = Path(__file__).resolve().parents[1]
        sources = [p for p in root.glob("*.py") if "SPRITES_TOKEN_FILE" in p.read_text()]
        assert sources



# ---------------------------------------------------------------------------
# SPRITES_TOKEN_FILE hardening (review round 1)
# ---------------------------------------------------------------------------

try:  # Captured before any fixture stubs them out (hermes_runtime does).
    import tools.credential_files as _credential_files

    _REAL_SYNC_SOURCES = {
        name: getattr(_credential_files, name)
        for name in ("get_credential_file_mounts", "iter_skills_files", "iter_cache_files")
    }
except Exception:  # pragma: no cover - hermes-agent without credential_files
    _REAL_SYNC_SOURCES = {}


def _use_real_sync_sources(monkeypatch):
    if not _REAL_SYNC_SOURCES:
        pytest.skip("hermes-agent has no tools.credential_files")
    for name, fn in _REAL_SYNC_SOURCES.items():
        monkeypatch.setattr(f"tools.credential_files.{name}", fn)
    from tools.credential_files import clear_credential_files
    clear_credential_files()
    try:
        from agent.skill_utils import _external_dirs_cache_clear
        _external_dirs_cache_clear()
    except Exception:
        pass


def _refresh_factory(env, new_sprite):
    client = MagicMock(name="new_client")
    client.get_sprite.return_value = new_sprite
    env._SpritesClient = MagicMock(return_value=client)
    return env._SpritesClient, client


def _invalid_status_401():
    """A websockets InvalidStatus for HTTP 401, as the exec handshake raises it."""
    pytest.importorskip("websockets")
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response
    return InvalidStatus(Response(401, "Unauthorized", Headers(), b"Unauthorized"))


class TestTokenSourceScope:
    """Blocker 1: once SPRITES_TOKEN_FILE is set, SPRITES_TOKEN is never used."""

    def _scoped_env(self, make_env, token_path):
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        scope = set_secret_scope({"SPRITES_TOKEN_FILE": str(token_path)})
        try:
            return make_env(task_id="scoped")
        finally:
            reset_secret_scope(scope)

    def test_retry_thread_rereads_the_scoped_file(self, make_env, sprites_sdk, tmp_path, monkeypatch):
        token = _secrets_dir(tmp_path) / "scoped-token"
        _write_token(token, "scoped-token-1")
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._scoped_env(make_env, token)
        assert env._token == "scoped-token-1"
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        _write_token(token, "scoped-token-2")
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.command.return_value = _ok_cmd(b"ok\n")
        factory, _ = _refresh_factory(env, new_sprite)

        handle = env._run_bash("echo ok", timeout=5)  # runs on a worker thread
        assert handle.wait(timeout=5) == 0
        factory.assert_called_once()
        assert factory.call_args.kwargs["token"] == "scoped-token-2"
        assert env._token == "scoped-token-2"

    def test_retry_thread_with_scoped_file_gone_fails_closed(self, make_env, sprites_sdk, tmp_path,
                                                             monkeypatch):
        token = _secrets_dir(tmp_path) / "scoped-token"
        _write_token(token, "scoped-token-1")
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._scoped_env(make_env, token)
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        token.unlink()
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        factory, _ = _refresh_factory(env, MagicMock())

        handle = env._run_bash("echo ok", timeout=5)
        assert handle.wait(timeout=5) != 0
        factory.assert_not_called()
        assert env._token == "scoped-token-1"

    @pytest.mark.parametrize("value", ["", " \t "], ids=["empty", "blank"])
    def test_blank_file_variable_fails_closed(self, sprites_sdk, monkeypatch, value):
        import __init__ as plugin_pkg
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        monkeypatch.setenv("SPRITES_TOKEN_FILE", value)
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        with pytest.raises(SpritesTokenError, match="SPRITES_TOKEN_FILE") as info:
            resolve_sprites_token()
        assert "legacy-env-token" not in str(info.value)
        status, message = plugin_pkg.SpritesProvider().probe()
        assert status == "needs_setup" and "SPRITES_TOKEN_FILE" in message

    def test_env_token_setup_never_rereads_after_401(self, make_env, sprites_sdk, monkeypatch):
        """Without a token file nothing can change, so a 401 is not retried."""
        env = make_env(task_id="envonly")
        monkeypatch.setenv("SPRITES_TOKEN", "other-env-token")
        env._mock_sprite.command = MagicMock(return_value=_auth_cmd())
        factory, _ = _refresh_factory(env, MagicMock())
        with pytest.raises(_AuthenticationError):
            env._exec_or_raise("true", 5, "probe")
        factory.assert_not_called()


class TestTokenFileNotSynced:
    """The token file's directory never reaches a file-sync upload."""

    @pytest.fixture()
    def env_with_token(self, make_env, sprites_sdk, isolated_home, monkeypatch):
        def build(token_path, content="sync-secret-token"):
            token_path.parent.mkdir(parents=True, exist_ok=True)
            _write_token(token_path, content)
            monkeypatch.setenv("SPRITES_TOKEN_FILE", str(token_path))
            env = make_env(task_id="sync")
            _use_real_sync_sources(monkeypatch)
            return env
        return build

    @staticmethod
    def _hosts(env):
        return {Path(h).resolve() for h, _ in env._sync_manager._get_files_fn()}

    def test_registered_credential_file_not_uploaded(self, env_with_token, isolated_home):
        from tools.credential_files import register_credential_file
        secret = isolated_home / "secrets" / "sprites-token"
        env = env_with_token(secret)
        assert register_credential_file("secrets/sprites-token")
        assert secret.resolve() not in self._hosts(env)

    def test_symlink_to_token_in_skills_not_uploaded(self, env_with_token, isolated_home):
        secret = isolated_home / "secrets" / "sprites-token"
        env = env_with_token(secret)
        skill_dir = isolated_home / "skills" / "alias"
        skill_dir.mkdir(parents=True)
        (skill_dir / "via-symlink").symlink_to(secret)
        (skill_dir / "SKILL.md").write_text("alias skill")
        hosts = {Path(h) for h, _ in env._sync_manager._get_files_fn()}
        assert skill_dir / "via-symlink" not in hosts
        assert skill_dir / "SKILL.md" in hosts
        from sprites_environment import _without_token_dir
        pairs = [(str(skill_dir / "via-symlink"), "/r/via-symlink")]
        assert _without_token_dir(pairs, env._token_source.token_dir) == []


class TestTokenFileContents:
    """Blocker 3 and Important 2: what a token file may hold, and how it is read."""

    @pytest.fixture()
    def token_path(self, tmp_path, monkeypatch):
        path = _secrets_dir(tmp_path) / "sprites-token"
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        return path

    @pytest.mark.parametrize("content", [b"tok-1", b"tok-1\n", b"tok-1\r\n"])
    def test_token_with_optional_trailing_newline(self, token_path, content):
        from sprites_environment import resolve_sprites_token
        token_path.write_bytes(content)
        assert resolve_sprites_token() == "tok-1"

    @pytest.mark.parametrize("content", [
        b"secret-line-1\nsecret-line-2", b"secret-line-1\n\n", b" secret-line-1",
        b"secret-line-1 secret-line-2", b"secret-line-1\x00", b"secret-line-1\x1b",
        "secret-line-1\u00e9".encode(), b"secret-line-1\xff",
    ], ids=["multiline", "two-newlines", "leading-space", "inner-space", "nul", "escape",
            "non-ascii", "invalid-utf8"])
    def test_malformed_token_rejected_without_contents(self, token_path, content):
        import traceback
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        token_path.write_bytes(content)
        with pytest.raises(SpritesTokenError, match="SPRITES_TOKEN_FILE") as info:
            resolve_sprites_token()
        text = "".join(traceback.format_exception(info.value)) + repr(info.value)
        assert "secret-line" not in text and "legacy-env-token" not in text

    def test_malformed_token_never_reaches_the_client(self, sprites_sdk, hermes_runtime, token_path):
        token_path.write_bytes(b"secret-line-1\nsecret-line-2")
        sprites_mod, _ = sprites_sdk
        from sprites_environment import SpritesEnvironment, SpritesTokenError
        with pytest.raises(SpritesTokenError) as info:
            SpritesEnvironment(task_id="bad")
        sprites_mod.SpritesClient.assert_not_called()
        assert "secret-line" not in str(info.value)

    def test_env_token_with_control_characters_rejected(self, monkeypatch):
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        monkeypatch.setenv("SPRITES_TOKEN", "secret-line-1\nsecret-line-2")
        with pytest.raises(SpritesTokenError) as info:
            resolve_sprites_token()
        assert "secret-line" not in str(info.value)

    def test_sdk_error_quoting_the_token_is_withheld(self, make_env, sprites_sdk, token_path):
        import traceback
        _write_token(token_path, "file-token-quoted")
        env = make_env(task_id="quoted")
        cmd = MagicMock()
        cmd.combined_output.side_effect = _NetworkError(
            "Illegal header value b'Bearer file-token-quoted'")
        env._mock_sprite.command = MagicMock(return_value=cmd)
        from sprites_environment import SpritesApiError
        with pytest.raises(SpritesApiError) as info:
            env._exec_or_raise("true", 5, "probe")
        text = "".join(traceback.format_exception(info.value))
        assert "file-token-quoted" not in text

    def test_fifo_is_refused_without_blocking(self, token_path):
        import threading
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs on this platform")
        os.mkfifo(token_path)
        result = {}

        def read():
            try:
                resolve_sprites_token()
            except SpritesTokenError as e:
                result["error"] = str(e)

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "reading a FIFO blocked"
        assert "not a regular file" in result["error"]

    def test_provider_probe_does_not_block_on_fifo(self, sprites_sdk, token_path):
        import threading
        import __init__ as plugin_pkg
        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs on this platform")
        os.mkfifo(token_path)
        result = {}
        t = threading.Thread(target=lambda: result.setdefault(
            "probe", plugin_pkg.SpritesProvider().probe()), daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "provider probe blocked on a FIFO"
        assert result["probe"][0] == "needs_setup"

    def test_device_is_refused(self, monkeypatch):
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        if not os.path.exists("/dev/zero"):
            pytest.skip("no /dev/zero")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", "/dev/zero")
        with pytest.raises(SpritesTokenError, match="not a regular file"):
            resolve_sprites_token()

    def test_oversized_file_refused(self, token_path):
        from sprites_environment import (
            _TOKEN_FILE_MAX_BYTES, SpritesTokenError, resolve_sprites_token,
        )
        token_path.write_bytes(b"x" * _TOKEN_FILE_MAX_BYTES)
        assert len(resolve_sprites_token()) == _TOKEN_FILE_MAX_BYTES
        token_path.write_bytes(b"x" * (_TOKEN_FILE_MAX_BYTES + 1))
        with pytest.raises(SpritesTokenError, match="larger than"):
            resolve_sprites_token()


class TestSdkShaped401:
    """Important 1: the 401 shapes sprites-py 0.5 actually raises."""

    @pytest.fixture()
    def token_file(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets" / "sprites-token"
        path.parent.mkdir()
        _write_token(path, "file-token-1")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        return path

    def test_exec_handshake_invalid_status_is_auth_error(self, sprites_sdk):
        from sprites_environment import _is_auth_error
        inner = _invalid_status_401()
        try:
            try:
                raise inner
            except Exception as e:
                raise _NetworkError(f"WebSocket command failed: {type(e).__name__}: {e}") from e
        except _NetworkError as wrapped:
            assert _is_auth_error(wrapped)

    def test_exec_handshake_401_is_retried(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="ws401")
        inner = _invalid_status_401()

        def handshake_401():
            raise _NetworkError("WebSocket command failed: InvalidStatus") from inner

        cmd = MagicMock()
        cmd.combined_output.side_effect = handshake_401
        env._mock_sprite.command = MagicMock(return_value=cmd)
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.command.return_value = _ok_cmd()
        factory, _ = _refresh_factory(env, new_sprite)
        _write_token(token_file, "file-token-2")
        env._exec_or_raise("true", 5, "probe")
        assert factory.call_args.kwargs["token"] == "file-token-2"
        assert new_sprite.command.call_count == 1

    def test_filesystem_401_without_status_code_is_retried(self, make_env, sprites_sdk, token_file,
                                                           tmp_path):
        """sprites-py turns an fs/write 401 into a FilesystemError with no status."""
        import httpx

        class _FilesystemError(_SpriteError):  # no status_code, no response
            pass

        def client_for(token):
            client = MagicMock(name=f"client({token})")
            status = 401 if token == "file-token-1" else 200
            client._client = httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(status)))
            return client

        env = make_env(task_id="fsplain")
        env._SpritesClient = MagicMock(side_effect=lambda token, timeout: client_for(token))
        env._client = env._new_client("file-token-1")

        def write_401(*args, **kwargs):
            env._client._client.put("https://sprites.invalid/fs/write", content=b"x")
            raise _FilesystemError("write /home/sprite/f: Unauthorized")

        old_remote = MagicMock()
        old_remote.write_bytes.side_effect = write_401
        env._fs = MagicMock()
        env._fs.__truediv__.return_value = old_remote
        new_remote = MagicMock()
        new_sprite = MagicMock(name="new_sprite")
        new_sprite.filesystem.return_value.__truediv__.return_value = new_remote
        host = tmp_path / "f.txt"
        host.write_bytes(b"data")
        _write_token(token_file, "file-token-2")
        env._SpritesClient.side_effect = None
        new_client = client_for("file-token-2")
        new_client.get_sprite.return_value = new_sprite
        env._SpritesClient.return_value = new_client

        env._sprite_upload(str(host), "/home/sprite/f.txt")
        new_remote.write_bytes.assert_called_once_with(b"data", mode=0o644, mkdir_parents=True)
        assert env._SpritesClient.call_args.kwargs["token"] == "file-token-2"

    def test_filesystem_error_without_401_is_not_retried(self, make_env, sprites_sdk, token_file,
                                                         tmp_path):
        import httpx

        class _FilesystemError(_SpriteError):
            pass

        env = make_env(task_id="fs500")
        client = MagicMock()
        client._client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(500)))
        env._SpritesClient = MagicMock(return_value=client)
        env._client = env._new_client("file-token-1")
        env._SpritesClient.reset_mock()

        def write_500(*args, **kwargs):
            env._client._client.put("https://sprites.invalid/fs/write", content=b"x")
            raise _FilesystemError("write /home/sprite/f: boom")

        remote = MagicMock()
        remote.write_bytes.side_effect = write_500
        env._fs = MagicMock()
        env._fs.__truediv__.return_value = remote
        host = tmp_path / "f.txt"
        host.write_bytes(b"data")
        _write_token(token_file, "file-token-2")
        from sprites_environment import SpritesHttpError
        with pytest.raises(SpritesHttpError) as info:
            env._sprite_upload(str(host), "/home/sprite/f.txt")
        assert info.value.status_code == 500
        assert isinstance(info.value.__cause__, _FilesystemError)
        env._SpritesClient.assert_not_called()


class TestLifecycle401:
    """Minor: initial get/create, home detection and cleanup delete also retry."""

    @pytest.fixture()
    def token_file(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets" / "sprites-token"
        path.parent.mkdir()
        _write_token(path, "file-token-1")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        return path

    def _clients(self, sprites_sdk, first, second):
        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(side_effect=[first, second])
        return sprites_mod.SpritesClient

    def test_initial_get_401_after_renewal_is_retried(self, sprites_sdk, hermes_runtime, token_file):
        sprite = _make_sprite("hermes-life")
        first, second = MagicMock(name="first"), MagicMock(name="second")

        def renew_then_401(name):
            _write_token(token_file, "file-token-2")
            raise _AuthenticationError("Authentication failed")

        first.get_sprite.side_effect = renew_then_401
        second.get_sprite.return_value = sprite
        factory = self._clients(sprites_sdk, first, second)
        from sprites_environment import SpritesEnvironment
        env = SpritesEnvironment(task_id="life")
        assert [c.kwargs["token"] for c in factory.call_args_list] == ["file-token-1", "file-token-2"]
        assert env._client is second and env._sprite is sprite
        first.close.assert_called_once()

    def test_create_401_after_renewal_is_retried(self, sprites_sdk, hermes_runtime, token_file):
        sprite = _make_sprite("hermes-life")
        first, second = MagicMock(name="first"), MagicMock(name="second")
        first.get_sprite.side_effect = _NotFoundError("not found")

        def renew_then_401(name):
            _write_token(token_file, "file-token-2")
            raise _AuthenticationError("Authentication failed")

        first.create_sprite.side_effect = renew_then_401
        second.create_sprite.return_value = sprite
        self._clients(sprites_sdk, first, second)
        from sprites_environment import SpritesEnvironment
        env = SpritesEnvironment(task_id="life")
        second.create_sprite.assert_called_once_with("hermes-life")
        assert env._sprite is sprite

    def test_cleanup_delete_401_after_renewal_is_retried(self, make_env, sprites_sdk, token_file):
        env = make_env(task_id="gone", persistent_filesystem=False)
        env._mock_sprite.delete.side_effect = _AuthenticationError("Authentication failed")
        new_sprite = MagicMock(name="new_sprite")
        factory, _ = _refresh_factory(env, new_sprite)
        _write_token(token_file, "file-token-2")
        env.cleanup()
        new_sprite.delete.assert_called_once()
        assert factory.call_args.kwargs["token"] == "file-token-2"


class TestRound2:
    """Review round 2: path meaning, renewed tokens in sync, create replay, scrub."""

    @pytest.fixture()
    def token_file(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets" / "sprites-token"
        path.parent.mkdir()
        _write_token(path, "file-token-1")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        return path

    def test_symlink_dotdot_path_is_refused(self, tmp_path, monkeypatch):
        """Round 5: ``link/..`` would differ from its lexical meaning, so it is refused."""
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        local, foreign = tmp_path / "local", tmp_path / "foreign"
        (foreign / "child").mkdir(parents=True)
        local.mkdir()
        (local / "link").symlink_to(foreign / "child", target_is_directory=True)
        (foreign / "token").write_text("intended-token")
        (local / "token").write_text("other-token")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(local / "link" / ".." / "token"))
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        with pytest.raises(SpritesTokenError, match="no symlinks") as info:
            resolve_sprites_token()
        for secret in ("intended-token", "other-token", "legacy-env-token"):
            assert secret not in str(info.value)

    def test_configured_create_get_401_does_not_repeat_the_post(self, sprites_sdk, hermes_runtime,
                                                                token_file, monkeypatch):
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        sprite = _make_sprite("hermes-made")
        first, second = MagicMock(name="first"), MagicMock(name="second")
        first.http_client.post.return_value = MagicMock(status_code=200, is_success=True)

        def renew_then_401(name):
            _write_token(token_file, "file-token-2")
            raise _AuthenticationError("Authentication failed for get sprite")

        first.get_sprite.side_effect = renew_then_401
        second.get_sprite.return_value = sprite
        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(side_effect=[first, second])
        from sprites_environment import SpritesEnvironment
        env = SpritesEnvironment(task_id="made", persistent_filesystem=False,
                                 image="registry.example/img:1")
        assert first.http_client.post.call_count == 1
        second.http_client.post.assert_not_called()
        assert env._sprite is sprite

    def test_configured_create_post_401_is_retried(self, sprites_sdk, hermes_runtime, token_file,
                                                   monkeypatch):
        monkeypatch.setattr("sprites_environment._resolve_profile_identity", lambda: None)
        sprite = _make_sprite("hermes-made")
        first, second = MagicMock(name="first"), MagicMock(name="second")

        def renew_then_401(*args, **kwargs):
            _write_token(token_file, "file-token-2")
            return MagicMock(status_code=401, is_success=False)

        first.http_client.post.side_effect = renew_then_401
        second.http_client.post.return_value = MagicMock(status_code=200, is_success=True)
        second.get_sprite.return_value = sprite
        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(side_effect=[first, second])
        from sprites_environment import SpritesEnvironment
        env = SpritesEnvironment(task_id="made", persistent_filesystem=False,
                                 image="registry.example/img:1")
        assert first.http_client.post.call_count == 1
        assert second.http_client.post.call_count == 1
        first.get_sprite.assert_not_called()
        assert env._sprite is sprite

    def test_archive_clear_error_quoting_token_is_withheld(self, make_env, sprites_sdk, token_file,
                                                           tmp_path, caplog):
        import logging
        from sprites_environment import _ARCHIVE_LEFT_EXIT, SpritesApiError, _ExecFailed
        from tools.environments.file_sync import FileSyncManager
        env = make_env(task_id="clear")

        def write(data, **kwargs):
            if data == b"":
                raise _SpriteError("write: credential rejected: file-token-1")

        remote = MagicMock()
        remote.write_bytes.side_effect = write
        env._fs = MagicMock()
        env._fs.__truediv__.return_value = remote

        def unpack_left_archive(*args, **kwargs):
            raise _ExecFailed("archive could not be removed", _ARCHIVE_LEFT_EXIT)

        env._exec_or_raise = unpack_left_archive
        host = tmp_path / "ordinary"
        host.write_text("ordinary")
        with pytest.raises(SpritesApiError):
            env._sprite_bulk_upload([(str(host), "/home/sprite/.hermes/ordinary")])
        caplog.set_level(logging.DEBUG)
        env._use_archives = True
        FileSyncManager(get_files_fn=lambda: [(str(host), "/home/sprite/.hermes/ordinary")],
                        upload_fn=env._sprite_upload, delete_fn=lambda *_: None,
                        bulk_upload_fn=env._sprite_bulk_upload).sync(force=True)
        assert "file-token-1" not in caplog.text

    def _failing_exec(self, env, error):
        cmd = MagicMock()
        cmd.combined_output.side_effect = error
        env._mock_sprite.command = MagicMock(return_value=cmd)

    def test_withheld_error_keeps_no_link_to_the_original(self, make_env, sprites_sdk, token_file):
        from sprites_environment import SpritesApiError
        env = make_env(task_id="ctx")
        self._failing_exec(env, _NetworkError("header Bearer file-token-1"))
        with pytest.raises(SpritesApiError) as info:
            env._exec_or_raise("true", 5, "probe")
        assert info.value.__context__ is None and info.value.__cause__ is None

    def test_token_in_exception_notes_is_withheld(self, make_env, sprites_sdk, token_file):
        from sprites_environment import SpritesApiError
        env = make_env(task_id="notes")
        error = _NetworkError("boom")
        error.add_note("sent Bearer file-token-1")
        self._failing_exec(env, error)
        with pytest.raises(SpritesApiError):
            env._exec_or_raise("true", 5, "probe")

    def test_token_in_command_output_is_withheld(self, make_env, sprites_sdk, token_file):
        from sprites_environment import SpritesApiError
        env = make_env(task_id="out")
        self._failing_exec(env, _ExitError("exit 1", 1, stdout=b"file-token-1\n"))
        with pytest.raises(SpritesApiError):
            env._exec_or_raise("true", 5, "probe")

    def test_ordinary_command_failure_unchanged(self, make_env, sprites_sdk, token_file):
        from sprites_environment import _ExecFailed
        env = make_env(task_id="plain")
        self._failing_exec(env, _ExitError("exit 3", 3, stdout=b"no such file\n"))
        with pytest.raises(_ExecFailed, match="no such file") as info:
            env._exec_or_raise("true", 5, "probe")
        assert info.value.exit_code == 3


class TestTokenDirRule:
    """Review round 5: two rules, nothing else.

    1. ``SPRITES_TOKEN_FILE`` is absolute and crosses no symlink (``lstat``
       on every component), so the configured path is the resolved path.
    2. File sync never uploads a file whose realpath is inside the token
       file's directory, and warns once per skipped path.
    """

    @pytest.fixture()
    def layout(self, tmp_path, monkeypatch):
        """The Hermes Cloud layout: HERMES_HOME=/opt/data, token in /opt/data/secrets."""
        home = tmp_path.resolve() / "opt-data"
        (home / "secrets").mkdir(parents=True)
        (home / "secrets").chmod(0o700)
        skill = home / "skills" / "weather"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("weather skill")
        (skill / "forecast.py").write_text("print('sunny')")
        monkeypatch.setenv("HERMES_HOME", str(home))
        _write_token(home / "secrets" / "sprites-token", "cloud-token-1")
        return home

    @pytest.fixture()
    def build(self, make_env, sprites_sdk, monkeypatch):
        def _build(token_path, task_id="static"):
            monkeypatch.setenv("SPRITES_TOKEN_FILE", str(token_path))
            env = make_env(task_id=task_id)  # its creation sync sees no files
            _use_real_sync_sources(monkeypatch)
            return env
        return _build

    @pytest.fixture(autouse=True)
    def _fresh_skip_warnings(self, monkeypatch):
        import sprites_environment
        monkeypatch.setattr(sprites_environment, "_skip_warned", set())

    @staticmethod
    def _sync(env):
        """Run the env's real FileSyncManager through its real archive upload.

        Returns ``(shipped {remote: bytes}, deleted [remote])``.
        """
        import io
        import tarfile
        shipped, deleted = {}, []

        def write_bytes(data, **kwargs):
            if not data:
                return
            try:
                with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                    for m in tar.getmembers():
                        shipped["/" + m.name] = tar.extractfile(m).read()
            except tarfile.TarError:
                shipped["<single>"] = data

        remote = MagicMock()
        remote.write_bytes.side_effect = write_bytes
        env._fs = MagicMock()
        env._fs.__truediv__.return_value = remote
        env._exec_or_raise = lambda *a, **k: None
        env._sync_manager._delete_fn = deleted.extend
        env._sync_manager.sync(force=True)
        return shipped, deleted

    @staticmethod
    def _skip_warnings(caplog):
        return [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "file sync skipped" in r.getMessage()]

    def test_cloud_layout_is_allowed_and_skills_sync(self, layout, build):
        env = build(layout / "secrets" / "sprites-token")
        assert env._token == "cloud-token-1"
        assert env._token_source.token_dir == str(layout / "secrets")
        shipped, _ = self._sync(env)
        assert shipped["/home/sprite/.hermes/skills/weather/SKILL.md"] == b"weather skill"
        assert shipped["/home/sprite/.hermes/skills/weather/forecast.py"] == b"print('sunny')"
        assert not any(b"cloud-token" in d for d in shipped.values())

    def test_renewal_temp_and_rename_never_synced(self, layout, build, monkeypatch, caplog):
        """The NAS writer (0600 temp in the token dir, renamed over the token).

        Every route that could name a token-dir file is opened: registered
        credential files and an external skills root over the whole home.
        """
        from tools.credential_files import register_credential_file
        secrets = layout / "secrets"
        token = secrets / "sprites-token"
        env = build(token)
        (layout / "config.yaml").write_text(
            "skills:\n  external_dirs:\n    - " + str(layout) + "\n")
        _use_real_sync_sources(monkeypatch)
        tmp = secrets / ".sprites-token.Ab12Cd"
        tmp.write_text("cloud-token-2")             # writer's temp, before rename
        tmp.chmod(0o600)
        for name in ("secrets/sprites-token", "secrets/.sprites-token.Ab12Cd"):
            assert register_credential_file(name)
        caplog.set_level(logging.WARNING, logger="sprites_environment")
        before, _ = self._sync(env)
        tmp.replace(token)                          # the rename
        _write_token(token, "cloud-token-3")        # and a second renewal
        after, _ = self._sync(env)
        for shipped in (before, after):
            assert not any(b"cloud-token" in d for d in shipped.values())
            assert not any("sprites-token" in k for k in shipped)
        assert any(k.endswith("weather/SKILL.md") for k in before)
        warnings = self._skip_warnings(caplog)
        assert warnings and all(str(secrets) in w for w in warnings)
        assert not any("cloud-token" in w for w in warnings)
        assert env._token_source.read() == "cloud-token-3"

    @pytest.mark.parametrize("shape", ["file", "parent", "home"])
    def test_symlinked_component_is_refused(self, layout, tmp_path, monkeypatch, sprites_sdk,
                                            hermes_runtime, shape):
        """A symlinked file, parent directory or HERMES_HOME-style ancestor."""
        import __init__ as plugin_pkg
        from sprites_environment import SpritesEnvironment, SpritesTokenError, resolve_sprites_token
        base = tmp_path.resolve()
        if shape == "file":
            configured = base / "token-link"
            configured.symlink_to(layout / "secrets" / "sprites-token")
        elif shape == "parent":
            configured = base / "secrets-link" / "sprites-token"
            (base / "secrets-link").symlink_to(layout / "secrets", target_is_directory=True)
        else:
            (base / "home-link").symlink_to(layout, target_is_directory=True)
            configured = base / "home-link" / "secrets" / "sprites-token"
        assert configured.read_text() == "cloud-token-1"
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(configured))
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        with pytest.raises(SpritesTokenError, match="has a symlink at") as info:
            resolve_sprites_token()
        assert "cloud-token" not in str(info.value)
        assert "legacy-env-token" not in str(info.value)
        sprites_mod, _ = sprites_sdk
        with pytest.raises(SpritesTokenError):
            SpritesEnvironment(task_id="refused")
        sprites_mod.SpritesClient.assert_not_called()
        provider = plugin_pkg.SpritesProvider()
        assert provider.is_available() is False
        status, message = provider.probe()
        assert status == "needs_setup" and "no symlinks" in message

    @pytest.mark.parametrize("value", ["secrets/sprites-token", "~/secrets/sprites-token",
                                       "{home}/secrets/../secrets/sprites-token",
                                       "{home}/secrets//sprites-token",
                                       "{home}/./secrets/sprites-token"],
                             ids=["relative", "tilde", "dotdot", "double-slash", "dot"])
    def test_non_canonical_path_is_refused(self, layout, monkeypatch, value):
        from sprites_environment import SpritesTokenError, resolve_sprites_token
        monkeypatch.chdir(layout)
        monkeypatch.setenv("SPRITES_TOKEN_FILE", value.format(home=layout))
        monkeypatch.setenv("SPRITES_TOKEN", "legacy-env-token")
        with pytest.raises(SpritesTokenError, match="no symlinks") as info:
            resolve_sprites_token()
        problem = ("is not an absolute path" if not value.startswith("{home}")
                   else "contains '.', '..'")
        assert problem in str(info.value)
        assert "cloud-token" not in str(info.value)
        assert "legacy-env-token" not in str(info.value)

    def test_component_turned_symlink_after_creation_fails_closed(self, layout, build, tmp_path):
        from sprites_environment import SpritesTokenError
        env = build(layout / "secrets" / "sprites-token")
        other = tmp_path.resolve() / "other"
        other.mkdir()
        _write_token(other / "sprites-token", "elsewhere-token")
        shutil.rmtree(layout / "secrets")
        (layout / "secrets").symlink_to(other, target_is_directory=True)
        with pytest.raises(SpritesTokenError, match="has a symlink at") as info:
            env._token_source.read()
        assert "elsewhere-token" not in str(info.value)

    def test_symlinked_hermes_home_keeps_its_skills(self, layout, build, tmp_path, monkeypatch):
        """Round 4 B2 layout: HERMES_HOME is a symlink, the token sits beside it."""
        user = tmp_path.resolve() / "user"
        user.mkdir()
        (user / ".hermes").symlink_to(layout, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(user / ".hermes"))
        _write_token(user / ".sprites-token", "user-token-1")
        env = build(user / ".sprites-token")
        assert env._token_source.token_dir == str(user)
        shipped, _ = self._sync(env)
        assert shipped["/home/sprite/.hermes/skills/weather/SKILL.md"] == b"weather skill"
        assert not any(b"user-token" in d for d in shipped.values())

    def test_files_outside_token_dir_are_unaffected(self, layout, build, monkeypatch, caplog):
        from tools.credential_files import register_credential_file
        (layout / "credentials").mkdir()
        (layout / "credentials" / "weather-api.json").write_text('{"k": "v"}')
        (layout / "secrets-archive").mkdir()             # shares the name prefix only
        (layout / "secrets-archive" / "notes.txt").write_text("notes")
        env = build(layout / "secrets" / "sprites-token")
        assert register_credential_file("credentials/weather-api.json")
        assert register_credential_file("secrets-archive/notes.txt")
        caplog.set_level(logging.WARNING, logger="sprites_environment")
        shipped, _ = self._sync(env)
        assert any(k.endswith("credentials/weather-api.json") for k in shipped)
        assert any(k.endswith("secrets-archive/notes.txt") for k in shipped)
        assert any(k.endswith("weather/SKILL.md") for k in shipped)
        assert self._skip_warnings(caplog) == []

    def test_file_in_token_dir_is_skipped_with_one_warning(self, layout, build, monkeypatch,
                                                         caplog):
        """Round 4 B3: a file kept in the token dir is skipped, never silently."""
        from tools.credential_files import register_credential_file
        env = build(layout / "secrets" / "sprites-token")
        (layout / "secrets" / "weather-api.json").write_text('{"secret": "svc-cred"}')
        assert register_credential_file("secrets/weather-api.json")
        caplog.set_level(logging.WARNING, logger="sprites_environment")
        shipped, _ = self._sync(env)
        assert not any(k.endswith("weather-api.json") for k in shipped)
        warnings = self._skip_warnings(caplog)
        assert len(warnings) == 1
        assert str(layout / "secrets" / "weather-api.json") in warnings[0]
        assert "svc-cred" not in warnings[0]
        caplog.clear()
        (layout / "skills" / "weather" / "forecast.py").write_text("print('rain')")
        self._sync(env)
        assert self._skip_warnings(caplog) == []        # once per path

    def test_symlink_into_token_dir_from_skills_is_skipped(self, layout, build):
        alias = layout / "skills" / "weather" / "token-alias"
        alias.symlink_to(layout / "secrets" / "sprites-token")
        env = build(layout / "secrets" / "sprites-token")
        shipped, _ = self._sync(env)
        assert not any(k.endswith("token-alias") for k in shipped)
        assert not any(b"cloud-token" in d for d in shipped.values())
        assert any(k.endswith("weather/SKILL.md") for k in shipped)

    # --- review round 3 cases: no read changes sync -------------------------

    def test_rejected_setting_then_corrected_keeps_skill_synced(self, layout, build, monkeypatch):
        import __init__ as plugin_pkg
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(layout / "skills" / "weather" / "SKILL.md"))
        status, _ = plugin_pkg.SpritesProvider().probe()
        assert status == "needs_setup"
        env = build(layout / "secrets" / "sprites-token")
        shipped, _ = self._sync(env)
        assert any(k.endswith("weather/SKILL.md") for k in shipped)
        assert any(k.endswith("weather/forecast.py") for k in shipped)

    def test_valid_token_read_from_any_file_leaves_no_exclusion(self, layout, build, tmp_path,
                                                               monkeypatch):
        from sprites_environment import resolve_sprites_token
        stray = tmp_path.resolve() / "stray" / "tok"
        stray.parent.mkdir()
        stray.write_text("stray-token")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(stray))
        assert resolve_sprites_token() == "stray-token"
        from sprites_environment import SpritesTokenError, _read_token_file
        (layout / "skills" / "weather" / "tok").write_text("skill-dir-token")
        assert _read_token_file(str(layout / "skills" / "weather" / "tok")) == "skill-dir-token"
        with pytest.raises(SpritesTokenError):
            _read_token_file(str(layout / "skills" / "weather" / "SKILL.md"))
        env = build(layout / "secrets" / "sprites-token")
        assert env._token_source.token_dir == str(layout / "secrets")
        shipped, _ = self._sync(env)
        assert any(k.endswith("weather/SKILL.md") for k in shipped)
        assert any(k.endswith("weather/forecast.py") for k in shipped)

    def test_other_scope_readiness_check_does_not_change_sync(self, layout, build):
        import __init__ as plugin_pkg
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        env = build(layout / "secrets" / "sprites-token")
        first, _ = self._sync(env)
        assert any(k.endswith("weather/SKILL.md") for k in first)
        scope = set_secret_scope(
            {"SPRITES_TOKEN_FILE": str(layout / "skills" / "weather" / "SKILL.md")})
        try:
            assert plugin_pkg.SpritesProvider().probe()[0] == "needs_setup"
        finally:
            reset_secret_scope(scope)
        (layout / "skills" / "weather" / "forecast.py").write_text("print('rain')")
        shipped, deleted = self._sync(env)
        assert deleted == []
        assert shipped == {"/home/sprite/.hermes/skills/weather/forecast.py": b"print('rain')"}

    def test_module_keeps_no_guard_machinery(self):
        import sprites_environment as mod
        gone = ("_check_token_dir", "_upload_roots", "_TokenFileExclusion", "_record_token_file")
        assert not [n for n in gone if hasattr(mod, n)]

    def test_env_token_mode_excludes_nothing(self, layout, build, monkeypatch, make_env):
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = make_env(task_id="envtok")
        _use_real_sync_sources(monkeypatch)
        assert env._token_source.token_dir is None
        shipped, _ = self._sync(env)
        assert any(k.endswith("weather/SKILL.md") for k in shipped)


@pytest.fixture()
def real_sprites_sdk(monkeypatch):
    """The installed sprites-py (skips when absent); drops any stub first."""
    for name in [n for n in sys.modules if n == "sprites" or n.startswith("sprites.")]:
        monkeypatch.delitem(sys.modules, name)
    before = set(sys.modules)
    sdk = pytest.importorskip("sprites")
    yield sdk
    for name in set(sys.modules) - before:
        if name == "sprites" or name.startswith("sprites."):
            sys.modules.pop(name, None)


class TestRealSdk401:
    """Important 1 against the real sprites-py objects (needs sprites-py installed)."""

    def _env(self, client, token_path, monkeypatch):
        import threading
        from sprites_environment import SpritesEnvironment, _token_source
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(token_path))
        env = object.__new__(SpritesEnvironment)
        env._SpritesClient = MagicMock()
        env._client_timeout = 30
        env._token_source = _token_source()
        env._client = client
        env._token = client.token
        env._sprite_name = "offline-probe"
        env._sprite = client.sprite("offline-probe")
        env._fs = env._sprite.filesystem("/")
        env._auth_lock = threading.Lock()
        env._exec_env = None
        return env

    def test_real_filesystem_401_refreshes_client(self, real_sprites_sdk, tmp_path, monkeypatch):
        import httpx
        token = _secrets_dir(tmp_path) / "sprites-token"
        token.write_text("file-token-new")
        old = real_sprites_sdk.SpritesClient(token="file-token-old")
        old._client.close()
        old._client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": "Unauthorized"})))
        env = self._env(old, token, monkeypatch)
        hooks = old._client.event_hooks
        from sprites_environment import _note_http_status
        hooks["response"].append(_note_http_status)  # what _new_client adds
        new_sprite = MagicMock(name="new_sprite")
        new_client = MagicMock(name="new_client")
        new_client.get_sprite.return_value = new_sprite
        env._SpritesClient.return_value = new_client
        data = tmp_path / "upload"
        data.write_bytes(b"payload")
        env._sprite_upload(str(data), "/home/sprite/file")
        assert env._SpritesClient.call_args.kwargs["token"] == "file-token-new"
        new_sprite.filesystem.return_value.__truediv__.return_value.write_bytes.assert_called_once()

    def test_real_configured_create_never_repeats_a_successful_post(self, real_sprites_sdk, tmp_path,
                                                                     monkeypatch):
        import httpx
        from sprites.exceptions import NotFoundError, SpriteError
        token = _secrets_dir(tmp_path) / "sprites-token"
        token.write_text("file-token-old")
        requests = []

        def handler(request):
            requests.append((request.method, request.headers.get("authorization")))
            if request.method == "POST":
                return httpx.Response(200, json={"name": "offline-probe"})
            if request.headers.get("authorization") == "Bearer file-token-old":
                token.write_text("file-token-new")
                return httpx.Response(401, json={"error": "Unauthorized"})
            return httpx.Response(200, json={"name": "offline-probe"})

        def make_client(token, timeout):
            client = real_sprites_sdk.SpritesClient(token=token, timeout=timeout)
            client._client.close()
            client._client = httpx.Client(transport=httpx.MockTransport(handler))
            return client

        env = self._env(make_client("file-token-old", 30), token, monkeypatch)
        env._SpritesClient = make_client
        env._client = env._new_client("file-token-old")
        env._token = "file-token-old"
        env._sprite = None
        env._create_config = {"image_url": "registry.example/img:1"}
        env._persistent = False
        env._NotFoundError, env._SpriteError = NotFoundError, SpriteError
        try:
            env._create_sprite("offline-probe")
        finally:
            env._client.close()
        assert requests == [
            ("POST", "Bearer file-token-old"),
            ("GET", "Bearer file-token-old"),
            ("GET", "Bearer file-token-new"),
        ]

    def test_new_client_hooks_real_httpx_client(self, real_sprites_sdk, tmp_path, monkeypatch):
        from sprites_environment import _note_http_status
        token = _secrets_dir(tmp_path) / "sprites-token"
        token.write_text("file-token-1")
        client = real_sprites_sdk.SpritesClient(token="file-token-1")
        env = self._env(client, token, monkeypatch)
        env._SpritesClient = real_sprites_sdk.SpritesClient
        made = env._new_client("file-token-1")
        try:
            assert _note_http_status in made._client.event_hooks["response"]
        finally:
            made.close()
            client.close()

    def test_real_exec_handshake_401_is_auth_error(self, real_sprites_sdk, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock
        from sprites.exceptions import NetworkError
        from sprites_environment import _is_auth_error
        monkeypatch.setattr("sprites.websocket.websockets.connect",
                            AsyncMock(side_effect=_invalid_status_401()))
        token = _secrets_dir(tmp_path) / "sprites-token"
        token.write_text("file-token-1")
        client = real_sprites_sdk.SpritesClient(token="file-token-1")
        try:
            with pytest.raises(NetworkError) as info:
                client.sprite("offline-probe").command("true", timeout=5).combined_output()
            assert _is_auth_error(info.value)
        finally:
            client.close()


class TestFsWriteErrorDetail:
    """An fs/write HTTP error names the status, code and request id.

    Driven through the real sprites-py ``write_bytes``: 0.5.1 cannot parse
    a Sprites 2 error body and raises ``AttributeError: 'dict' object has
    no attribute 'lower'`` instead.
    """

    TOKEN = "env-token-secret"

    def _env(self, real_sprites_sdk, handler, hooked=True):
        import threading
        import httpx
        from sprites_environment import SpritesEnvironment

        def make_client(token, timeout):
            client = real_sprites_sdk.SpritesClient(token=token, timeout=timeout)
            client._client.close()
            client._client = httpx.Client(transport=httpx.MockTransport(handler))
            return client

        env = object.__new__(SpritesEnvironment)
        env._SpritesClient = make_client
        env._client_timeout = 30
        env._token_source = MagicMock(path=None)  # an env token: no 401 refresh
        env._token = self.TOKEN
        env._client = env._new_client(self.TOKEN) if hooked else make_client(self.TOKEN, 30)
        env._sprite_name = "offline-probe"
        env._sprite = env._client.sprite("offline-probe")
        env._fs = env._sprite.filesystem("/")
        env._auth_lock = threading.Lock()
        env._exec_env = None
        env._remote_home = "/home/sprite"
        env._use_archives = True
        return env

    def _upload(self, env, tmp_path, content=b"file-body-bytes"):
        host = tmp_path / "f.txt"
        host.write_bytes(content)
        try:
            env._sprite_upload(str(host), "/home/sprite/f.txt")
        finally:
            env._client.close()

    @staticmethod
    def _envelope(status, code, message, request_id, details=None):
        import httpx
        error = {"code": code, "message": message, "request_id": request_id}
        if details is not None:
            error["details"] = details
        return lambda request: httpx.Response(status, json={"error": error})

    def test_413_names_status_code_and_request_id(self, real_sprites_sdk, tmp_path, monkeypatch):
        from sprites_environment import SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, self._envelope(
            413, "file_too_large", "File is too large.", "GN4cREQ413",
            {"max_bytes": 16777216}))
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        text = str(info.value)
        assert "413" in text and "file_too_large" in text and "GN4cREQ413" in text
        assert "File is too large." in text
        assert info.value.status_code == 413
        assert info.value.code == "file_too_large"
        assert info.value.request_id == "GN4cREQ413"
        assert info.value.retry_after_ms is None
        # The SDK's own (broken-parser) error is kept as the cause.
        assert isinstance(info.value.__cause__, AttributeError)
        assert "file-body-bytes" not in text

    def test_429_sets_retry_after_ms(self, real_sprites_sdk, tmp_path, monkeypatch):
        from sprites_environment import SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, self._envelope(
            429, "rate_limited", "Rate limited.", "GN4cREQ429",
            {"limit": 300, "operation": "exec_open", "retry_ms": 60000}))
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert info.value.status_code == 429
        assert info.value.code == "rate_limited"
        assert info.value.retry_after_ms == 60000
        assert "60000" in str(info.value)

    def test_502_with_empty_body(self, real_sprites_sdk, tmp_path, monkeypatch):
        import httpx
        from sprites_environment import SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, lambda request: httpx.Response(502))
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert info.value.status_code == 502
        assert info.value.code is None
        assert "HTTP 502" in str(info.value)

    def test_plain_string_error_shape(self, real_sprites_sdk, tmp_path, monkeypatch):
        import httpx
        from sprites_environment import SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk,
                        lambda request: httpx.Response(400, json={"error": "bad path"}))
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert info.value.status_code == 400
        assert "HTTP 400" in str(info.value) and "bad path" in str(info.value)

    def test_bulk_upload_archive_write_reports_status(self, real_sprites_sdk, tmp_path,
                                                      monkeypatch):
        from sprites_environment import SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, self._envelope(
            413, "file_too_large", "File is too large.", "GN4cBULK"))
        host = tmp_path / "a.txt"
        host.write_bytes(b"archive member")
        try:
            with pytest.raises(SpritesHttpError) as info:
                env._sprite_bulk_upload([(str(host), "/home/sprite/a.txt")])
        finally:
            env._client.close()
        assert info.value.status_code == 413 and "GN4cBULK" in str(info.value)

    def test_token_echoed_by_server_is_never_in_message(self, real_sprites_sdk, tmp_path,
                                                        monkeypatch):
        from sprites_environment import SpritesHttpError, _mentions
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, self._envelope(
            403, "forbidden", f"token {self.TOKEN} may not write", "GN4cTOK"))
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert not _mentions(info.value, (self.TOKEN,))
        assert info.value.status_code == 403 and "GN4cTOK" in str(info.value)
        assert self.TOKEN not in (info.value.error_message or "")

    def test_error_without_http_error_is_unchanged(self, real_sprites_sdk, tmp_path, monkeypatch):
        """A genuine AttributeError (no HTTP error seen) surfaces as is."""
        import httpx
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, lambda request: httpx.Response(200, json={}))

        class _Broken:
            def write_bytes(self, *args, **kwargs):
                env._client._client.put("https://sprites.invalid/fs/write", content=b"x")
                raise AttributeError("a real bug")

        env._fs = MagicMock()
        env._fs.__truediv__.return_value = _Broken()
        with pytest.raises(AttributeError, match="a real bug"):
            self._upload(env, tmp_path)

    def test_stale_error_from_an_earlier_call_is_not_reused(self, real_sprites_sdk, tmp_path,
                                                            monkeypatch):
        import httpx
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, lambda request: httpx.Response(500))
        env._client._client.get("https://sprites.invalid/other")  # earlier HTTP error

        class _Broken:
            def write_bytes(self, *args, **kwargs):
                raise AttributeError("a real bug")

        env._fs = MagicMock()
        env._fs.__truediv__.return_value = _Broken()
        with pytest.raises(AttributeError, match="a real bug"):
            self._upload(env, tmp_path)

    def test_client_without_event_hooks_raises_sdk_error(self, real_sprites_sdk, tmp_path,
                                                         monkeypatch):
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, self._envelope(
            413, "file_too_large", "File is too large.", "GN4cNOHOOK"), hooked=False)
        with pytest.raises(AttributeError):  # the SDK's own error, unchanged
            self._upload(env, tmp_path)

    def test_client_object_without_event_hooks_still_builds(self, tmp_path, monkeypatch):
        from sprites_environment import SpritesEnvironment
        env = object.__new__(SpritesEnvironment)
        env._SpritesClient = lambda token, timeout: object()
        env._client_timeout = 30
        assert env._new_client("t") is not None

    def test_sync_manager_warning_names_status(self, real_sprites_sdk, tmp_path, monkeypatch,
                                               caplog):
        """The line FileSyncManager logs on a failed sync names the status."""
        import logging
        from tools.environments.file_sync import FileSyncManager
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, self._envelope(
            413, "file_too_large", "File is too large.", "GN4cSYNC"))
        host = tmp_path / "a.txt"
        host.write_bytes(b"x")
        manager = FileSyncManager(
            get_files_fn=lambda: [(str(host), "/home/sprite/a.txt")],
            upload_fn=env._sprite_upload,
            delete_fn=lambda paths: None,
            bulk_upload_fn=env._sprite_bulk_upload,
        )
        try:
            with caplog.at_level(logging.WARNING):
                manager.sync(force=True)
        finally:
            env._client.close()
        assert any("413" in r.getMessage() and "GN4cSYNC" in r.getMessage()
                   for r in caplog.records)


class TestFsWriteErrorDetailRefreshAndBodies:
    """fs/write error reporting around the 401 refresh and with streamed bodies.

    Real sprites-py 0.5.1 transport over ``httpx.MockTransport``; the tokens
    are made-up strings.
    """

    OLD = "file-token-old-secret"
    NEW = "file-token-new-secret"

    def _env(self, real_sprites_sdk, handler, tmp_path, monkeypatch):
        import threading
        import httpx
        from sprites_environment import SpritesEnvironment, _token_source

        token_file = _secrets_dir(tmp_path) / "sprites-token"
        token_file.write_text(self.NEW)  # the candidate the 401 refresh reads
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(token_file))

        def make_client(token, timeout):
            client = real_sprites_sdk.SpritesClient(token=token, timeout=timeout)
            client._client.close()
            client._client = httpx.Client(transport=httpx.MockTransport(handler))
            return client

        env = object.__new__(SpritesEnvironment)
        env._SpritesClient = make_client
        env._client_timeout = 30
        env._token_source = _token_source()
        env._token = self.OLD
        env._client = env._new_client(self.OLD)
        env._sprite_name = "offline-probe"
        env._sprite = env._client.sprite("offline-probe")
        env._fs = env._sprite.filesystem("/")
        env._auth_lock = threading.Lock()
        env._exec_env = None
        env._remote_home = "/home/sprite"
        env._use_archives = True
        return env

    @staticmethod
    def _error(status, code, message, request_id):
        import httpx
        return httpx.Response(status, json={"error": {
            "code": code, "message": message, "request_id": request_id}})

    def _upload(self, env, tmp_path, remote="/home/sprite/f.txt"):
        host = tmp_path / "f.txt"
        host.write_bytes(b"file-body-bytes")
        try:
            env._sprite_upload(str(host), remote)
        finally:
            env._client.close()

    def test_failed_refresh_reports_the_uploads_own_401(self, real_sprites_sdk, tmp_path,
                                                        monkeypatch):
        from sprites_environment import SpritesHttpError

        def handler(request):
            if request.method == "PUT":
                return self._error(401, "unauthorized", "Bad token.", "REQ-UPLOAD")
            return self._error(404, "not_found", "No such Sprite.", "REQ-LOOKUP")

        env = self._env(real_sprites_sdk, handler, tmp_path, monkeypatch)
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert info.value.status_code == 401
        assert info.value.request_id == "REQ-UPLOAD"
        assert "HTTP 401" in str(info.value) and "REQ-LOOKUP" not in str(info.value)
        assert env._token == self.OLD  # the refresh declined

    def test_candidate_token_echoed_by_refresh_lookup_never_leaks(self, real_sprites_sdk,
                                                                   tmp_path, monkeypatch):
        from sprites_environment import SpritesHttpError, _mentions

        def handler(request):
            if request.method == "PUT":
                return self._error(401, "unauthorized", "Bad token.", "REQ-UPLOAD")
            return self._error(403, "forbidden", f"token {self.NEW} may not read",
                               f"REQ-{self.NEW}")

        env = self._env(real_sprites_sdk, handler, tmp_path, monkeypatch)
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert not _mentions(info.value, (self.NEW, self.OLD))
        assert info.value.status_code == 401

    def test_candidate_token_in_upload_error_fields_is_withheld(self, real_sprites_sdk,
                                                                 tmp_path, monkeypatch):
        """Every token tried during the call is withheld, not just the current one."""
        from sprites_environment import SpritesHttpError, _mentions

        def handler(request):
            if request.method == "PUT":
                return self._error(401, "unauthorized", f"rejected; try {self.NEW}",
                                   f"REQ-{self.NEW}")
            return self._error(404, "not_found", "No such Sprite.", "REQ-LOOKUP")

        env = self._env(real_sprites_sdk, handler, tmp_path, monkeypatch)
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert not _mentions(info.value, (self.NEW,))
        assert info.value.status_code == 401
        for value in (info.value.code, info.value.error_message, info.value.request_id):
            assert self.NEW not in (value or "")
        # Only the fields holding a token are withheld; the rest is kept.
        assert info.value.code == "unauthorized" and "unauthorized" in str(info.value)
        assert "/home/sprite/f.txt" in str(info.value)

    def test_token_in_remote_path_is_never_in_message(self, real_sprites_sdk, tmp_path,
                                                      monkeypatch):
        from sprites_environment import SpritesHttpError, _mentions
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        env = self._env(real_sprites_sdk, lambda request: self._error(
            413, "file_too_large", "File is too large.", "REQ-PATH"), tmp_path, monkeypatch)
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path, remote=f"/home/sprite/{self.OLD}/f.txt")
        assert not _mentions(info.value, (self.OLD,))
        assert info.value.status_code == 413 and "REQ-PATH" in str(info.value)

    @staticmethod
    def _counting_stream(chunks):
        import httpx

        class _Stream(httpx.SyncByteStream):
            pulled = 0

            def __iter__(self):
                for chunk in chunks:
                    _Stream.pulled += len(chunk)
                    yield chunk

        return _Stream

    def test_upload_hook_reads_a_streamed_body_only_up_to_the_budget(self, real_sprites_sdk,
                                                                      tmp_path, monkeypatch):
        import httpx
        from sprites_environment import _ERROR_BODY_MAX_BYTES, SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        chunk = b"x" * 4096
        stream = self._counting_stream([chunk] * 256)  # 1 MiB, no Content-Length
        pulled_after_hook = []

        def handler(request):
            return httpx.Response(413, stream=stream())

        env = self._env(real_sprites_sdk, handler, tmp_path, monkeypatch)
        env._client._client.event_hooks["response"].append(
            lambda response: pulled_after_hook.append(stream.pulled))
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert pulled_after_hook and pulled_after_hook[0] <= _ERROR_BODY_MAX_BYTES + len(chunk)
        assert info.value.status_code == 413 and info.value.code is None
        # The SDK still got the whole body afterwards (it read it to parse).
        assert stream.pulled == 256 * len(chunk)

    def test_upload_hook_parses_a_small_streamed_body(self, real_sprites_sdk, tmp_path,
                                                      monkeypatch):
        import httpx
        import json
        from sprites_environment import SpritesHttpError
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        body = json.dumps({"error": {"code": "rate_limited", "message": "Slow down.",
                                     "request_id": "REQ-STREAM",
                                     "details": {"retry_ms": 60000}}}).encode()
        stream = self._counting_stream([body[:10], body[10:]])
        env = self._env(real_sprites_sdk, lambda request: httpx.Response(429, stream=stream()),
                        tmp_path, monkeypatch)
        with pytest.raises(SpritesHttpError) as info:
            self._upload(env, tmp_path)
        assert info.value.code == "rate_limited" and info.value.request_id == "REQ-STREAM"
        assert info.value.retry_after_ms == 60000

    def test_streaming_request_outside_upload_is_not_drained(self, real_sprites_sdk, tmp_path,
                                                             monkeypatch):
        import httpx
        from sprites_environment import _http_auth
        monkeypatch.delenv("SPRITES_TOKEN_FILE", raising=False)
        stream = self._counting_stream([b"y" * 4096] * 16)
        env = self._env(real_sprites_sdk, lambda request: httpx.Response(401, stream=stream()),
                        tmp_path, monkeypatch)
        _http_auth.saw_401 = False
        try:
            with env._client._client.stream("GET", "https://sprites.invalid/exec") as response:
                assert stream.pulled == 0
                assert response.status_code == 401
            assert _http_auth.saw_401 is True
            assert _http_auth.last_error == {"status_code": 401}
        finally:
            env._client.close()
