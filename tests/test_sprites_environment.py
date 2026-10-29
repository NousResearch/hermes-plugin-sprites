"""Unit tests for the Sprites cloud sandbox environment backend.

These exercise SpritesEnvironment against a mocked sprites-py SDK; no
network or token required. Live-API checks live under
tests/test_sprites_terminal_live.py.
"""

import json
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
