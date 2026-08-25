"""Unit tests for the Sprites cloud sandbox environment backend.

These exercise SpritesEnvironment against a mocked sprites-py SDK; no
network or token required. Live-API checks live under
tests/integration/test_sprites_terminal.py.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock sprites-py SDK
# ---------------------------------------------------------------------------

class _NotFoundError(Exception):
    pass


class _SpriteError(Exception):
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

@pytest.fixture()
def sprites_sdk(monkeypatch):
    return _patch_sprites_imports(monkeypatch)


@pytest.fixture()
def make_env(sprites_sdk, monkeypatch):
    """Build a SpritesEnvironment instance against a mocked SDK.

    Returns a factory; keyword args mirror SpritesEnvironment.__init__.
    The factory accepts an optional ``get_side_effect`` to control what
    ``client.get_sprite()`` does (e.g. raise NotFoundError to force create).
    """
    monkeypatch.setenv("SPRITES_TOKEN", "test-token")
    # Don't try to lazy-install the SDK during tests
    monkeypatch.setattr(
        "tools.lazy_deps.ensure", lambda *a, **k: None, raising=False
    )
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
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        import __init__ as plugin_pkg
        import tools.terminal_tool as tt
        from agent import terminal_env_registry as reg

        reg._reset_for_tests()
        try:
            reg.register_provider(plugin_pkg.SpritesProvider())
            monkeypatch.setenv("TERMINAL_ENV", "sprites")
            monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
            monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
            assert tt._session_isolation_enabled() is True
            # Docker-only paths (workspace mounts, container teardown) stay off.
            assert tt._docker_session_isolation_enabled() is False
            # An ordinary session task id no longer collapses onto the shared key.
            assert tt._resolve_container_task_id("session-abc123") != "default"
            # Persistent mode keeps the documented shared-Sprite contract.
            monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
            assert tt._session_isolation_enabled() is False
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

    def test_unlink_failure_raises(self, make_env):
        env = make_env(task_id="del", persistent_filesystem=True)
        remote = MagicMock()
        remote.unlink.side_effect = OSError("websocket dropped")
        env._fs = MagicMock()
        env._fs.__truediv__ = MagicMock(return_value=remote)
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
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        import __init__ as plugin_pkg
        import tools.terminal_tool as tt
        from agent import terminal_env_registry as reg

        reg._reset_for_tests()
        request_cleanup = reg._reset_for_tests
        try:
            reg.register_provider(plugin_pkg.SpritesProvider())
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

        def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
            captured["env_type"] = env_type
            captured["container_config"] = kwargs.get("container_config")
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
        monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        tt.terminal_tool(command="pwd")

        assert captured["env_type"] == "sprites"
        cc = captured["container_config"]
        assert cc is not None, (
            "sprites must be in terminal_tool's container_config builder set; "
            "container_config=None silently discards container_persistent"
        )
        assert cc["container_persistent"] is False
        request_cleanup()

    def test_create_environment_passes_persistence_and_task_id(self, monkeypatch):
        """Registry dispatch: _create_environment falls through to the provider."""
        import tools.terminal_tool as tt
        import sprites_environment as sprites_mod
        from agent import terminal_env_registry as reg

        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        import __init__ as plugin_pkg  # noqa: F401 — the plugin package

        captured = {}

        class _FakeSpritesEnv:
            def __init__(self, cwd, timeout, persistent_filesystem, task_id):
                captured.update(
                    cwd=cwd,
                    timeout=timeout,
                    persistent_filesystem=persistent_filesystem,
                    task_id=task_id,
                )

        monkeypatch.setattr(sprites_mod, "SpritesEnvironment", _FakeSpritesEnv)
        reg._reset_for_tests()
        try:
            reg.register_provider(plugin_pkg.SpritesProvider())

            env = tt._create_environment(
                env_type="sprites",
                image="ignored",
                cwd="/root",
                timeout=60,
                container_config={"container_persistent": False},
                task_id="tid-ephemeral",
            )

            assert captured["persistent_filesystem"] is False
            assert captured["task_id"] == "tid-ephemeral"
            assert captured["cwd"] == "/root"
            assert getattr(env, "_hermes_backend_name", None) == "sprites"
        finally:
            reg._reset_for_tests()

    def test_no_base_url_kwarg(self, make_env, sprites_sdk):
        """SpritesClient is constructed without a base_url override (endpoint is fixed)."""
        env = make_env(task_id="urlcheck")
        sprites_mod, _ = sprites_sdk
        _, kwargs = sprites_mod.SpritesClient.call_args
        assert "base_url" not in kwargs


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

    def test_timeout_surfaces_124(self, make_env, sprites_sdk):
        env = make_env(task_id="rbto")
        _, exc_mod = sprites_sdk
        cmd = MagicMock()
        cmd.combined_output.side_effect = exc_mod.TimeoutError("deadline")
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("sleep 999", timeout=1)
        handle.wait()
        assert handle.returncode == 124


# ---------------------------------------------------------------------------
# File-sync push (upload_fn behavior)
# ---------------------------------------------------------------------------

class TestFileSyncPush:
    def test_upload_writes_via_filesystem_api(self, make_env, tmp_path):
        env = make_env(task_id="fs")
        # Build a fake host file
        host_file = tmp_path / "secret.txt"
        host_file.write_bytes(b"hello")

        # Mock the SpritePath returned by `fs / remote_path`
        remote_path_obj = MagicMock()
        env._fs.__truediv__.return_value = remote_path_obj

        env._sprite_upload(str(host_file), "/home/sprite/.hermes/foo")
        remote_path_obj.parent.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )
        remote_path_obj.write_bytes.assert_called_once_with(b"hello")

    def test_delete_invokes_unlink_per_path(self, make_env):
        env = make_env(task_id="fsdel")
        remote_obj = MagicMock()
        env._fs.__truediv__.return_value = remote_obj
        env._sprite_delete(["/home/sprite/.hermes/a", "/home/sprite/.hermes/b"])
        # Each path → one unlink call
        assert remote_obj.unlink.call_count == 2
        remote_obj.unlink.assert_any_call(missing_ok=True)


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
