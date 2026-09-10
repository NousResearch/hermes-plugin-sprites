"""Integration tests for the Sprites terminal backend.

Requires SPRITES_TOKEN to be set. Run with:
    TERMINAL_ENV=sprites pytest tests/integration/test_sprites_terminal.py -v

SAFETY: every test runs in a run-unique ``hermes-test-{uuid8}-…`` Sprite
namespace (see ``_force_sprites``), so the suite does not touch the real
profile Sprite names production naming emits (a production task id would
have to spell out this run's random uuid hex to collide).
"""

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# Capture the token at import time. The project-wide hermetic conftest
# wipes anything ending in _TOKEN before each test runs, so we save the
# value here and re-inject it via the autouse fixture below.
_SPRITES_TOKEN = os.getenv("SPRITES_TOKEN")
if not _SPRITES_TOKEN:
    pytest.skip("SPRITES_TOKEN not set", allow_module_level=True)

# Import terminal_tool via importlib to avoid tools/__init__.py side effects.
# IMPORTANT: this creates a module object DISTINCT from `tools.terminal_tool`;
# every helper and global this file touches (terminal_tool, cleanup_vm,
# _resolve_container_task_id, _active_environments) must come from THIS
# module object, or assertions would read a registry the executed code never
# wrote to. (sprites_environment is imported normally by both, so
# patching it affects the executed path.)
import importlib.util

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

spec = importlib.util.spec_from_file_location(
    "terminal_tool", parent_dir / "tools" / "terminal_tool.py"
)
terminal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terminal_module)

terminal_tool = terminal_module.terminal_tool
cleanup_vm = terminal_module.cleanup_vm
_resolve_container_task_id = terminal_module._resolve_container_task_id
_active_environments = terminal_module._active_environments

# One unique Sprite namespace per test run, so parallel/repeated runs don't
# collide and — critically — the suite never touches a production Sprite name.
_RUN_ID = uuid.uuid4().hex[:8]


def _test_sprite_name(task_id: str) -> str:
    from sprites_environment import _collapse_slug
    return f"hermes-test-{_RUN_ID}-{_collapse_slug(task_id) or 'default'}"


@pytest.fixture(autouse=True)
def _force_sprites(monkeypatch):
    # Re-inject the token the hermetic conftest deleted.
    monkeypatch.setenv("SPRITES_TOKEN", _SPRITES_TOKEN)
    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    # Match the documented "ephemeral test" default — tests clean up after themselves.
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    # Sandbox every test into the run-unique namespace: without this, task-id
    # collapse would resume the operator's REAL hermes-{profile}-default
    # Sprite and the ephemeral teardown would DELETE it (filesystem and all).
    # Both naming paths are pinned: persistent envs resolve via
    # _resolve_sprite_name, ephemeral envs (this suite's default) via
    # _ephemeral_sprite_name.
    import sprites_environment as sprites_mod
    monkeypatch.setattr(sprites_mod, "_resolve_sprite_name", _test_sprite_name)
    monkeypatch.setattr(sprites_mod, "_ephemeral_sprite_name", _test_sprite_name)


@pytest.fixture()
def task_id(request):
    """Unique task_id per test; environment is cleaned up afterwards.

    Cleanup must use the CONTAINER key the env was registered under —
    `_resolve_container_task_id` is mode-dependent: with this suite's
    non-persistent config, session isolation keys per task id; under
    persistent mode ordinary ids collapse to "default". Resolving at
    teardown time (same env state) always yields the registration key.
    """
    tid = f"sprites_test_{request.node.name}"
    yield tid
    cleanup_vm(_resolve_container_task_id(tid))


def _run(command, task_id, **kwargs):
    result = terminal_tool(command, task_id=task_id, **kwargs)
    return json.loads(result)


class TestSpritesBasic:
    def test_echo(self, task_id):
        r = _run("echo 'Hello from a Sprite!'", task_id)
        assert r["exit_code"] == 0
        assert "Hello from a Sprite!" in r["output"]

    def test_nonzero_exit(self, task_id):
        r = _run("exit 42", task_id)
        assert r["exit_code"] == 42

    def test_os_info(self, task_id):
        r = _run("uname -a", task_id)
        assert r["exit_code"] == 0
        assert "Linux" in r["output"]

    def test_python_available(self, task_id):
        r = _run("python3 --version || python --version", task_id)
        assert r["exit_code"] == 0
        assert "Python" in r["output"]


class TestSpritesFilesystem:
    def test_write_and_read_file(self, task_id):
        _run("echo 'sprites content' > /tmp/sprites_test.txt", task_id)
        r = _run("cat /tmp/sprites_test.txt", task_id)
        assert r["exit_code"] == 0
        assert "sprites content" in r["output"]

    def test_env_var_persistence(self, task_id):
        _run("export SPRITES_TEST_VAR=heyo", task_id)
        r = _run("echo $SPRITES_TEST_VAR", task_id)
        assert r["exit_code"] == 0
        assert "heyo" in r["output"]


class TestSpritesIdentity:
    def test_runs_inside_a_sprite(self, task_id):
        """Output should confirm we're in a Sprite, not on the host."""
        r = _run("sprite-env info 2>/dev/null || echo MISSING", task_id)
        assert r["exit_code"] == 0
        if "MISSING" in r["output"]:
            pytest.skip("sprite-env CLI not present inside the Sprite")
        # `_resolve_container_task_id` collapses every ordinary task_id to
        # "default", and the suite pins Sprite naming to the run-unique test
        # namespace (see _force_sprites). Production naming semantics are
        # covered by tests/tools/test_sprites_environment.py::TestSpriteNaming.
        expected_name = _test_sprite_name(_resolve_container_task_id(task_id))
        assert expected_name in r["output"]
        # Sanity: the boot_id from inside the Sprite must differ from this
        # process's view (i.e. command did NOT run on the host).
        host_boot = open("/proc/sys/kernel/random/boot_id").read().strip()
        r2 = _run("cat /proc/sys/kernel/random/boot_id", task_id)
        assert host_boot not in r2["output"]


class TestSpritesPersistence:
    def test_filesystem_survives_session_recycle(self):
        """Write a marker, tear down the env, resume — file should still be there.

        NOTE: `_resolve_container_task_id` is persistence-mode-dependent.
        Under persistent mode (this test) ordinary task ids collapse to
        "default", so `_active_environments` keys the live env under
        "default" — NOT under the raw task string; `cleanup_vm(<raw task>)`
        would pop nothing (a vacuous recycle that silently reuses the same
        in-memory env object). Tear down via the registration key so the
        second _run genuinely re-creates the environment and resumes the
        Sprite by name over the API. All registry reads use the SAME module
        object the commands executed through (`terminal_module`, bound at
        import).
        """
        task = "sprites_test_persist"
        # Persistence must be set BEFORE computing the env key: with
        # container_persistent=false this suite runs session-isolated
        # (per-task keys), while persistent mode collapses ordinary ids
        # to the shared "default" key — the key is mode-dependent.
        os.environ["TERMINAL_CONTAINER_PERSISTENT"] = "true"
        env_key = _resolve_container_task_id(task)
        try:
            _run("echo 'survive' > /tmp/sprites_persist.txt", task)

            # Prove the env actually lives under the collapsed key, then
            # recycle it. persistent=true → the Sprite itself stays alive.
            assert env_key in _active_environments, (
                f"env registered under {list(_active_environments)}, "
                f"expected key {env_key!r}"
            )
            first_env = _active_environments[env_key]
            cleanup_vm(env_key)
            assert env_key not in _active_environments

            r = _run("cat /tmp/sprites_persist.txt", task)
            assert r["exit_code"] == 0
            assert "survive" in r["output"]
            # And the read ran in a NEW environment object (a real resume,
            # not a lingering reference to the old one).
            assert _active_environments.get(env_key) is not first_env
        finally:
            os.environ["TERMINAL_CONTAINER_PERSISTENT"] = "false"
            # The live env was constructed while persistence was "true", so
            # its baked-in _persistent flag would make plain cleanup LEAVE the
            # test Sprite running (billing forever). Flip the flag on the
            # object before cleanup so teardown genuinely deletes it.
            live = _active_environments.get(env_key)
            if live is not None and hasattr(live, "_persistent"):
                live._persistent = False
            cleanup_vm(env_key)
