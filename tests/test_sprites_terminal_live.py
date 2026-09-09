"""Opt-in live checks with isolated profiles and owned, run-unique Sprites.

Run via Hermes's scripts/run_tests.sh with --sprites-live-token-file pointing
to a private token file. No operator profile or existing Sprite is adopted.
"""

import json
from pathlib import Path
import shutil
import stat
import threading
import time
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def live(monkeypatch, tmp_path, request):
    token_path = request.config.getoption("--sprites-live-token-file")
    if not token_path:
        pytest.skip("Live tests require explicit --sprites-live-token-file authorization")
    token_path = Path(token_path)
    if token_path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        pytest.fail("Live token file must be private (chmod 600)")
    token = token_path.read_text().strip()
    if not token:
        pytest.fail("Live token file is empty")

    import httpx
    from sprites import SpritesClient
    from sprites.exceptions import NotFoundError
    from sprites.types import URLSettings
    import yaml

    run_id = uuid4().hex[:12]
    label = "hermes-live-" + uuid4().hex
    attempted = set()
    created = {}
    managers = []
    root = tmp_path / ".hermes"
    profile = root / "profiles" / f"live-{run_id}-a"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("SPRITES_TOKEN", token)
    monkeypatch.delenv("SPRITE_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    from hermes_cli import plugins
    from agent import terminal_env_registry as registry
    import tools.terminal_tool as terminal
    from tools.terminal_tool_lifecycle import cleanup_vm

    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: root / "empty-bundled")
    monkeypatch.setattr(terminal, "_active_environments", {})
    monkeypatch.setattr(terminal, "_last_activity", {})
    monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
    control = httpx.Client(
        base_url="https://api.sprites.dev", headers={"Authorization": "Bearer " + token}, timeout=60,
    )

    def allowed(name):
        return name.startswith((f"hermes-live-{run_id}-", f"hermes-eph-live-{run_id}-"))

    def info(name):
        assert allowed(name), "Refusing access outside the test namespace"
        response = control.get("/v1/sprites/" + name)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def owned(name, data):
        assert name in attempted and label in data.get("labels", []), "Not a Sprite created by this test"
        if name in created:
            assert data["id"] == created[name], "Test Sprite identity changed"

    real_create = SpritesClient.create_sprite
    real_get = SpritesClient.get_sprite
    real_destroy = SpritesClient.destroy_sprite

    def create(client, name, *args, **kwargs):
        assert info(name) is None, "Refusing to adopt an existing Sprite"
        attempted.add(name)
        sprite = real_create(client, name, *args, labels=[label], url_settings=URLSettings(auth="sprite"), **kwargs)
        created[name] = sprite.id
        print(json.dumps({"created": name, "id": sprite.id}), flush=True)
        return sprite

    def get(client, name):
        data = info(name)
        if data is None:
            raise NotFoundError("Test Sprite does not exist")
        owned(name, data)
        return real_get(client, name)

    def destroy(client, name):
        data = info(name)
        if data is not None:
            owned(name, data)
            real_destroy(client, name)

    monkeypatch.setattr(SpritesClient, "create_sprite", create)
    monkeypatch.setattr(SpritesClient, "get_sprite", get)
    monkeypatch.setattr(SpritesClient, "destroy_sprite", destroy)

    def select_profile(suffix, persistent):
        # Model separate CLI invocations: a previous session's remote cwd must
        # not seed a new profile that has never created that directory.
        monkeypatch.setattr(terminal, "_session_cwd", {})
        home = root / "profiles" / f"live-{run_id}-{suffix}"
        home.mkdir(parents=True, exist_ok=True)
        skill = home / "skills" / "live-fixture" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("Harmless live sync fixture\n")
        plugin_dir = home / "plugins" / "sprites"
        if not plugin_dir.exists():
            shutil.copytree(
                Path(__file__).resolve().parents[1], plugin_dir,
                ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "tests"),
            )
        (home / "config.yaml").write_text(yaml.safe_dump({
            "plugins": {"enabled": ["sprites"]},
            "terminal": {"backend": "sprites", "cwd": "/root", "timeout": 60,
                         "container_persistent": persistent},
        }))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(terminal, "_terminal_config_bridge_attempted", False)
        manager = plugins.get_plugin_manager()
        monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
        manager.discover_and_load(force=True)
        managers.append(manager)
        assert registry.get_provider("sprites") is not None
        return home

    def run(command, task="default", **kwargs):
        result = json.loads(terminal.terminal_tool(command, task_id=task, **kwargs))
        assert result.get("status") != "disabled", result.get("error", "Backend disabled")
        return result

    try:
        yield select_profile, run, terminal, cleanup_vm, info, run_id
    finally:
        # Cleanup the environment clients first, then any persistent/orphaned test Sprite.
        cleanup_errors = []
        for key in list(terminal._active_environments):
            try:
                cleanup_vm(key)
            except Exception as exc:
                cleanup_errors.append((str(key), type(exc).__name__))
        for name in sorted(attempted):
            try:
                data = info(name)
                if data is not None:
                    owned(name, data)
                    control.delete("/v1/sprites/" + name).raise_for_status()
                assert info(name) is None, "Deletion not confirmed"
                print(json.dumps({"deleted_and_verified": name}), flush=True)
            except Exception as exc:
                cleanup_errors.append((name, type(exc).__name__))
        for manager in managers:
            manager.unload()
        registry._reset_for_tests()
        control.close()
        assert not cleanup_errors, f"Test Sprite cleanup needs attention: {cleanup_errors}"


def test_live_execution_persistence_profiles_and_cleanup(live):
    select_profile, run, terminal, cleanup_vm, info, run_id = live
    home = select_profile("a", True)
    first = run("printf 'hello'; printf 'warning' >&2; exit 7")
    assert first["exit_code"] == 7 and "hello" in first["output"] and "warning" in first["output"]
    env = terminal._active_environments[terminal._resolve_container_task_id("default")]
    first_name = env._sprite_name
    assert info(first_name)["url_settings"]["auth"] == "sprite"
    result = run("uname -s; python3 -c 'print(6 * 7)'; pwd")
    assert result["exit_code"] == 0 and "Linux" in result["output"] and "42" in result["output"]
    assert result["output"].splitlines()[-1] == "/root"
    assert run("mkdir -p /tmp/hermes-live-work")["exit_code"] == 0
    assert run("pwd", workdir="/tmp/hermes-live-work")["output"].strip() == "/tmp/hermes-live-work"
    synced_skill = f"{env._remote_home}/.hermes/skills/live-fixture/SKILL.md"
    assert "Harmless live sync fixture" in run(f"cat {synced_skill}")["output"]
    (home / "skills" / "live-fixture" / "SKILL.md").unlink()
    env._sync_manager.sync(force=True)
    assert run(f"test ! -e {synced_skill}")["exit_code"] == 0
    assert run('test -z "${SPRITES_TOKEN+x}${SPRITE_TOKEN+x}"')["exit_code"] == 0
    assert run("printf persistent > /tmp/hermes-live-marker")["exit_code"] == 0
    assert run("export HERMES_LIVE_MARKER=kept")["exit_code"] == 0
    assert "kept" in run("printf '%s' \"$HERMES_LIVE_MARKER\"")["output"]
    cleanup_vm(terminal._resolve_container_task_id("default"))
    assert info(first_name) is not None
    assert "persistent" in run("cat /tmp/hermes-live-marker")["output"]
    resumed = terminal._active_environments[terminal._resolve_container_task_id("default")]
    assert resumed is not env and resumed._sprite_name == first_name

    # Client timeout is not a remote kill guarantee. The remote workload is bounded regardless.
    timed = run("sleep 4; printf finished > /tmp/hermes-live-timeout", timeout=1)
    assert timed["exit_code"] == 124, timed
    time.sleep(5)
    after_timeout = run("test -f /tmp/hermes-live-timeout && echo continued || echo stopped")
    assert after_timeout["exit_code"] == 0
    print(json.dumps({"remote_after_client_timeout": after_timeout["output"].strip()}), flush=True)
    # A host interrupt also returns promptly without promising remote termination.
    from tools.interrupt import set_interrupt
    thread_id = threading.get_ident()
    timer = threading.Timer(1, lambda: set_interrupt(True, thread_id))
    timer.start()
    try:
        interrupted = run("sleep 6; printf finished > /tmp/hermes-live-interrupt", timeout=15)
        assert interrupted["exit_code"] == 130
    finally:
        timer.cancel()
        timer.join()
        set_interrupt(False, thread_id)
    time.sleep(7)
    after_interrupt = run("test -f /tmp/hermes-live-interrupt && echo continued || echo stopped")
    assert after_interrupt["exit_code"] == 0
    print(json.dumps({"remote_after_client_interrupt": after_interrupt["output"].strip()}), flush=True)
    cleanup_vm(terminal._resolve_container_task_id("default"))

    select_profile("b", True)
    isolated = run("test ! -e /tmp/hermes-live-marker")
    assert isolated["exit_code"] == 0, isolated
    second = terminal._active_environments[terminal._resolve_container_task_id("default")]
    assert second._sprite_name != first_name
    cleanup_vm(terminal._resolve_container_task_id("default"))

    select_profile("c", False)
    task = f"live-{run_id}-ephemeral"
    assert run("echo ephemeral", task=task)["exit_code"] == 0
    key = terminal._resolve_container_task_id(task)
    ephemeral_name = terminal._active_environments[key]._sprite_name
    cleanup_vm(key)
    assert info(ephemeral_name) is None
