"""Streams into a Sprite: ``SpritesEnvironment.exec_prefix`` and the ``sprite_stream.py`` helper.

The helper runs against a local server that speaks the Sprites exec WebSocket protocol (non-TTY:
stream byte 0 stdin, 4 end of stdin, 1 stdout, 2 stderr, 3 exit status; a ``session_info`` text
frame first) and runs the requested argv as a real local process. No token or network needed.
"""
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from test_sprites_environment import (  # noqa: F401  (fixtures)
    hermes_runtime,
    isolated_home,
    make_env,
    sprites_sdk,
)

HELPER = Path(__file__).resolve().parents[1] / "sprite_stream.py"


# ---------------------------------------------------------------------------
# exec_prefix
# ---------------------------------------------------------------------------

class TestExecPrefix:
    def test_provider_declares_streams(self):
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("sprites_plugin_under_test", HELPER.parent / "__init__.py",
                                       submodule_search_locations=[str(HELPER.parent)])
        mod = module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert mod.SpritesProvider.supports_streams is True

    def test_env_token_passes_the_variable_name_not_the_token(self, make_env):
        env = make_env()
        prefix = env.exec_prefix()
        assert prefix[:2] == [sys.executable, str(HELPER)]
        assert prefix[prefix.index("--sprite") + 1] == "hermes-default"
        assert prefix[prefix.index("--token-env") + 1] == "SPRITES_TOKEN"
        assert "test-token" not in prefix
        assert prefix[-1] == "--" and "--no-stdin" not in prefix and "--user" not in prefix

    def test_token_file_passes_the_path(self, make_env, tmp_path, monkeypatch):
        path = tmp_path / "secrets" / "sprites-token"
        path.parent.mkdir()
        path.write_text("file-token\n")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        prefix = make_env().exec_prefix()
        assert prefix[prefix.index("--token-file") + 1] == str(path)
        assert "--token-env" not in prefix and "file-token" not in prefix

    def test_user_and_no_stdin(self, make_env):
        prefix = make_env().exec_prefix(user="pn", interactive=False)
        assert prefix[prefix.index("--user") + 1] == "pn"
        assert "--no-stdin" in prefix

    def test_token_only_in_secret_scope_cannot_stream(self, make_env, monkeypatch):
        env = make_env()
        monkeypatch.delenv("SPRITES_TOKEN")
        assert env.exec_prefix() is None


# ---------------------------------------------------------------------------
# sprite_stream.py against a local exec server
# ---------------------------------------------------------------------------

class _FakeSprites:
    """Exec WebSocket and the session kill endpoint behind ONE port, like api.sprites.dev.

    websockets serves GET handshakes only, so a small front door sends each connection to the
    WebSocket server or to an HTTP server by its first bytes."""

    def __init__(self):
        from websockets.sync.server import serve
        self.sessions = []
        self.kills = []
        self.auth = []
        self._ws = serve(self._handle, "127.0.0.1", 0)
        self.ws_port = self._ws.socket.getsockname()[1]
        threading.Thread(target=self._ws.serve_forever, daemon=True).start()
        fake = self

        class Kill(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.kills.append((self.path, json.loads(body or b"{}"), self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        self._http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Kill)
        http_port = self._http.server_address[1]
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

        self._front = socket.create_server(("127.0.0.1", 0))
        self.port = self._front.getsockname()[1]
        self._closed = False

        def splice(a, b):
            try:
                while data := a.recv(65536):
                    b.sendall(data)
            except OSError:
                pass
            for s in (a, b):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        def route(conn):
            first = conn.recv(4, socket.MSG_PEEK)
            back = socket.create_connection(("127.0.0.1", http_port if first == b"POST" else self.ws_port))
            threading.Thread(target=splice, args=(conn, back), daemon=True).start()
            splice(back, conn)

        def accept():
            while not self._closed:
                try:
                    conn, _ = self._front.accept()
                except OSError:
                    return
                threading.Thread(target=route, args=(conn,), daemon=True).start()

        threading.Thread(target=accept, daemon=True).start()

    @property
    def api(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self._closed = True
        self._front.close()
        self._ws.shutdown()
        self._http.shutdown()

    def _handle(self, ws):
        q = parse_qs(urlparse(ws.request.path).query)
        self.auth.append(ws.request.headers.get("Authorization"))
        argv = q["cmd"]
        sid = f"sess-{len(self.sessions) + 1}"
        self.sessions.append({"id": sid, "argv": argv, "env": q.get("env", [])})
        ws.send(json.dumps({"type": "session_info", "session_id": sid, "tty": False}))
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def feed():
            try:
                for frame in ws:
                    if frame[:1] == b"\x00":
                        proc.stdin.write(frame[1:]); proc.stdin.flush()
                    elif frame[:1] == b"\x04":
                        proc.stdin.close()
                        return
            except Exception:
                pass

        def pipe(src, tag):
            while chunk := os.read(src.fileno(), 65536):
                ws.send(tag + chunk)

        threading.Thread(target=feed, daemon=True).start()
        err = threading.Thread(target=pipe, args=(proc.stderr, b"\x02"), daemon=True)
        err.start()
        pipe(proc.stdout, b"\x01")
        err.join()
        ws.send(b"\x03" + bytes([proc.wait() & 0xFF]))


@pytest.fixture()
def fake():
    server = _FakeSprites()
    yield server
    server.close()


def _helper(fake, *argv, extra=(), token="tok-1"):
    cmd = [sys.executable, str(HELPER), "--api", fake.api, "--sprite", "s1",
           "--token-env", "T", *extra, "--", *argv]
    return cmd, {**os.environ, "T": token}


def test_binary_stream_round_trips_and_exit_status_passes_through(fake):
    payload = bytes(range(256)) * 4096  # 1 MiB, every byte value
    cmd, env = _helper(fake, sys.executable, "-c",
                       "import sys; d=sys.stdin.buffer.read(); sys.stdout.buffer.write(d); "
                       "sys.stderr.write('done'); sys.exit(7)")
    proc = subprocess.run(cmd, input=payload, capture_output=True, env=env, timeout=60)
    assert proc.stdout == payload
    assert proc.stderr == b"done"
    assert proc.returncode == 7
    assert fake.auth == ["Bearer tok-1"]


def test_output_is_delivered_before_the_command_ends(fake):
    """An RFB server writes its banner and then waits: the helper must not buffer it."""
    cmd, env = _helper(fake, sys.executable, "-u", "-c",
                       "import sys; sys.stdout.write('RFB 003.008\\n'); sys.stdout.flush(); sys.stdin.read()")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    try:
        assert proc.stdout.read(12) == b"RFB 003.008\n"
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    assert proc.returncode == 0


def test_no_stdin_ends_remote_stdin_at_once(fake):
    cmd, env = _helper(fake, "cat", extra=("--no-stdin",))
    proc = subprocess.run(cmd, stdin=subprocess.PIPE, capture_output=True, env=env, timeout=30)
    assert proc.returncode == 0 and proc.stdout == b""


def test_user_wraps_the_command_in_sudo_with_that_users_home(fake):
    cmd, env = _helper(fake, "true", extra=("--user", "pn"))
    subprocess.run(cmd, capture_output=True, env=env, timeout=30)
    assert fake.sessions[0]["argv"] == ["sudo", "-n", "-H", "-u", "pn", "--", "true"]


def test_env_pairs_are_sent(fake):
    cmd, env = _helper(fake, "true", extra=("--env", "HOME=/root"))
    subprocess.run(cmd, capture_output=True, env=env, timeout=30)
    assert fake.sessions[0]["env"] == ["HOME=/root"]


def test_sigkilled_stream_has_its_session_killed(fake):
    """Hermes ends streams with Popen.kill(); the session must not outlive it (it keeps the Sprite awake)."""
    cmd, env = _helper(fake, "sleep", "30")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    _wait_for(lambda: fake.sessions)
    time.sleep(0.5)  # past session_info: the reaper is running
    proc.kill()
    proc.wait(timeout=10)
    _wait_for(lambda: fake.kills)
    assert fake.kills == [("/v1/sprites/s1/exec/sess-1/kill", {"signal": "TERM"}, "Bearer tok-1")]


def test_sigterm_also_kills_the_session(fake):
    cmd, env = _helper(fake, "sleep", "30")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    _wait_for(lambda: fake.sessions)
    time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    _wait_for(lambda: fake.kills)
    assert [k[0] for k in fake.kills] == ["/v1/sprites/s1/exec/sess-1/kill"]


def test_a_finished_command_leaves_no_kill(fake):
    cmd, env = _helper(fake, "true")
    assert subprocess.run(cmd, capture_output=True, env=env, timeout=30).returncode == 0
    time.sleep(1.0)
    assert fake.kills == []


def _wait_for(pred, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)
