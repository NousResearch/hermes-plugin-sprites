"""Streams into a Sprite: ``SpritesEnvironment.exec_prefix`` and the ``sprite_stream.py`` helper.

The helper runs against a local server that speaks the Sprites exec WebSocket protocol (non-TTY:
stream byte 0 stdin, 4 end of stdin, 1 stdout, 2 stderr, 3 exit status; a ``session_info`` text
frame first) and runs the requested argv as a real local process, plus the session kill endpoint
behind the same port. No token or network needed.
"""
import http.server
import json
import os
import signal
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
    def test_provider_declares_streams(self, monkeypatch):
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("sprites_plugin_under_test", HELPER.parent / "__init__.py",
                                       submodule_search_locations=[str(HELPER.parent)])
        mod = module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        assert mod.SpritesProvider.supports_streams is True

    def test_env_token_passes_the_variable_name_not_the_token(self, make_env):
        prefix = make_env().exec_prefix()
        assert prefix[:2] == [sys.executable, str(HELPER)]
        assert prefix[prefix.index("--sprite") + 1] == "hermes-default"
        assert prefix[prefix.index("--token-env") + 1] == "SPRITES_TOKEN"
        assert not any("test-token" in word for word in prefix)
        assert prefix[-1] == "--" and "--no-stdin" not in prefix and "--user" not in prefix

    def test_token_file_passes_the_path(self, make_env, tmp_path, monkeypatch):
        path = tmp_path / "secrets" / "sprites-token"
        path.parent.mkdir()
        path.write_text("file-token\n")
        monkeypatch.setenv("SPRITES_TOKEN_FILE", str(path))
        prefix = make_env().exec_prefix()
        assert prefix[prefix.index("--token-file") + 1] == str(path)
        assert "--token-env" not in prefix and not any("file-token" in word for word in prefix)

    def test_user_and_no_stdin(self, make_env):
        prefix = make_env().exec_prefix(user="pn", interactive=False)
        assert prefix[prefix.index("--user") + 1] == "pn"
        assert "--no-stdin" in prefix

    def test_token_only_in_secret_scope_cannot_stream(self, make_env, monkeypatch):
        env = make_env()
        monkeypatch.delenv("SPRITES_TOKEN")
        assert env.exec_prefix() is None


# ---------------------------------------------------------------------------
# Local Sprites exec server
# ---------------------------------------------------------------------------

class _FakeSprites:
    """Exec WebSocket and the session kill endpoint behind ONE port, like api.sprites.dev.

    websockets serves GET handshakes only, so a small front door sends each connection to the
    WebSocket server or to an HTTP server by its first bytes. argv[0] values starting with ``@``
    script protocol misbehaviour instead of running a command."""

    def __init__(self):
        from websockets.sync.server import serve
        self.sessions = []
        self.procs = {}
        self.kills = []
        self.auth = []
        self.url_lengths = []
        self.session_info_delay = 0.0
        self._ws = serve(self._handle, "127.0.0.1", 0, max_size=None)
        ws_port = self._ws.socket.getsockname()[1]
        threading.Thread(target=self._ws.serve_forever, daemon=True).start()
        fake = self

        class Kill(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                fake.kills.append((self.path, body, self.headers.get("Authorization")))
                sid = self.path.split("/exec/")[1].split("/")[0]
                proc = fake.procs.get(sid)
                if proc is None or proc.poll() is not None:
                    self.send_response(404)
                else:
                    proc.send_signal(getattr(signal, "SIG" + body.get("signal", "TERM")))
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        self._http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Kill)
        http_port = self._http.server_address[1]
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

        self._front = socket.create_server(("127.0.0.1", 0))
        self.port = self._front.getsockname()[1]

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
            back = socket.create_connection(("127.0.0.1", http_port if first == b"POST" else ws_port))
            threading.Thread(target=splice, args=(conn, back), daemon=True).start()
            splice(back, conn)

        def accept():
            while True:
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
        self._front.close()
        self._ws.shutdown()
        self._http.shutdown()
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.kill()

    def _handle(self, ws):
        self.url_lengths.append(len(ws.request.path))
        q = parse_qs(urlparse(ws.request.path).query)
        self.auth.append(ws.request.headers.get("Authorization"))
        argv = q["cmd"]
        if argv[0] == "@not-json":
            ws.send("<html>bad gateway</html>")
            return
        sid = f"sess-{len(self.sessions) + 1}"
        self.sessions.append({"id": sid, "argv": argv, "env": q.get("env", [])})
        time.sleep(self.session_info_delay)
        ws.send(json.dumps({"type": "session_info", "session_id": sid, "tty": False}))
        if argv[0] == "@error":
            ws.send(json.dumps({"type": "error", "error": {"code": "exec_limit", "message": "too many"}}))
            time.sleep(5)
            return
        if argv[0] == "@bare-exit":
            ws.send(b"\x03")
            return
        if argv[0] == "@drop":
            return  # socket closes with no exit status
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.procs[sid] = proc

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
            try:
                while chunk := os.read(src.fileno(), 65536):
                    ws.send(tag + chunk)
            except Exception:
                pass

        threading.Thread(target=feed, daemon=True).start()
        err = threading.Thread(target=pipe, args=(proc.stderr, b"\x02"), daemon=True)
        err.start()
        pipe(proc.stdout, b"\x01")
        err.join()
        try:
            ws.send(b"\x03" + bytes([proc.wait() & 0xFF]))
        except Exception:
            pass


@pytest.fixture()
def fake():
    server = _FakeSprites()
    yield server
    server.close()


def _helper(fake, *argv, extra=(), token="tok-1", token_args=None):
    auth = token_args or ["--token-env", "T"]
    cmd = [sys.executable, str(HELPER), "--api", fake.api, "--sprite", "s1", *auth, *extra, "--", *argv]
    return cmd, {**os.environ, "T": token}


def _wait_for(pred, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Stream behaviour
# ---------------------------------------------------------------------------

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


def test_a_command_too_long_for_a_url_runs_through_the_stdin_bootstrap(fake):
    """Bot Desktop's setup script embeds its launcher and wallpaper: ~30 KB of argv."""
    big = "x" * 40000
    script = f"import sys; data=sys.stdin.buffer.read(); print(len(sys.argv[1]), data.decode())"
    cmd, env = _helper(fake, sys.executable, "-c", script, big)
    # The bootstrap in the Sprite is `python3 -c ...`; locally, python3 must resolve.
    proc = subprocess.run(cmd, input=b"after", capture_output=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b"40000 after\n"
    assert fake.url_lengths[0] < 6000
    assert fake.sessions[0]["argv"][0] == "python3"


def test_the_size_limit_counts_the_encoded_url(fake):
    """Shell punctuation percent-escapes to three bytes each: 3 KB of it is a 9 KB URL."""
    script = "true # " + "$'\"/=&\n" * 400
    assert len(script.encode()) < 4096
    cmd, env = _helper(fake, "bash", "-c", script, extra=[x for i in range(8) for x in ("--env", f"K{i}=v")])
    assert subprocess.run(cmd, capture_output=True, env=env, timeout=60).returncode == 0
    assert fake.sessions[0]["argv"][0] == "python3"
    assert fake.url_lengths[0] < 8192


def test_the_bootstrap_hands_over_default_signal_handling(fake):
    """An ignored SIGPIPE inherited through exec breaks `yes | head` and pipefail scripts."""
    # Not Python (it ignores both at startup itself): a shell, which can never undo an inherited ignore.
    cmd, env = _helper(fake, "sh", "-c", "grep SigIgn /proc/$$/status", "x" * 8000)
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert fake.sessions[0]["argv"][0] == "python3"
    mask = int(proc.stdout.split()[1], 16)
    assert mask & ((1 << (signal.SIGPIPE - 1)) | (1 << (signal.SIGXFSZ - 1))) == 0


def test_a_missing_command_under_the_bootstrap_exits_127(fake):
    cmd, env = _helper(fake, "/nonexistent/cmd", "x" * 8000)
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert proc.returncode == 127 and b"Traceback" not in proc.stderr


def test_an_undecodable_argument_still_runs(fake):
    """A surrogate-escaped name cannot be put in a URL; it goes through the bootstrap intact."""
    name = os.fsdecode(b"caf\xe9")
    cmd, env = _helper(fake, sys.executable, "-c",
                       "import os,sys; sys.stdout.buffer.write(os.fsencode(sys.argv[1]))", name)
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b"caf\xe9"


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


def test_token_file_is_read_like_the_backend_reads_it(fake, tmp_path):
    path = tmp_path / "secrets" / "sprites-token"
    path.parent.mkdir()
    path.write_bytes(b"file-tok\r\n")
    path.chmod(0o600)
    cmd, env = _helper(fake, "true", token_args=["--token-file", str(path)])
    assert subprocess.run(cmd, capture_output=True, env=env, timeout=30).returncode == 0
    assert fake.auth == ["Bearer file-tok"]


# ---------------------------------------------------------------------------
# Stream failures are 255 with one line, never a command's answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv0, message", [
    ("@error", "exec_limit"),
    ("@not-json", "unreadable control frame"),
    ("@bare-exit", None),
    ("@drop", "closed before an exit status"),
])
def test_protocol_failures_exit_255(fake, argv0, message):
    cmd, env = _helper(fake, argv0)
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
    assert proc.returncode == 255
    assert time.monotonic() - started < 4  # an error frame must not wait for the server to hang up
    if message:
        assert message in proc.stderr.decode()
        assert "Traceback" not in proc.stderr.decode()


def test_unreachable_api_exits_255_with_one_line(fake):
    dead = socket.create_server(("127.0.0.1", 0))
    port = dead.getsockname()[1]
    dead.close()
    cmd = [sys.executable, str(HELPER), "--api", f"http://127.0.0.1:{port}", "--sprite", "s1",
           "--token-env", "T", "--", "true"]
    proc = subprocess.run(cmd, capture_output=True, env={**os.environ, "T": "tok"}, timeout=90)
    assert proc.returncode == 255
    assert proc.stderr.decode().startswith("sprite_stream: could not open the exec socket")
    assert proc.stderr.count(b"\n") == 1


def test_missing_token_exits_255_without_connecting(fake):
    cmd, env = _helper(fake, "true", token="")
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
    assert proc.returncode == 255 and b"no usable Sprites token" in proc.stderr
    assert fake.auth == []


# ---------------------------------------------------------------------------
# Session cleanup (a Sprites session outlives its socket and keeps the Sprite awake)
# ---------------------------------------------------------------------------

def _kill_paths(fake):
    return [(path, body["signal"]) for path, body, _ in fake.kills]


@pytest.mark.parametrize("how", ["kill", "terminate"])
def test_an_ended_stream_has_its_session_killed(fake, how):
    """Hermes ends streams with Popen.kill(); the remote command must not outlive it."""
    cmd, env = _helper(fake, "sleep", "60")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    _wait_for(lambda: "sess-1" in fake.procs)
    time.sleep(0.3)  # session_info delivered to the helper
    getattr(proc, how)()
    proc.wait(timeout=10)
    _wait_for(lambda: fake.kills)
    assert _kill_paths(fake)[0] == ("/v1/sprites/s1/exec/sess-1/kill", "TERM")
    assert fake.kills[0][2] == "Bearer tok-1"
    _wait_for(lambda: fake.procs["sess-1"].poll() is not None)


def test_a_gone_stdout_reader_kills_the_session(fake):
    cmd, env = _helper(fake, "yes")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    proc.stdout.read(4096)
    proc.stdout.close()
    assert proc.wait(timeout=30) == 141
    _wait_for(lambda: fake.kills)
    assert _kill_paths(fake)[0] == ("/v1/sprites/s1/exec/sess-1/kill", "TERM")


def test_a_command_that_ignores_term_gets_kill(fake):
    cmd, env = _helper(fake, sys.executable, "-c",
                       "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    _wait_for(lambda: "sess-1" in fake.procs)
    time.sleep(0.5)
    proc.kill()
    _wait_for(lambda: len(fake.kills) >= 2, timeout=30)
    assert [s for _, s in _kill_paths(fake)] == ["TERM", "KILL"]
    _wait_for(lambda: fake.procs["sess-1"].poll() is not None)


def test_a_finished_command_leaves_no_kill(fake):
    cmd, env = _helper(fake, "true")
    assert subprocess.run(cmd, capture_output=True, env=env, timeout=30).returncode == 0
    time.sleep(1.0)
    assert fake.kills == []


def test_reaper_contract(fake):
    """The reaper kills only a session it was told about and not told finished."""
    def reaper(lines):
        cmd = [sys.executable, str(HELPER), "--reap", "--api", fake.api, "--sprite", "s1",
               "--token-env", "T", "--"]
        r = subprocess.Popen(cmd, stdin=subprocess.PIPE, env={**os.environ, "T": "tok-2"})
        r.stdin.write(lines)
        r.stdin.close()
        assert r.wait(timeout=60) == 0

    reaper(b"")                              # no session yet: nothing to kill
    reaper(b"sid:sess-9\nfinished\n")        # finished: left alone
    assert fake.kills == []
    reaper(b"sid:sess-9\n")                  # died mid-command (session unknown to the fake: 404)
    assert fake.kills == [("/v1/sprites/s1/exec/sess-9/kill", {"signal": "TERM"}, "Bearer tok-2")]
