"""Run one command inside a Sprite with this process's stdin/stdout as its stdin/stdout.

``SpritesEnvironment.exec_prefix()`` returns ``[python, this file, <options>, "--"]``. Hermes appends
``bash -c <script>`` and spawns the result whenever it needs a long-lived byte stream inside the
sandbox: the Bot Screen's RFB relay, cua-driver's MCP server over stdio, forwarded TCP connections
(the bot's Chromium DevTools port), and short probes. Bytes pass through unchanged in both
directions.

Exit status: the remote command's, or 255 when the stream itself failed (no token, connection or
handshake refused, an error frame, the socket closed before an exit status), as ``ssh`` does. The
reason goes to stderr in one line.

Transport: the Sprites exec WebSocket without a TTY
(``wss://<api>/v1/sprites/<name>/exec?cmd=...&stdin=true``). Every binary frame starts with a
stream byte: 0 stdin and 4 end of stdin (sent), 1 stdout, 2 stderr and 3 + one status byte
(received). The first text frame is ``session_info`` with the session id. A command too long for
a URL (Bot Desktop's setup script embeds its launcher and wallpaper) is sent as the first stdin
bytes to a short ``python3`` bootstrap in the Sprite, which reads exactly that many bytes and
execs it.

A Sprites exec session keeps running when its socket closes, and a running session keeps the
Sprite awake. Hermes ends these streams with SIGKILL (``Popen.kill``), which no handler can catch,
so a reaper process, started in its own session BEFORE connecting, holds a pipe from this one. It
is told the session id as soon as it is known and whether the command finished; when the pipe
closes on a session that did not finish, it kills it (TERM, then KILL).

    sprite_stream.py --api URL --sprite NAME (--token-file PATH | --token-env VAR)
                     [--user USER] [--env K=V]... [--no-stdin] -- ARGV...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from urllib.parse import quote, urlencode

STDIN, STDOUT, STDERR, EXIT, STDIN_EOF = 0, 1, 2, 3, 4
TRANSPORT_FAILED = 255
#: Longest encoded query (argv and env pairs, after percent-escaping) sent in the URL; anything longer
#: goes through the stdin bootstrap. Request lines past 8 KiB are refused by common front ends.
MAX_QUERY_BYTES = 6144
_FINISHED = b"finished\n"
# Python starts with SIGPIPE and SIGXFSZ ignored and an ignore survives exec, so both go back to the
# default first: the command must see the signal state it would get from a direct exec.
_BOOTSTRAP = (
    "import json,os,signal,sys\n"
    "for s in (signal.SIGPIPE,signal.SIGXFSZ): signal.signal(s,signal.SIG_DFL)\n"
    "n=int(sys.argv[1]);b=b''\n"
    "while len(b)<n:\n"
    "  c=os.read(0,n-len(b))\n"
    "  if not c: sys.exit(255)\n"
    "  b+=c\n"
    "a=json.loads(b)\n"
    "try: os.execvp(a[0],a)\n"
    "except OSError as e:\n"
    "  sys.stderr.write(f'{a[0]}: {e.strerror}\\n'); sys.exit(127)\n"
)


class StreamError(Exception):
    """The stream failed (not the command): reported in one line, exit 255."""


def _token(args) -> str:
    if args.token_file:
        from sprites_token import SpritesTokenError, _TokenSource
        try:
            return _TokenSource(path=args.token_file).read()
        except SpritesTokenError as e:
            raise StreamError(str(e)) from None
    token = os.environ.get(args.token_env, "")
    if not token or any(not 0x21 <= ord(c) <= 0x7E for c in token):
        raise StreamError(f"${args.token_env} holds no usable Sprites token")
    return token


def _sprite_url(api: str, sprite: str) -> str:
    return f"{api.rstrip('/')}/v1/sprites/{quote(sprite, safe='')}"


def remote_argv(args) -> list[str]:
    argv = list(args.argv)
    if args.user:
        # -H: $HOME of that user. The exec API's own `user` parameter leaves HOME to the image.
        argv = ["sudo", "-n", "-H", "-u", args.user, "--", *argv]
    return argv


def plan(args) -> tuple[str, bytes]:
    """``(exec URL, bytes to send first on stdin)``: the argv itself, or a bootstrap fed the argv."""
    def query(argv: list[str]) -> str:
        params = [("cmd", a) for a in argv] + [("path", argv[0]), ("stdin", "true")]
        return urlencode(params + [("env", kv) for kv in args.env])

    argv = remote_argv(args)
    first = b""
    try:
        q = query(argv)
    except UnicodeEncodeError:  # an undecodable name (surrogate escapes): JSON carries it, a URL cannot
        q = ""
    if not q or len(q) > MAX_QUERY_BYTES:
        first = json.dumps(argv).encode()
        q = query(["python3", "-c", _BOOTSTRAP, str(len(first))])
    base = _sprite_url(args.api, args.sprite)
    scheme, rest = base.split("://", 1)
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/exec?{q}", first


# ── reaper ──────────────────────────────────────────────────────────────────────────────────────────

def _start_reaper(args) -> subprocess.Popen:
    cmd = [sys.executable, os.path.abspath(__file__), "--reap", "--api", args.api, "--sprite", args.sprite]
    cmd += ["--token-file", args.token_file] if args.token_file else ["--token-env", args.token_env]
    return subprocess.Popen(cmd + ["--"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)


def _tell(reaper: subprocess.Popen, line: bytes) -> None:
    pipe = reaper.stdin
    assert pipe is not None  # started with stdin=PIPE
    try:
        pipe.write(line)
        pipe.flush()
    except (OSError, ValueError):
        pass  # a dead reaper cannot be helped here; the stream itself goes on


def _kill(args, session: str, signal_name: str) -> bool:
    """POST the kill; True when it was accepted, False when the session was already gone or every try
    failed (backing off 1, 2, 4, 8 s)."""
    import httpx
    url = f"{_sprite_url(args.api, args.sprite)}/exec/{quote(session, safe='')}/kill"
    for delay in (0, 1, 2, 4, 8):
        time.sleep(delay)
        try:
            resp = httpx.post(url, json={"signal": signal_name}, timeout=30,
                              headers={"Authorization": f"Bearer {_token(args)}"})
        except (httpx.HTTPError, StreamError):
            continue
        if resp.status_code in (200, 404):  # 404: the session is already gone
            return resp.status_code == 200
    return False


def reap(args) -> int:
    """Wait for the stream process; kill its session unless it reported the command finished."""
    session, finished = None, False
    for line in sys.stdin.buffer:
        if line.startswith(b"sid:"):
            session = line[4:].strip().decode("ascii", "replace")
        elif line == _FINISHED:
            finished = True
    if session is None or finished:
        return 0
    if _kill(args, session, "TERM"):
        time.sleep(5)
        _kill(args, session, "KILL")  # 404 when TERM already ended it
    return 0


# ── stream ──────────────────────────────────────────────────────────────────────────────────────────

def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _pump_stdin(ws, first: bytes, interactive: bool) -> None:
    try:
        if first:
            ws.send(bytes([STDIN]) + first)
        if interactive:
            while True:
                data = os.read(0, 65536)
                if not data:
                    break
                ws.send(bytes([STDIN]) + data)
        ws.send(bytes([STDIN_EOF]))
    except Exception:  # the socket closed under us: the receive loop reports it
        pass


def _text(frame: str) -> dict:
    try:
        msg = json.loads(frame)
    except ValueError:
        raise StreamError(f"unreadable control frame from Sprites: {frame[:200]!r}") from None
    if not isinstance(msg, dict):
        raise StreamError(f"unexpected control frame from Sprites: {frame[:200]!r}")
    if msg.get("type") == "error" or msg.get("error"):
        err = msg.get("error")
        detail = json.dumps(err) if isinstance(err, (dict, list)) else str(err or msg)
        raise StreamError(f"Sprites exec error: {detail[:500]}")
    return msg


def _receive(ws) -> int:
    from websockets.exceptions import ConnectionClosed
    while True:
        try:
            frame = ws.recv()
        except ConnectionClosed as e:
            raise StreamError(f"the exec socket closed before an exit status ({e})") from None
        if isinstance(frame, str):
            msg = _text(frame)
            if msg.get("type") == "exit":
                return int(msg.get("exit_code", TRANSPORT_FAILED))
            continue
        if not frame:
            continue
        kind, payload = frame[0], frame[1:]
        if kind == STDOUT:
            _write_all(1, payload)
        elif kind == STDERR:
            _write_all(2, payload)
        elif kind == EXIT:
            return payload[0] if payload else TRANSPORT_FAILED


def stream(args, reaper: subprocess.Popen) -> int:
    from websockets.sync.client import connect
    try:
        url, first = plan(args)
        ws = connect(url, additional_headers={"Authorization": f"Bearer {_token(args)}"},
                     max_size=None, open_timeout=60, ping_interval=20, ping_timeout=60, close_timeout=1)
    except StreamError:
        raise
    except Exception as e:  # refused, DNS, TLS, handshake status (401, 404, 429)
        raise StreamError(f"could not open the exec socket for {args.sprite}: {type(e).__name__}: {e}") from None
    try:
        info = ws.recv()
        if not isinstance(info, str):
            raise StreamError("Sprites sent data before session_info")
        session = _text(info).get("session_id")
        if not session:
            raise StreamError(f"session_info without a session id: {info[:200]!r}")
        _tell(reaper, f"sid:{session}\n".encode())
        threading.Thread(target=_pump_stdin, args=(ws, first, not args.no_stdin), daemon=True).start()
        status = _receive(ws)
        _tell(reaper, _FINISHED)
        return status
    except (StreamError, BrokenPipeError):
        # BrokenPipeError: our stdout reader is gone; main() exits without telling the reaper, which kills.
        raise
    except Exception as e:
        raise StreamError(f"{type(e).__name__}: {e}") from None
    finally:
        try:
            ws.close()
        except Exception:
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sprite_stream.py")
    p.add_argument("--api", required=True)
    p.add_argument("--sprite", required=True)
    tok = p.add_mutually_exclusive_group(required=True)
    tok.add_argument("--token-file")
    tok.add_argument("--token-env")
    p.add_argument("--user")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--no-stdin", action="store_true")
    p.add_argument("--reap", action="store_true")
    p.add_argument("argv", nargs="*")
    args = p.parse_args(argv)
    if args.reap:
        return reap(args)
    if not args.argv:
        p.error("no command after --")
    # Started before connecting, so a kill at any point after the session exists is covered.
    # SIGTERM/SIGHUP keep their default action: the process dies, the pipe closes, the reaper kills.
    reaper = _start_reaper(args)
    try:
        return stream(args, reaper)
    except StreamError as e:
        try:
            sys.stderr.write(f"sprite_stream: {e}\n")
            sys.stderr.flush()
        except OSError:
            sys.stderr = open(os.devnull, "w")  # else interpreter shutdown fails on the same pipe (exit 120)
        return TRANSPORT_FAILED
    except BrokenPipeError:
        return 141
    finally:
        try:
            if reaper.stdin is not None:
                reaper.stdin.close()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
