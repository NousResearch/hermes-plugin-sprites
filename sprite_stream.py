"""Run one command inside a Sprite with this process's stdin/stdout as its stdin/stdout.

``SpritesEnvironment.exec_prefix()`` returns ``[python, this file, <options>, "--"]``. Hermes appends
``bash -c <script>`` and spawns the result whenever it needs a long-lived byte stream inside the
sandbox: the Bot Screen's RFB relay, cua-driver's MCP server over stdio, forwarded TCP connections
(the bot's Chromium DevTools port), and short probes. Bytes pass through unchanged in both
directions; the process exits with the remote command's exit status.

Transport: the Sprites exec WebSocket without a TTY
(``wss://<api>/v1/sprites/<name>/exec?cmd=...&stdin=true``). Every binary frame starts with a
stream byte: 0 stdin and 4 end of stdin (sent), 1 stdout, 2 stderr and 3 + one status byte
(received). The first text frame is ``session_info`` with the session id.

A Sprites exec session keeps running when its socket closes, and a running session keeps the
Sprite awake. Hermes ends these streams with SIGKILL (``Popen.kill``), which no handler can catch,
so a reaper process started in its own session holds a pipe from this one: unless this process
tells it the command finished, it kills the session when the pipe closes.

    sprite_stream.py --api URL --sprite NAME (--token-file PATH | --token-env VAR)
                     [--user USER] [--env K=V]... [--no-stdin] -- ARGV...
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from urllib.parse import quote, urlencode

STDIN, STDOUT, STDERR, EXIT, STDIN_EOF = 0, 1, 2, 3, 4
_FINISHED = b"finished"
_MAX_TOKEN_BYTES = 8192


def _read_token(args) -> str:
    if args.token_file:
        fd = os.open(args.token_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            raw = os.read(fd, _MAX_TOKEN_BYTES + 1)
        finally:
            os.close(fd)
        token = raw.decode("ascii", "replace")
        if token.endswith("\n"):
            token = token[:-1]
    else:
        token = os.environ.get(args.token_env, "")
    if not token or len(token) > _MAX_TOKEN_BYTES or any(not 0x21 <= ord(c) <= 0x7E for c in token):
        source = args.token_file or f"${args.token_env}"
        raise SystemExit(f"sprite_stream: no usable Sprites token in {source}")
    return token


def _sprite_url(api: str, sprite: str) -> str:
    return f"{api.rstrip('/')}/v1/sprites/{quote(sprite, safe='')}"


def _exec_url(args) -> str:
    argv = list(args.argv)
    if args.user:
        # -H: $HOME of that user. The exec API's own `user` parameter leaves HOME to the image.
        argv = ["sudo", "-n", "-H", "-u", args.user, "--", *argv]
    params = [("cmd", a) for a in argv] + [("path", argv[0]), ("stdin", "true")]
    params += [("env", kv) for kv in args.env]
    base = _sprite_url(args.api, args.sprite)
    scheme, rest = base.split("://", 1)
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/exec?{urlencode(params)}"


def _start_reaper(args, session_id: str) -> subprocess.Popen:
    cmd = [sys.executable, os.path.abspath(__file__), "--reap", session_id, "--api", args.api,
           "--sprite", args.sprite]
    cmd += ["--token-file", args.token_file] if args.token_file else ["--token-env", args.token_env]
    return subprocess.Popen(cmd + ["--"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)


def _reap(args) -> int:
    """Wait for the stream process; kill its session unless it reported the command finished."""
    told = b""
    while True:
        chunk = os.read(0, 64)
        if not chunk:
            break
        told += chunk
    if told == _FINISHED:
        return 0
    import httpx
    url = f"{_sprite_url(args.api, args.sprite)}/exec/{quote(args.reap, safe='')}/kill"
    for _ in range(3):
        try:
            resp = httpx.post(url, json={"signal": "TERM"}, timeout=30,
                              headers={"Authorization": f"Bearer {_read_token(args)}"})
            if resp.status_code in (200, 404):
                return 0
        except httpx.HTTPError:
            pass
    return 1


def _pump_stdin(ws, interactive: bool) -> None:
    try:
        if interactive:
            while True:
                data = os.read(0, 65536)
                if not data:
                    break
                ws.send(bytes([STDIN]) + data)
        ws.send(bytes([STDIN_EOF]))
    except Exception:  # the socket closed under us: the main loop reports it
        pass


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _run(args) -> int:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    token = _read_token(args)
    ws = connect(_exec_url(args), additional_headers={"Authorization": f"Bearer {token}"},
                 max_size=None, ping_interval=20, ping_timeout=60, close_timeout=1)
    reaper = None
    status = 255
    try:
        first = ws.recv()
        if isinstance(first, str):
            info = json.loads(first)
            if info.get("session_id"):
                reaper = _start_reaper(args, str(info["session_id"]))
        else:  # no session_info: treat it as data
            status = _handle_frame(first)
            if status is not None:
                return status
        threading.Thread(target=_pump_stdin, args=(ws, not args.no_stdin), daemon=True).start()
        while True:
            try:
                frame = ws.recv()
            except ConnectionClosed:
                return 255  # closed before an exit status; the reaper kills what is left
            if isinstance(frame, str):
                msg = json.loads(frame)
                if msg.get("type") == "exit":
                    status = int(msg.get("exit_code", 0))
                    break
                continue
            code = _handle_frame(frame)
            if code is not None:
                status = code
                break
        if reaper is not None and reaper.stdin is not None:
            reaper.stdin.write(_FINISHED)
        return status
    finally:
        if reaper is not None and reaper.stdin is not None:
            try:
                reaper.stdin.close()
            except OSError:
                pass
        try:
            ws.close()
        except Exception:
            pass


def _handle_frame(frame: bytes):
    if not frame:
        return None
    kind, payload = frame[0], frame[1:]
    if kind == STDOUT:
        _write_all(1, payload)
    elif kind == STDERR:
        _write_all(2, payload)
    elif kind == EXIT:
        return payload[0] if payload else 0
    return None


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
    p.add_argument("--reap")
    p.add_argument("argv", nargs="*")
    args = p.parse_args(argv)
    if args.reap:
        return _reap(args)
    if not args.argv:
        p.error("no command after --")
    # SIGTERM / SIGHUP end the stream like EOF would not: exit now and let the reaper kill the session.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: os._exit(143))
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
