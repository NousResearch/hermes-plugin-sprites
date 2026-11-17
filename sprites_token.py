"""Read the Sprites token from ``SPRITES_TOKEN_FILE``: one implementation for the backend and the
stream helper.

Standard library only: ``sprite_stream.py`` imports it in a subprocess that has no Hermes imports.
"""
from __future__ import annotations

import os
import stat

#: Names a file holding the Sprites token (Hermes Cloud writes a short-lived,
#: per-instance token there and rewrites it on renewal). Not a secret itself.
SPRITES_TOKEN_FILE_ENV = "SPRITES_TOKEN_FILE"

#: Largest token file read. A token is a few hundred bytes; anything bigger
#: is not a token file, and the bound keeps a stray large file out of memory.
_TOKEN_FILE_MAX_BYTES = 8192

_TOKEN_NOT_FROM_ENV = (
    f"SPRITES_TOKEN is not used while {SPRITES_TOKEN_FILE_ENV} is set."
)


class SpritesTokenError(ValueError):
    """No usable Sprites token. The message never contains the token."""


def _read_token_file(path: str) -> str:
    """Read the token from ``path``; raise ``SpritesTokenError`` if unusable.

    The file holds the token alone, optionally followed by one newline. Only
    a regular file is read (opened non-blocking, so a FIFO cannot stall the
    caller), at most ``_TOKEN_FILE_MAX_BYTES``. A token must be printable
    ASCII with no whitespace, so it can never break the HTTP header it is
    sent in. Error messages name the path, never the contents.
    """
    def fail(problem: str) -> SpritesTokenError:
        return SpritesTokenError(
            f"{SPRITES_TOKEN_FILE_ENV} is set to {path!r} but {problem}. "
            + _TOKEN_NOT_FROM_ENV
        )

    flags = (os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))
    try:
        fd = os.open(path, flags)
    except (OSError, ValueError) as e:
        raise fail(f"the file could not be opened ({type(e).__name__})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise fail("it is not a regular file")
        chunks, size = [], 0
        while size <= _TOKEN_FILE_MAX_BYTES:
            chunk = os.read(fd, _TOKEN_FILE_MAX_BYTES + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    except OSError as e:
        raise fail(f"the file could not be read ({type(e).__name__})") from None
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > _TOKEN_FILE_MAX_BYTES:
        raise fail(f"the file is larger than {_TOKEN_FILE_MAX_BYTES} bytes")
    if data.endswith(b"\r\n"):
        data = data[:-2]
    elif data.endswith(b"\n"):
        data = data[:-1]
    if not data:
        raise fail("the file is empty")
    if any(b < 0x21 or b > 0x7E for b in data):
        raise fail(
            "the file holds whitespace, control or non-ASCII characters "
            "(it must hold the token alone, optionally followed by one newline)"
        )
    return data.decode("ascii")


class _TokenSource:
    """Where an environment gets its token: a token file, or a fixed env value.

    Decided once, in the caller's secret scope, when the environment is
    created. Later reads (after a 401, possibly on a worker thread that has
    no secret scope) re-read the same file and never look up variables again,
    so they cannot switch to a different credential. ``token_dir`` is the
    token file's directory, which file sync never uploads from (``None`` for
    an env token). ``repr`` shows the path only.
    """

    __slots__ = ("_env_token", "path", "token_dir")

    def __init__(self, path: "str | None" = None, env_token: "str | None" = None):
        self.path = path
        self._env_token = env_token
        self.token_dir = os.path.dirname(path) if path is not None else None

    def read(self) -> str:
        if self.path is None:
            return self._env_token
        _check_token_path(self.path)
        return _read_token_file(self.path)

    def __repr__(self) -> str:
        return f"_TokenSource(path={self.path!r})"


def _check_token_path(path: str) -> None:
    """Refuse a ``SPRITES_TOKEN_FILE`` that is not absolute or crosses a symlink.

    Every component is checked with ``lstat``, the file included, so the
    configured path is the path the OS opens and its directory is the one
    file sync leaves out. ``.``/``..`` components and repeated separators are
    refused for the same reason. Checked when the environment is created and
    again before every read.
    """
    def refuse(problem: str) -> SpritesTokenError:
        return SpritesTokenError(
            f"{SPRITES_TOKEN_FILE_ENV} is set to {path!r}, which {problem}. It "
            "must be an absolute path with no symlinks in it, naming a file in "
            "a directory that holds only the token. " + _TOKEN_NOT_FROM_ENV
        )

    if not os.path.isabs(path):
        raise refuse("is not an absolute path")
    if os.path.normpath(path) != path or path.startswith(os.sep * 2):
        raise refuse("contains '.', '..', a trailing separator or repeated separators")
    current = os.sep
    for name in path.split(os.sep)[1:]:
        current = os.path.join(current, name)
        try:
            mode = os.lstat(current).st_mode
        except (OSError, ValueError) as e:
            raise refuse(f"could not be checked at {current!r} ({type(e).__name__})") from None
        if stat.S_ISLNK(mode):
            raise refuse(f"has a symlink at {current!r}")
