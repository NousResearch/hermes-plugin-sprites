"""Sprites execution environment (standalone Hermes plugin).

Originally authored by Kyle McLaren (@kylemclaren, Fly.io) as hermes-agent
PR #30112, hardened in salvage PR #93523. Extracted to this standalone
plugin repo when terminal backends became pluggable (hermes-agent PR #94400).

Uses the sprites-py SDK (https://github.com/superfly/sprites-py) to run
commands in Sprites — stateful cloud sandboxes on Fly.io, with
checkpoint & restore. Persistent by default: each Sprite outlives the session
and is reused via a deterministic, profile-scoped name — ``hermes-{task_id}``
on the default profile, ``hermes-{display}-{digest}`` on named profiles (see
``_resolve_sprite_name`` for the full contract). Cleanup leaves the Sprite
running when ``persistent_filesystem`` is True; the Sprite is deleted otherwise.

Two optional plugin settings (``plugins.entries.sprites.settings.*``) change
the naming: ``name_prefix`` replaces the leading ``hermes`` of every name so
one deployment's Sprites are confined to ``<prefix>-*``, and ``scope:
instance`` makes every task id share one persistent Sprite per profile (see
``resolve_naming_settings``).

Four more optional settings (``image``, ``max_cpus``, ``max_ram_gb``,
``max_disk_gb``) choose the image and sizing of Sprites this plugin creates
(see ``resolve_create_settings``). With none set, creation sends exactly the
request it always has.

Sprites 2.0 runtime: the only filesystem endpoint is ``fs/write`` (it creates
parent directories itself); there is no read, list, stat or delete endpoint.
File sync therefore uploads with ``fs/write`` and unpacks or deletes through
exec. Each ``fs/write`` and each exec counts against a per-Sprite request rate
limit, so changed files travel as tar archives: one ``fs/write`` and
one exec per archive, however many files it holds.

Token: ``SPRITES_TOKEN_FILE``, when set, names a file holding the token. The
file is read again for every new environment and after an HTTP 401, so a
token rewritten in place (Hermes Cloud renews it weekly) is used without a
restart. Unset, ``SPRITES_TOKEN`` / ``SPRITE_TOKEN`` are read as before (see
``resolve_sprites_token``).
"""

import hashlib
import io
import logging
import os
import posixpath
import re
import shlex
import stat
import tarfile
import threading
import uuid
from pathlib import Path
from urllib.parse import quote as _url_quote

from tools.environments.base import BaseEnvironment

try:
    from tools.environments.base_output import _ThreadedProcessHandle
except ModuleNotFoundError as exc:
    # Hermes releases before the September 2026 module split keep it in base.
    # Do not hide a missing dependency inside an otherwise available module.
    if exc.name != "tools.environments.base_output":
        raise
    from tools.environments.base import _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
)

logger = logging.getLogger(__name__)

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


class SpritesApiError(RuntimeError):
    """A Sprites API error whose original text held the token (withheld)."""


def _default_lookup(name: str):
    from agent.secret_scope import get_secret

    return get_secret(name)


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


def _token_source(lookup=None) -> _TokenSource:
    """Decide where the token comes from; see ``resolve_sprites_token``."""
    lookup = lookup or _default_lookup
    raw = lookup(SPRITES_TOKEN_FILE_ENV)
    if raw is not None:
        path = raw.strip()
        if not path:
            raise SpritesTokenError(
                f"{SPRITES_TOKEN_FILE_ENV} is set but empty. " + _TOKEN_NOT_FROM_ENV
            )
        _check_token_path(path)
        return _TokenSource(path=path)
    token = lookup("SPRITES_TOKEN") or lookup("SPRITE_TOKEN")
    if not token:
        raise SpritesTokenError(
            "Sprites backend requires SPRITES_TOKEN (or SPRITES_TOKEN_FILE). "
            "Run `hermes setup terminal` or set SPRITES_TOKEN in .env."
        )
    if any(ord(c) < 0x20 or ord(c) > 0x7E for c in token):
        # Such a value can never be sent as an HTTP header; refusing it here
        # keeps it out of the HTTP library's error text.
        raise SpritesTokenError(
            "SPRITES_TOKEN holds control or non-ASCII characters, so it cannot "
            "be a Sprites token."
        )
    return _TokenSource(env_token=token)


def resolve_sprites_token(lookup=None) -> str:
    """Return the Sprites token, read fresh on every call.

    When ``SPRITES_TOKEN_FILE`` is set (to anything, even an empty value),
    the token comes from that file only: a missing, unreadable, non-regular,
    oversized, empty or malformed file is an error, and ``SPRITES_TOKEN`` is
    deliberately NOT used as a fallback, since a leftover env token may be a
    broader legacy token than the scoped one the file holds. The path must
    be absolute with no symlinks in it (see ``_check_token_path``). The file holds
    the token alone, optionally followed by one newline. Whoever writes the
    file must replace it atomically (write a temp file, then rename), so a
    read never sees a half-written token.

    When ``SPRITES_TOKEN_FILE`` is unset, ``SPRITES_TOKEN`` or
    ``SPRITE_TOKEN`` is used, as before.

    ``lookup(name)`` reads a variable (default: ``agent.secret_scope.get_secret``).
    Raises ``SpritesTokenError`` when no token is found.
    """
    return _token_source(lookup).read()


#: Host paths already reported as skipped by ``_without_token_dir``.
_skip_warned: set = set()
_skip_warned_lock = threading.Lock()


def _without_token_dir(files, token_dir: "str | None"):
    """``files`` (``(host_path, remote_path)`` pairs) minus the token directory.

    A file whose realpath is inside ``token_dir`` (the directory of
    ``SPRITES_TOKEN_FILE``, symlink-free by ``_check_token_path``) is never
    uploaded: that covers the current token, a renewed one and the writer's
    temp files. Each skipped path is logged once as a warning (the path,
    never the contents), so a file wrongly kept there is not hidden silently.
    """
    files = list(files)
    if token_dir is None:
        return files
    kept = []
    for host, remote in files:
        try:
            inside = _is_within(os.path.realpath(host), token_dir)
        except (OSError, ValueError):
            inside = True  # unresolvable: do not upload it
        if not inside:
            kept.append((host, remote))
            continue
        with _skip_warned_lock:
            first = host not in _skip_warned
            _skip_warned.add(host)
        if first:
            logger.warning(
                "Sprites: file sync skipped %s: it is inside %s, the %s "
                "directory, which must hold only the token",
                host, token_dir, SPRITES_TOKEN_FILE_ENV,
            )
    return kept


def _is_within(path: str, directory: str) -> bool:
    """True when ``path`` is ``directory`` or inside it (both absolute)."""
    return path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)


#: Set on the calling thread when an HTTP response with status 401 arrives
#: (an httpx response hook on every client this plugin makes). sprites-py
#: turns a filesystem 401 into a ``FilesystemError`` that keeps no status.
_http_auth = threading.local()


def _note_http_status(response) -> None:
    if getattr(response, "status_code", None) == 401:
        _http_auth.saw_401 = True


def _is_auth_error(exc: "BaseException | None") -> bool:
    """True for a Sprites API 401, however the SDK surfaced it.

    sprites-py 0.5 raises ``AuthenticationError`` for control-plane calls, an
    ``APIError`` with ``status_code``, or (exec) a ``NetworkError`` caused by
    websockets' ``InvalidStatus``, whose status is ``response.status_code``.
    """
    try:
        from sprites.exceptions import AuthenticationError
    except ImportError:
        AuthenticationError = ()
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if AuthenticationError and isinstance(exc, AuthenticationError):
            return True
        if getattr(exc, "status_code", None) == 401:
            return True
        if getattr(getattr(exc, "response", None), "status_code", None) == 401:
            return True
        # Follow the chain a traceback would show (``raise ... from None`` stops it).
        exc = exc.__cause__ if exc.__cause__ is not None else (
            None if exc.__suppress_context__ else exc.__context__
        )
    return False


def _mentions(exc: BaseException, secrets) -> bool:
    """True when ``exc``'s text (or its chain's) contains any of ``secrets``."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        try:
            parts = [str(exc), repr(exc), *map(str, getattr(exc, "__notes__", ()) or ())]
            for attr in ("stdout", "stderr"):
                value = getattr(exc, attr, None)
                if isinstance(value, (bytes, bytearray)):
                    parts.append(bytes(value).decode("utf-8", errors="replace"))
                elif isinstance(value, str):
                    parts.append(value)
            text = "\n".join(parts)
        except Exception:
            return True  # unprintable: assume the worst
        if any(s and s in text for s in secrets):
            return True
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False

#: Hard bound on generated Sprite names. The name feeds the
#: ``{name}-random.sprites.app`` hostname, so it must fit a DNS label.
_MAX_NAME_LEN = 63

#: Hex chars of identity digest carried in non-legacy names (48 bits — a
#: collision needs ~2^24 coexisting identities, versus 2^12 at the previous
#: 6 chars, where a collision between valid profile names was demonstrated).
_DIGEST_LEN = 12

#: Leading component of every name when no ``name_prefix`` is configured.
_DEFAULT_HEAD = "hermes"

#: Longest accepted ``name_prefix``. Leaves at least 22 characters for the
#: ``-{display}-{digest}`` tail, so even the bounded form keeps its full
#: 12-char digest plus some readable display text.
_MAX_PREFIX_LEN = 40

#: A ``name_prefix`` must already be a clean DNS-label fragment: lowercase
#: alphanumerics separated by single hyphens. It is never slugified — a prefix
#: is an authority boundary, so an invalid one is an error, not a suggestion.
_PREFIX_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

#: Accepted ``scope`` values. ``session`` keeps one Sprite per task id;
#: ``instance`` shares one persistent Sprite across every task id.
_SCOPES = frozenset({"session", "instance"})

#: Task id every persistent environment resolves under in ``instance`` scope.
_INSTANCE_TASK_ID = "default"

#: Create-time limits from the Sprites 2.0 API docs (``config`` on create).
_MAX_CPUS_RANGE = (1, 128)
_MAX_GB_RANGE = (1, 1024)

#: Client wait for a create that sends ``config``. Pulling an image the
#: platform has not cached yet takes a while (first pulls of a ~GB image
#: measured 64-73 s; cached, under 5 s), past the SDK's 120 s create timeout.
#: Sprites 2.0 sends nothing until the Sprite is running, so this read
#: timeout bounds the whole wait. Budget: Hermes's terminal tool deadline is
#: 420 s by default; initial GET (well under a second) + 240 s create stays
#: under it, so a slow pull ends in a retryable error, not a tool timeout.
_CREATE_WITH_CONFIG_TIMEOUT = 240.0


class SpriteStillCreating(TimeoutError):
    """The configured create outlived its wait; the platform keeps going."""

#: Client wait for the exec that deletes synced files.
_DELETE_TIMEOUT = 60.0

#: Client wait for the exec that unpacks a sync archive.
_UNPACK_TIMEOUT = 120.0

#: Client wait for the exec that joins a chunked upload: ``_UNPACK_TIMEOUT``
#: (what one archive of up to 4 MiB gets) plus one second per MiB joined.
#: The join reads and writes the file once (twice when it is retried
#: through ``sudo -n``), so this allows for a disk far slower than 1 MiB/s.
#: Capped at 600 s, the longest exec the Sprites 2 docs allow ("Exec":
#: ``timeout_ms`` "at most 600000 (10 minutes)").
_JOIN_TIMEOUT_PER_MIB = 1.0
_JOIN_TIMEOUT_MAX = 600.0


def _join_timeout(size: int) -> float:
    """Client wait for joining a chunked upload of ``size`` bytes."""
    mib = -(-size // (1024 * 1024))
    return min(_JOIN_TIMEOUT_MAX, _UNPACK_TIMEOUT + _JOIN_TIMEOUT_PER_MIB * mib)

#: Largest body one sync ``fs/write`` sends: a whole tar archive (headers,
#: padding and end-of-archive blocks included), or one file in the
#: one-by-one fallback. The Sprites 2 docs cap a write at 16 MiB ("Files":
#: "Files are at most 16 MiB"; "Limits": "Files: 16 MiB per fs/write"), and
#: a 17 MiB write gets 413 ``file_too_large``. Observed, not documented: a
#: 5 MiB write came back 502 with an empty body after 8 s, while 4.75 MiB
#: and below succeeded. 4 MiB stays under both. A file too large for an
#: archive of its own is sent in raw chunks of at most this size and joined
#: on the Sprite (see ``_assemble_script``).
_SYNC_WRITE_MAX_BYTES = 4 * 1024 * 1024

#: Upper bound on one delete command, measured URL-encoded (the command
#: travels in the exec URL), so a very large batch is split into several execs.
_DELETE_COMMAND_MAX = 6000

#: OCI reference pieces (distribution/reference grammar, simplified): a
#: repository path component, a tag, and an ``algorithm:hex`` digest.
_REPO_COMPONENT_RE = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*")
_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_DIGEST_RE = re.compile(r"[a-z0-9]+(?:[+._-][a-z0-9]+)*:[A-Fa-f0-9]{32,}")
_DIGEST_HEX_LEN = {"sha256": 64, "sha512": 128}
_REGISTRY_RE = re.compile(r"[A-Za-z0-9.-]+(?::[0-9]+)?|\[[0-9A-Fa-f:]+\](?::[0-9]+)?")


def _validate_image_ref(image) -> str:
    """Return ``image`` if it is a pinned OCI reference, else raise ``ValueError``.

    Sprites requires an explicit tag (not ``latest``) or a digest. The check
    follows the OCI reference grammar closely enough to reject malformed
    values (empty or non-hex digests, stray separators) before any API call.
    """
    if not isinstance(image, str):
        raise ValueError(
            "Sprites plugin setting image must be a string, got "
            f"{type(image).__name__} ({image!r})"
        )

    def _bad(reason: str) -> ValueError:
        return ValueError(f"Sprites plugin setting image {image!r} is invalid: {reason}")

    if any(c.isspace() for c in image):
        raise _bad("it must not contain whitespace")
    if image.count("@") > 1:
        raise _bad("more than one '@'")
    name, at, digest = image.partition("@")
    if at and not _DIGEST_RE.fullmatch(digest):
        raise _bad("a digest looks like name@sha256:<hex>")
    algo, _, hexpart = digest.partition(":")
    if at and algo in _DIGEST_HEX_LEN and len(hexpart) != _DIGEST_HEX_LEN[algo]:
        raise _bad(f"a {algo} digest has exactly {_DIGEST_HEX_LEN[algo]} hex characters")
    head, slash, last = name.rpartition("/")
    repo_last, colon, tag = last.partition(":")
    if colon and not _TAG_RE.fullmatch(tag):
        raise _bad(f"tag {tag!r} is not a valid tag")
    path = f"{head}{slash}{repo_last}"
    parts = path.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        if not _REGISTRY_RE.fullmatch(parts[0]):
            raise _bad(f"registry {parts[0]!r} is not valid")
        parts = parts[1:]
    if not parts or not all(_REPO_COMPONENT_RE.fullmatch(p) for p in parts):
        raise _bad("the repository name must be lowercase letters, digits and . _ - separators")
    if not at and not tag:
        raise _bad("it needs an explicit tag or digest (for example ubuntu:24.04 or name@sha256:<hex>)")
    if not at and tag == "latest":
        raise _bad("the latest tag is rejected by Sprites; pin a version tag or a digest")
    return image


def _validate_int_setting(key: str, value, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    # bool is an int subclass; ``true`` in YAML is not a size.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Sprites plugin setting {key} must be a whole number, got "
            f"{type(value).__name__} ({value!r})"
        )
    if not lo <= value <= hi:
        raise ValueError(
            f"Sprites plugin setting {key} {value} is out of range ({lo} to {hi})"
        )
    return value


def resolve_create_settings(image=None, max_cpus=None, max_ram_gb=None,
                            max_disk_gb=None) -> dict:
    """Validate the create-time plugin settings.

    Returns the Sprites ``config`` object to send on create, holding only the
    keys that are set (``None`` and ``""`` mean unset). An empty dict means
    "create exactly as before". An invalid value raises ``ValueError``: a
    typo must not silently create a Sprite with a different image or size.
    """
    config: dict = {}
    if image not in (None, ""):
        config["image_url"] = _validate_image_ref(image)
    for key, value, bounds in (
        ("max_cpus", max_cpus, _MAX_CPUS_RANGE),
        ("max_ram_gb", max_ram_gb, _MAX_GB_RANGE),
        ("max_disk_gb", max_disk_gb, _MAX_GB_RANGE),
    ):
        if value not in (None, ""):
            config[key] = _validate_int_setting(key, value, bounds)
    return config


def resolve_naming_settings(name_prefix=None, scope=None) -> tuple[str | None, str]:
    """Validate the raw ``name_prefix`` / ``scope`` plugin settings.

    Returns ``(prefix, scope)`` where ``prefix`` is ``None`` when unset.
    ``None`` and ``""`` mean "unset" for both keys. Anything else that is not
    a valid value raises ``ValueError`` — misconfiguration must fail closed,
    because silently ignoring a prefix would create Sprites outside the
    namespace the operator meant to confine this deployment to, and silently
    ignoring a scope typo would start creating one Sprite per session.
    """
    prefix = None
    if name_prefix not in (None, ""):
        if not isinstance(name_prefix, str):
            raise ValueError(
                "Sprites plugin setting name_prefix must be a string, got "
                f"{type(name_prefix).__name__} ({name_prefix!r})"
            )
        if len(name_prefix) > _MAX_PREFIX_LEN or not _PREFIX_RE.fullmatch(name_prefix):
            raise ValueError(
                f"Sprites plugin setting name_prefix {name_prefix!r} is invalid: "
                "use lowercase letters, digits and single hyphens (no leading, "
                f"trailing or doubled hyphens), at most {_MAX_PREFIX_LEN} characters"
            )
        prefix = name_prefix
    if scope in (None, ""):
        resolved_scope = "session"
    elif isinstance(scope, str) and scope in _SCOPES:
        resolved_scope = scope
    else:
        raise ValueError(
            f"Sprites plugin setting scope {scope!r} is invalid: expected one of "
            + ", ".join(sorted(_SCOPES))
        )
    return prefix, resolved_scope


def _require_within_prefix(name: str, prefix: str | None) -> None:
    """Refuse any Sprite name outside the configured prefix or the DNS bound.

    This is the last check before a name reaches the Sprites API (and is
    re-applied to the Sprite the API hands back), so a future naming bug
    cannot create or adopt a Sprite that belongs to another deployment.
    """
    if prefix is None:
        return
    if not isinstance(name, str) or not name.startswith(f"{prefix}-") or len(name) > _MAX_NAME_LEN:
        raise RuntimeError(
            f"Sprites backend refused Sprite name {name!r}: it is outside the "
            f"configured name_prefix {prefix!r} (or exceeds {_MAX_NAME_LEN} chars)"
        )


#: One sync lock per Sprite name, shared by every environment in this process
#: that targets that Sprite (see ``_SharedSpriteSyncManager``).
_SYNC_LOCKS: dict[str, threading.Lock] = {}
_SYNC_LOCKS_GUARD = threading.Lock()


def _shared_sync_lock(sprite_name: str) -> threading.Lock:
    with _SYNC_LOCKS_GUARD:
        return _SYNC_LOCKS.setdefault(sprite_name, threading.Lock())


class _SharedSpriteSyncManager(FileSyncManager):
    """File sync whose transactions are serialised per Sprite, not per environment.

    Several environments can resume one Sprite (``scope: instance`` does so
    for every session; ``delegate_task`` children always have). Each keeps
    its own sync history, but they all write the same remote paths. With
    only per-manager locks, one environment can upload an old copy of a file
    after another uploaded the newer one, and neither retries because each
    history says its own upload succeeded. Holding a lock shared by every
    manager of the same Sprite around the whole transaction makes each
    upload read the host file after the previous upload finished, so the
    last write is always the newest content. (Environments in other
    processes are not covered.)
    """

    def __init__(self, *args, shared_lock: threading.Lock, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_lock = shared_lock

    def sync(self, *, force: bool = False) -> None:
        with self._shared_lock:
            super().sync(force=force)


def _collapse_slug(value: str) -> str:
    """Plain DNS-safe collapse: lowercase, non-alnum runs become one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")


def _identity_digest(*parts: str) -> str:
    """Collision-resistant digest of an ordered tuple of raw strings.

    Each component is length-prefixed before hashing, so the encoding is
    unambiguous: ``("a\\x1fb", "c")`` and ``("a", "b\\x1fc")`` digest
    differently no matter what bytes the components contain.
    """
    h = hashlib.sha256()
    for part in parts:
        b = (part or "").encode("utf-8")
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
    return h.hexdigest()[:_DIGEST_LEN]


def _bounded_name(display: str, digest: str, head: str = _DEFAULT_HEAD) -> str:
    """Compose ``{head}-{display}-{digest}`` within ``_MAX_NAME_LEN``.

    The head (``hermes`` or the configured ``name_prefix``) and the digest
    are authoritative and never truncated; the human-readable display slug
    absorbs all the shortening.
    """
    budget = _MAX_NAME_LEN - len(head) - 1 - 1 - len(digest)
    display = (display or "")[:max(budget, 0)].strip("-")
    if display:
        return f"{head}-{display}-{digest}"
    return f"{head}-{digest}"


def _slugify_name_component(value: str) -> str:
    """Reduce an arbitrary string to a Sprite/Fly-safe name component.

    Sprite names are DNS-ish: lowercase ``[a-z0-9-]`` with no leading/trailing
    or doubled hyphens. Anything else (``/``, ``.``, uppercase, unicode from a
    profile directory or subagent id) is collapsed to a single hyphen.

    Because a Sprite name is a durable trust boundary (it selects a live VM,
    not just a label), lossy collapsing must not merge distinct inputs: when
    the slug is not byte-identical to the input, a short hash of the raw value
    is appended so e.g. ``team_prod`` and ``team-prod`` stay distinct. Values
    that are already slug-clean — every name the backend historically
    produced — are unchanged, so existing Sprites keep resolving. (A
    crafted clean value that equals another value's slug+hash output can
    still impersonate it; within one component this is accepted — task ids
    are internal, not attacker-supplied across trust domains.)
    """
    raw = value or ""
    slug = _collapse_slug(raw)
    if slug == raw:
        return slug
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}" if slug else digest


def _resolve_profile_identity() -> str | None:
    """Raw profile identity for Sprite naming, derived from HERMES_HOME.

    Returns ``None`` for the default profile (``~/.hermes`` or
    ``~/.hermes/profiles/default``), the profile directory name for a
    standard named profile, and a unique ``home:<path>`` identity for a
    custom HERMES_HOME outside the profiles tree — a nonstandard home is
    its own trust domain and must never silently share the default
    profile's Sprite.

    Deliberately does NOT use ``file_safety._resolve_active_profile_name``:
    that helper swallows resolution failures and returns ``"default"``,
    which for a durable sandbox identity is a fail-open into another trust
    domain. Here a resolution failure raises instead (fail closed).
    """
    try:
        from agent.file_safety import _hermes_home_path, _hermes_root_path
        home = _hermes_home_path().resolve()
        root = _hermes_root_path().resolve()
    except Exception as e:
        raise RuntimeError(
            "Sprites backend could not resolve the active Hermes profile; "
            f"refusing to fall back to the default profile's Sprite: {e}"
        ) from e
    if home == root:
        return None
    try:
        rel = home.relative_to(root / "profiles")
    except ValueError:
        return f"home:{home}"
    name = rel.parts[0] if rel.parts else None
    if name is None or name == "default":
        return None
    return name


def _resolve_sprite_name(task_id: str, prefix: str | None = None) -> str:
    """Deterministic, profile-scoped Sprite name.

    A Sprite is persistent and resumed *by name*, so its name is the durable
    identity of a session's live sandbox (processes, sockets, PID space — not
    just a filesystem snapshot). Naming contract:

    - Default profile, short task: ``hermes-{task_slug}`` — byte-compatible
      with the historical names, so already-created Sprites keep resolving.
      Clean task ids are verbatim-injective; lossy ones carry a
      per-component hash (see ``_slugify_name_component``).
    - Named profile / custom HERMES_HOME: ``hermes-{display}-{digest12}``
      where the digest covers the raw ``(profile, task_id)`` pair via a
      length-prefixed encoding (see ``_identity_digest``). The digest is the
      authoritative identity: it is collision-resistant across component
      boundaries, display-slug collisions, and separator forgery; the
      display prefix is cosmetic and is truncated to keep the whole name
      within a DNS label.
    - Any name that would exceed the DNS bound is composed via
      ``_bounded_name`` with the digest preserved intact.

    Profile resolution failure raises rather than falling back: silently
    entering the default profile's durable namespace would put a named
    profile's session inside another trust domain's live VM.

    ``prefix`` (an already-validated ``name_prefix``) replaces the leading
    ``hermes`` in every form above; ``None`` keeps the historical names
    byte-for-byte.
    """
    head = prefix or _DEFAULT_HEAD
    profile_raw = _resolve_profile_identity()
    if profile_raw is None:
        task_slug = _slugify_name_component(task_id) or "default"
        legacy = f"{head}-{task_slug}"
        if len(legacy) <= _MAX_NAME_LEN:
            return legacy
        # Too long for a DNS label (e.g. a session-keyed task id): fall to
        # the bounded form; the pair digest carries the identity.
        return _bounded_name(
            _collapse_slug(task_id), _identity_digest("", task_id or ""), head
        )
    profile_slug = _collapse_slug(profile_raw) or "profile"
    task_display = _collapse_slug(task_id) or "default"
    digest = _identity_digest(profile_raw, task_id or "")
    return _bounded_name(f"{profile_slug}-{task_display}", digest, head)


def _ephemeral_sprite_name(task_id: str, prefix: str | None = None) -> str:
    """Unique, non-resumable Sprite name for a non-persistent run.

    Ephemeral Sprites are created fresh and deleted on cleanup; their names
    deliberately contain a random component so no two runs — concurrent or
    sequential, same process or not — can ever attach the same live VM, and
    a stale survivor of a crashed run is never silently resumed.
    """
    nonce = uuid.uuid4().hex[:12]
    display = _collapse_slug(task_id) or "default"
    return _bounded_name(f"eph-{display}", nonce, prefix or _DEFAULT_HEAD)


def _delete_script(paths: list[str]) -> str:
    """``rm -f`` the paths, retrying with ``sudo -n`` if the plain rm fails."""
    quoted = " ".join(shlex.quote(p) for p in paths)
    return f"rm -f -- {quoted} || sudo -n rm -f -- {quoted}"


def _delete_batches(paths: list[str]) -> list[list[str]]:
    """Split ``paths`` so each URL-encoded delete command fits ``_DELETE_COMMAND_MAX``.

    A single path longer than the budget still gets its own batch.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    for p in paths:
        candidate = current + [p]
        if current and len(_url_quote(_delete_script(candidate), safe="")) > _DELETE_COMMAND_MAX:
            batches.append(current)
            candidate = [p]
        current = candidate
    if current:
        batches.append(current)
    return batches


def _tar_info(remote_path: str, size: int, mtime: float) -> tarfile.TarInfo:
    """The member header ``_build_archive`` writes for one file."""
    info = tarfile.TarInfo(remote_path.lstrip("/"))
    info.size = size
    info.mode = 0o644
    info.mtime = int(mtime)
    return info


def _tar_member_bytes(remote_path: str, size: int, mtime: float) -> int:
    """Bytes one member takes in the archive: its header blocks (more than one
    for a long or non-ASCII name) plus its data padded to a whole block."""
    header = _tar_info(remote_path, size, mtime).tobuf(
        tarfile.DEFAULT_FORMAT, tarfile.ENCODING, "surrogateescape")
    return len(header) + -(-size // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE


def _tar_archive_bytes(members_bytes: int) -> int:
    """Archive size for members totalling ``members_bytes``: tarfile ends the
    archive with two zero blocks and pads it to a whole record."""
    total = members_bytes + 2 * tarfile.BLOCKSIZE
    return -(-total // tarfile.RECORDSIZE) * tarfile.RECORDSIZE


def _fits_one_write(remote_path: str, size: int, mtime: float) -> bool:
    """True when the file, alone in an archive, is within the write limit
    (which also bounds it as a single ``fs/write``)."""
    member = _tar_member_bytes(remote_path, size, mtime)
    return _tar_archive_bytes(member) <= _SYNC_WRITE_MAX_BYTES


def _archive_batches(
    files: list[tuple[str, str]],
) -> tuple[list[list[tuple[str, str]]], list[tuple[str, str]]]:
    """Split ``files`` into archive batches and files to send in chunks.

    Each archive ``_build_archive`` makes from a batch, measured in real tar
    bytes, is at most ``_SYNC_WRITE_MAX_BYTES``. A file too large even for an
    archive of its own is returned in the second list instead, to be sent
    in chunks; no file is left out.
    """
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    chunked: list[tuple[str, str]] = []
    size = 0
    for host_path, remote_path in files:
        st = Path(host_path).stat()
        member = _tar_member_bytes(remote_path, st.st_size, st.st_mtime)
        if _tar_archive_bytes(member) > _SYNC_WRITE_MAX_BYTES:
            chunked.append((host_path, remote_path))
            continue
        if current and _tar_archive_bytes(size + member) > _SYNC_WRITE_MAX_BYTES:
            batches.append(current)
            current, size = [], 0
        current.append((host_path, remote_path))
        size += member
    if current:
        batches.append(current)
    return batches, chunked


def _build_archive(files: list[tuple[str, str]]) -> bytes:
    """Uncompressed tar (the Sprite needs no gzip) of ``files``.

    Each member is the remote path without its leading ``/``, mode 0644, owner
    root: the file mode a single ``fs/write`` sets. Host owner, permissions
    and file type are not carried over.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for host_path, remote_path in files:
            data = Path(host_path).read_bytes()
            info = _tar_info(remote_path, len(data), Path(host_path).stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


#: Archives and upload directories not modified for this long are taken to
#: be leftovers of a sync that never finished (transport failure, killed
#: process) and are removed by a later sync's sweep.
#:
#: A chunked upload modifies its directory with every part it writes (each
#: part is a new entry) and touches it again when the join starts, so it is
#: only ever idle for the longest of: one part write, or the join. The join
#: is abandoned by the client after at most ``_JOIN_TIMEOUT_MAX`` (600 s).
#: One part write is one ``fs/write`` of at most ``_SYNC_WRITE_MAX_BYTES``
#: (4 MiB), retried at most once after a 401: observed writes of that size
#: finish in seconds (a failing 5 MiB one took 8.2 s), and for two of them
#: to take two hours the link would have to carry under 1.2 KB/s without
#: ever stalling for the HTTP client's timeout. The whole upload may take
#: longer than this, which is fine: what is compared is the directory's
#: idle time, not its age. An archive is idle only between its write and its
#: unpack (``_UNPACK_TIMEOUT``). A sync stalled for longer than this loses
#: its staging; its join or unpack then fails and the sync is retried.
_STALE_SYNC_MINUTES = 120

#: Exit status of the unpack script when the archive could not be removed.
_ARCHIVE_LEFT_EXIT = 86


def _sweep_script(staging_dir: str) -> str:
    """Remove archives and chunked-upload directories in ``staging_dir``
    left behind by earlier syncs that never finished (only names this plugin
    makes). An upload directory goes as a whole, parts and temp output
    alike, and only once the directory itself has been idle for
    ``_STALE_SYNC_MINUTES``, so an upload still in flight is never touched."""
    staging = shlex.quote(staging_dir)
    return (
        f"find {staging} -mindepth 1 -maxdepth 1 "
        f"\\( -type f -name '.sync-*.tar' -o -type d -name '.sync-*.d' \\) "
        f"-mmin +{_STALE_SYNC_MINUTES} -exec rm -rf -- {{}} +"
    )


def _unpack_script(archive_path: str) -> str:
    """Unpack the archive at ``/``, retrying with ``sudo -n``, then remove it.

    The archive is a root-owned 0600 ``fs/write`` file, so a non-root exec
    user (the base image runs commands as ``sprite``) unpacks it through
    ``sudo -n``; files from earlier syncs are root-owned for the same reason.
    Exits with tar's status if the unpack fails, and ``_ARCHIVE_LEFT_EXIT``
    if the archive could not be removed. Also removes archives in the same
    directory left behind by earlier syncs whose unpack never ran.
    """
    q = shlex.quote(archive_path)
    tar = f"tar -xf {q} -C / --no-same-owner"
    sweep = _sweep_script(posixpath.dirname(archive_path))
    return (
        f"{{ {tar} || sudo -n {tar}; }}; rc=$?; "
        f"rm -f -- {q} 2>/dev/null || sudo -n rm -f -- {q} 2>/dev/null "
        f"|| rc={_ARCHIVE_LEFT_EXIT}; "
        f"{{ {sweep} || sudo -n {sweep}; }} >/dev/null 2>&1; "
        f"exit $rc"
    )


#: Joins the parts of a chunked upload; run as ``sh -c J sh DIR DEST N``.
#: Refuses an incomplete upload (fewer than N parts) and a destination that
#: is a directory or a symlink to one (``[ -d ]`` follows symlinks; ``mv
#: -T`` also refuses a real directory that appears in between). A symlink to
#: a file is replaced by the file, as a single ``fs/write`` would.
_JOIN_BODY = (
    'c=0; for p in "$1"/[0-9]*; do [ -f "$p" ] && c=$((c+1)); done; '
    '[ "$c" -eq "$3" ] || { echo "upload has $c of $3 parts" >&2; exit 65; }; '
    '[ ! -d "$2" ] || { echo "$2 is a directory" >&2; exit 66; }; '
    'p=${2%/*}; mkdir -p -- "${p:-/}" && '
    'cat -- "$1"/[0-9]* > "$1/out" && '
    'chmod 0644 -- "$1/out" && mv -fT -- "$1/out" "$2"'
)


def _assemble_script(upload_dir: str, remote_path: str, part_count: int) -> str:
    """Join the ``part_count`` parts in ``upload_dir`` into ``remote_path``,
    then remove ``upload_dir``.

    The Sprites 2 docs offer no append or partial write ("Files": one
    ``PUT fs/write`` per file, at most 16 MiB), so a file larger than one
    write arrives as numbered part files (``00000``, ``00001``, ...) in its
    own directory and is joined here. The command has the same length
    whatever the number of parts: the parts are found by a glob (the
    zero-padded names sort in order in any locale) and counted against
    ``part_count``, and the cleanup is one ``rm -rf``. The parts are
    concatenated into ``out`` inside ``upload_dir``, set to 0644 like a
    single ``fs/write``, and renamed over the destination with ``mv -fT``
    (GNU coreutils, which the Sprites base image ships as ``mv``). Sync
    destinations are under ``~/.hermes`` like the staging directory, so the
    rename stays on one filesystem: a reader sees the old file or the new
    one, never a partial one. Nothing
    is written next to the destination, so a join cut short leaves nothing
    there; ``out`` goes with the directory, by this script or a later
    sweep. The directory is touched first so the sweep sees it as live.

    The parts are root-owned 0600 ``fs/write`` files, so a non-root exec
    user retries through ``sudo -n`` as ``_unpack_script`` does. Exits
    non-zero if the join fails (the destination is left as it was), and
    ``_ARCHIVE_LEFT_EXIT`` if the join worked but ``upload_dir`` could not
    be removed. Stale archives and upload directories are swept as well.
    """
    args = " ".join(shlex.quote(a) for a in (upload_dir, remote_path, str(part_count)))
    sweep = _sweep_script(posixpath.dirname(upload_dir))
    return (
        f"set -- {args}; J={shlex.quote(_JOIN_BODY)}; "
        f'{{ touch -- "$1" || sudo -n touch -- "$1"; }} >/dev/null 2>&1; '
        f'sh -c "$J" sh "$@" || {{ rm -f -- "$1/out" 2>/dev/null; '
        f'sudo -n sh -c "$J" sh "$@"; }}; rc=$?; '
        f'rm -rf -- "$1" 2>/dev/null || sudo -n rm -rf -- "$1" 2>/dev/null '
        f"|| {{ [ $rc -ne 0 ] || rc={_ARCHIVE_LEFT_EXIT}; }}; "
        f"{{ {sweep} || sudo -n {sweep}; }} >/dev/null 2>&1; "
        f"exit $rc"
    )


class _ExecFailed(RuntimeError):
    """A sync exec ran and exited non-zero; ``exit_code`` is its status."""

    def __init__(self, message: str, exit_code):
        super().__init__(message)
        self.exit_code = exit_code

class SpritesEnvironment(BaseEnvironment):
    """Sprites backend: stateful cloud sandboxes on Fly.io.

    Spawn-per-call via ``_ThreadedProcessHandle`` wrapping blocking
    ``sprite.command(...).combined_output()`` calls. The SDK timeout is
    used to bound the client wait, not as a remote process-kill guarantee.
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        cwd: str = "/root",
        timeout: int = 60,
        persistent_filesystem: bool = True,
        task_id: str = "default",
        name_prefix: str | None = None,
        scope: str | None = None,
        image: str | None = None,
        max_cpus: int | None = None,
        max_ram_gb: int | None = None,
        max_disk_gb: int | None = None,
    ):
        requested_cwd = cwd
        # Validate settings before anything else (including the SDK import
        # and token lookup) so misconfiguration fails closed with a clear
        # message and no Sprites API call is ever made.
        prefix, scope = resolve_naming_settings(name_prefix, scope)
        create_config = resolve_create_settings(
            image, max_cpus, max_ram_gb, max_disk_gb
        )
        super().__init__(cwd=cwd, timeout=timeout)

        # Standalone plugin dependencies are installed alongside the plugin,
        # not through Hermes's allowlist of built-in lazy dependencies.
        from sprites import SpritesClient
        from sprites.exceptions import NotFoundError, SpriteError

        self._NotFoundError = NotFoundError
        self._SpriteError = SpriteError

        # Decide the token source here, in the caller's secret scope, and
        # read the token fresh (SPRITES_TOKEN_FILE may have been rewritten
        # since the last environment). Re-reads after a 401 use the same
        # source, never a new variable lookup.
        self._SpritesClient = SpritesClient
        self._client_timeout = max(30.0, float(timeout))
        self._token_source = _token_source()
        self._token = self._token_source.read()
        self._client = self._new_client(self._token)
        self._auth_lock = threading.Lock()
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._name_prefix = prefix
        self._scope = scope
        self._create_config = create_config
        self._lock = threading.Lock()
        self._sprite = None
        # Extra environment for every exec; set only when the Sprite reports
        # no $HOME (see below). None keeps exec requests unchanged.
        self._exec_env: dict | None = None

        if persistent_filesystem:
            # ``instance`` scope: every task id (gateway sessions, cron,
            # delegate_task children) resumes the same per-profile Sprite.
            name_task_id = _INSTANCE_TASK_ID if scope == "instance" else task_id
            sprite_name = _resolve_sprite_name(name_task_id, prefix)
            _require_within_prefix(sprite_name, prefix)
            self._sprite_name = sprite_name
            try:
                self._sprite = self._with_auth_retry(
                    lambda: self._client.get_sprite(sprite_name))
                logger.info(
                    "Sprites: resumed existing sprite %s for task %s",
                    self._sprite.name, task_id,
                )
                if create_config:
                    logger.debug(
                        "Sprites: image/sizing settings apply only when a "
                        "Sprite is created; existing sprite %s is used as-is",
                        self._sprite.name,
                    )
            except NotFoundError:
                # Cross-process first-use race: two processes can both see
                # 404 here; one create wins and the other gets a duplicate-
                # name error. Adopt the winner instead of failing — re-GET
                # the exact deterministic name, and only re-raise the create
                # error if the Sprite genuinely does not exist (a real
                # create failure, not a race).
                try:
                    self._sprite = self._create_sprite(sprite_name)
                    logger.info(
                        "Sprites: created sprite %s for task %s",
                        self._sprite.name, task_id,
                    )
                except SpriteError as create_err:
                    try:
                        self._sprite = self._with_auth_retry(
                            lambda: self._client.get_sprite(sprite_name))
                        logger.info(
                            "Sprites: adopted sprite %s created concurrently "
                            "by another process (task %s)",
                            self._sprite.name, task_id,
                        )
                    except NotFoundError:
                        raise create_err
        else:
            # Ephemeral (container_persistent: false): the sandbox must not
            # survive or be shared across sessions (#82731 contract). A
            # deterministic name would let two independent ephemeral runs
            # attach one live VM — and either cleanup would delete it out
            # from under the other — or silently resume a stale survivor of
            # a crashed prior run. Mint a unique name and NEVER adopt: this
            # constructor only ever creates.
            # ``scope`` does not apply here: per-run uniqueness is the whole
            # point of ephemeral mode, so instance scope never shares these.
            sprite_name = _ephemeral_sprite_name(task_id, prefix)
            _require_within_prefix(sprite_name, prefix)
            self._sprite_name = sprite_name
            self._sprite = self._create_sprite(sprite_name)
            logger.info(
                "Sprites: created ephemeral sprite %s for task %s",
                self._sprite.name, task_id,
            )

        # Detect remote home dir for .hermes sync target.
        self._remote_home = "/root"
        try:
            home = self._with_auth_retry(
                lambda: self._sprite.command(
                    "bash", "-c", "echo $HOME", timeout=15).combined_output()
            ).decode().strip()
            if home:
                self._remote_home = home
                if requested_cwd in {"~", "/root"}:
                    self.cwd = home
            else:
                self._apply_home_fallback()
        except Exception:
            pass

        self._fs = self._sprite.filesystem("/")
        self._use_archives = True
        self._sync_manager = _SharedSpriteSyncManager(
            # Never ship the token file's directory into the Sprite. This is
            # the one place host paths are seen: FileSyncManager copies each
            # file to a staging dir before upload_fn / bulk_upload_fn run.
            get_files_fn=lambda: _without_token_dir(
                iter_sync_files(f"{self._remote_home}/.hermes"),
                self._token_source.token_dir),
            upload_fn=self._sprite_upload,
            bulk_upload_fn=self._sprite_bulk_upload,
            delete_fn=self._sprite_delete,
            shared_lock=_shared_sync_lock(sprite_name),
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    # ------------------------------------------------------------------
    # Token renewal
    # ------------------------------------------------------------------

    def _new_client(self, token: str):
        """A client for ``token`` that records HTTP 401 responses (see ``_http_auth``)."""
        client = self._SpritesClient(token=token, timeout=self._client_timeout)
        hooks = getattr(getattr(client, "_client", None), "event_hooks", None)
        if isinstance(hooks, dict) and isinstance(hooks.get("response"), list):
            hooks["response"].append(_note_http_status)
        return client

    def _with_auth_retry(self, fn):
        """Call ``fn()``; after an HTTP 401, retry it once if the token changed.

        Only a token file can change, so only ``SPRITES_TOKEN_FILE`` setups
        retry. A 401 means the request was refused before anything ran, so
        the retry cannot run a command or create a Sprite twice. ``fn`` must
        read ``self._client`` / ``self._sprite`` / ``self._fs`` when called,
        so the retry goes through the new client. With an unchanged or
        unreadable token the 401 is raised as is. An error whose text holds
        the token is replaced by ``SpritesApiError``.
        """
        client, token = self._client, self._token
        try:
            return self._call_once(fn)
        except Exception as e:
            error = e
        if self._was_auth_error(error) and self._refresh_client_after_401(client):
            try:
                return self._call_once(fn)
            except Exception as e:
                error = e
        if _mentions(error, (token, self._token)):
            # Raised outside the except blocks: no __context__ links back
            # to the original error.
            name = type(error).__name__
            error = None
            raise SpritesApiError(
                f"Sprites API call failed ({name}); its message was withheld "
                "because it contained the Sprites token."
            )
        raise error

    @staticmethod
    def _call_once(fn):
        _http_auth.saw_401 = False
        return fn()

    @staticmethod
    def _was_auth_error(exc: BaseException) -> bool:
        return _is_auth_error(exc) or bool(getattr(_http_auth, "saw_401", False))

    def _refresh_client_after_401(self, failed_client) -> bool:
        """Re-read the token file; on a change, switch to a new client. True to retry."""
        if self._token_source.path is None:
            return False  # an env token cannot change under a running process
        with self._auth_lock:
            if self._client is not failed_client:
                return True  # another call already switched clients
            try:
                token = self._token_source.read()
            except SpritesTokenError as e:
                logger.warning("Sprites: HTTP 401 and the token could not be re-read: %s", e)
                return False
            if token == self._token:
                return False
            client = self._new_client(token)
            try:
                sprite = client.get_sprite(self._sprite_name) if self._sprite is not None else None
            except Exception as e:
                logger.warning(
                    "Sprites: token changed after HTTP 401, but %s could not be "
                    "reopened with it: %s", self._sprite_name, type(e).__name__,
                )
                try:
                    client.close()
                except Exception:
                    pass
                return False
            old_client = self._client
            self._client, self._token = client, token
            if sprite is not None:
                self._sprite = sprite
                self._fs = sprite.filesystem("/")
            try:
                old_client.close()
            except Exception:
                pass
            logger.info("Sprites: token changed after HTTP 401; retrying with the new token")
            return True

    # ------------------------------------------------------------------
    # Creation and exec helpers
    # ------------------------------------------------------------------

    def _create_sprite(self, sprite_name: str):
        """Create ``sprite_name``, with the configured image/sizing if any.

        With no create settings this is the SDK's plain ``create_sprite(name)``
        call, byte-for-byte the request the plugin has always sent.

        With settings, POST /v1/sprites directly: sprites-py 0.5's
        ``SpriteConfig`` sends ``ram_mb`` / ``cpus`` / ``storage_gb`` and has no
        image field, while the API takes ``image_url`` / ``max_cpus`` /
        ``max_ram_gb`` / ``max_disk_gb``. The settings apply at create only.
        Never send ``PUT config`` to change an existing Sprite: a partial
        config update was observed to drop ``image_url`` and leave the Sprite
        unable to boot.

        Each request retries on its own after a 401 (``_with_auth_retry``):
        the GET after a successful POST never repeats the POST.
        """
        if not self._create_config:
            return self._with_auth_retry(lambda: self._client.create_sprite(sprite_name))

        body = {"name": sprite_name, "config": dict(self._create_config)}
        self._with_auth_retry(lambda: self._post_create(sprite_name, body))
        return self._with_auth_retry(lambda: self._client.get_sprite(sprite_name))

    def _post_create(self, sprite_name: str, body: dict) -> None:
        """POST /v1/sprites once; raise the SDK's exception types on failure."""
        import httpx
        from sprites.exceptions import AuthenticationError, NetworkError

        try:
            response = self._client.http_client.post(
                f"{self._client.base_url}/v1/sprites",
                json=body,
                timeout=_CREATE_WITH_CONFIG_TIMEOUT,
            )
        except httpx.ReadTimeout as e:
            # Not a SpriteError, so the persistent path does not re-GET a
            # half-created Sprite now; the next construction's GET adopts it.
            image = self._create_config.get("image_url")
            pull = f" (the first pull of image {image} can take a few minutes)" if image else ""
            if not self._persistent:
                logger.warning("Sprites: ephemeral sprite %s may be orphaned if its "
                               "create completes later", sprite_name)
            raise SpriteStillCreating(
                f"Sprite {sprite_name} is still being created{pull}; "
                "retry the command shortly."
            ) from e
        except httpx.RequestError as e:
            raise NetworkError(f"Network error creating sprite: {e}") from e
        # Same exception types as the SDK's create_sprite (all SpriteError
        # subclasses). A 409 (name taken) raises SpriteError like any other
        # failure, so the caller's re-GET adopts a Sprite another process
        # created.
        if response.status_code == 401:
            raise AuthenticationError("Authentication failed for create sprite")
        if response.status_code == 404:
            raise self._NotFoundError("Resource not found for create sprite")
        if not response.is_success:
            raise self._SpriteError(
                f"Failed create sprite (status {response.status_code}): {response.text}"
            )

    def _apply_home_fallback(self) -> None:
        """Handle a Sprite whose commands run with ``$HOME`` unset.

        Imported images can run commands as root with no ``HOME``. When the
        user is root, use ``/root`` as the sync target and pass ``HOME`` on
        every later exec so ``~`` resolves there. Otherwise keep the
        ``/root`` fallback without changing exec.
        """
        try:
            uid = self._with_auth_retry(
                lambda: self._sprite.command(
                    "sh", "-c", "id -u", timeout=15).combined_output()
            ).decode().strip()
        except Exception as e:
            logger.debug("Sprites: could not read uid on %s: %s", self._sprite_name, e)
            return
        if uid == "0":
            self._remote_home = "/root"
            self._exec_env = {"HOME": "/root"}
            logger.info(
                "Sprites: %s reports no $HOME; using /root for exec and file sync",
                self._sprite_name,
            )

    def _command_kwargs(self) -> dict:
        """Extra ``sprite.command`` kwargs; empty unless the HOME fallback is active."""
        return {"env": dict(self._exec_env)} if self._exec_env else {}

    # ------------------------------------------------------------------
    # File sync callbacks
    # ------------------------------------------------------------------

    def _sprite_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file with ``fs/write``, or in chunks if it is
        larger than one write (``_SYNC_WRITE_MAX_BYTES``).

        ``mkdir_parents=True`` makes the write create missing parent
        directories, so no separate mkdir (which needs ``fs/list``, absent on
        the Sprites 2.0 runtime) is made. Mode and parent creation are the
        sprites-py defaults the plugin has always sent.
        """
        with open(host_path, "rb") as f:
            data = f.read(_SYNC_WRITE_MAX_BYTES + 1)
        if len(data) > _SYNC_WRITE_MAX_BYTES:
            del data
            self._sprite_upload_chunked(host_path, remote_path)
            return
        self._with_auth_retry(
            lambda: (self._fs / remote_path).write_bytes(data, mode=0o644, mkdir_parents=True)
        )

    def _sprite_upload_chunked(self, host_path: str, remote_path: str) -> None:
        """Upload one file too large for a single write as chunks, then join
        them on the Sprite with one exec (see ``_assemble_script``).

        Each chunk of at most ``_SYNC_WRITE_MAX_BYTES`` is read from the
        file as it is sent, so at most one chunk is held in memory, and goes
        in one ``fs/write`` to its own numbered part file (root-only, mode
        0600, like an archive) in a directory of its own,
        ``~/.hermes/.sync-staging/.sync-<random>.d``. The file is read to its
        end, so all of it is sent whatever its size. Any failure raises, so
        ``FileSyncManager`` rolls back and retries; the directory is removed
        on a best-effort basis, and otherwise by a later sync's sweep.
        """
        upload_dir = posixpath.join(
            self._remote_home, ".hermes", ".sync-staging",
            f".sync-{uuid.uuid4().hex}.d")
        parts: list[str] = []
        size = 0
        try:
            with open(host_path, "rb") as f:
                while True:
                    chunk = f.read(_SYNC_WRITE_MAX_BYTES)
                    if not chunk and parts:
                        break
                    part = posixpath.join(upload_dir, f"{len(parts):05d}")
                    parts.append(part)
                    self._with_auth_retry(
                        lambda: (self._fs / part).write_bytes(
                            chunk, mode=0o600, mkdir_parents=True)
                    )
                    size += len(chunk)
                    last = len(chunk) < _SYNC_WRITE_MAX_BYTES
                    del chunk  # drop it before reading the next one
                    if last:
                        break
            self._exec_or_raise(
                _assemble_script(upload_dir, remote_path, len(parts)),
                _join_timeout(size),
                f"joining {len(parts)} uploaded part(s) into {remote_path}",
            )
        except _ExecFailed as e:
            if e.exit_code != _ARCHIVE_LEFT_EXIT:
                raise  # the join script removed the directory itself
            # The file is in place; only its parts could not be removed.
            # Empty them so they hold no file data (as for an archive).
            logger.warning("%s; emptying the parts instead", e)
            for part in parts:
                self._with_auth_retry(
                    lambda: (self._fs / part).write_bytes(b"", mode=0o600, mkdir_parents=True))
        except BaseException:
            self._remove_upload_dir_quietly(upload_dir)
            raise

    def _remove_upload_dir_quietly(self, upload_dir: str) -> None:
        """Best-effort removal of a chunked upload's directory after a failure."""
        rm = f"rm -rf -- {shlex.quote(upload_dir)}"
        try:
            self._exec_or_raise(
                f"{rm} 2>/dev/null || sudo -n {rm}", _DELETE_TIMEOUT,
                "removing an unfinished chunked upload")
        except Exception as e:  # the sweep in a later sync gets it
            logger.debug("Sprites: could not remove upload directory: %s", e)

    def _sprite_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload changed files as archives: per archive, one ``fs/write`` of a
        tar into ``~/.hermes/.sync-staging`` and one exec that unpacks and
        removes it.

        One ``fs/write`` per file made a sync cost as many requests as there
        are files. The skills tree alone is a few hundred files, past the
        per-Sprite request rate limit, so every sync was cut off by HTTP 429,
        rolled back, and resent in full on the next command (and the burst
        could rate-limit the command itself).

        When the unpack exec runs but fails (no ``tar``, a non-root user
        without working sudo, a full disk), the batch is sent the old way,
        one ``fs/write`` per file, which reports its own error if it fails
        too. A transport failure raises at once. Either way a failure
        reaches ``FileSyncManager``, which rolls back and retries.

        If the archive could not be removed (a non-root user without sudo
        cannot delete a root-owned file), it is overwritten with nothing and
        this environment stops using archives, so they do not pile up.

        No write is larger than ``_SYNC_WRITE_MAX_BYTES``. A file too large
        for an archive of its own is sent in chunks and joined on the Sprite
        (``_sprite_upload_chunked``), on this path and the one-by-one path
        alike. Every file passed in is written, or this raises.
        """
        batches, chunked = _archive_batches(files)
        for batch in batches:
            if not self._use_archives:
                for host_path, remote_path in batch:
                    self._sprite_upload(host_path, remote_path)
                continue
            # 0600: only root can read the archive, which may hold credential
            # files whose own directories are more restrictive than its own.
            archive = posixpath.join(
                self._remote_home, ".hermes", ".sync-staging",
                f".sync-{uuid.uuid4().hex}.tar")
            payload = _build_archive(batch)
            if len(payload) > _SYNC_WRITE_MAX_BYTES:
                # A file grew between being measured and being read (the
                # staged copies FileSyncManager passes in do not). Fail the
                # sync rather than send a write the Sprite refuses or skip
                # a file the manager would then record as synced.
                raise RuntimeError(
                    f"Sprites: sync archive is {len(payload)} bytes, over the "
                    f"{_SYNC_WRITE_MAX_BYTES}-byte upload limit")
            self._with_auth_retry(
                lambda: (self._fs / archive).write_bytes(payload, mode=0o600, mkdir_parents=True)
            )
            try:
                self._exec_or_raise(
                    _unpack_script(archive), _UNPACK_TIMEOUT,
                    f"unpacking {len(batch)} synced file(s)",
                )
            except _ExecFailed as e:
                logger.warning(
                    "%s; uploading the %d file(s) one by one instead", e, len(batch),
                )
                if e.exit_code == _ARCHIVE_LEFT_EXIT:
                    self._use_archives = False
                    self._with_auth_retry(
                        lambda: (self._fs / archive).write_bytes(
                            b"", mode=0o600, mkdir_parents=True))
                for host_path, remote_path in batch:
                    self._sprite_upload(host_path, remote_path)
        for host_path, remote_path in chunked:
            self._sprite_upload_chunked(host_path, remote_path)

    def _exec_or_raise(self, script: str, timeout: float, what: str) -> None:
        """Run ``script`` with ``sh -c``; a non-zero exit raises ``_ExecFailed``."""
        from sprites.exceptions import ExitError

        def run():
            return self._sprite.command(
                "sh", "-c", script, timeout=timeout, **self._command_kwargs(),
            ).combined_output()

        try:
            self._with_auth_retry(run)
        except ExitError as e:
            output = ((e.stdout or b"") + (e.stderr or b"")).decode(
                "utf-8", errors="replace"
            ).strip()
            code = e.exit_code() if callable(getattr(e, "exit_code", None)) else None
            raise _ExecFailed(
                f"Sprites: {what} failed (exit {'?' if code is None else code}): {output}",
                code,
            ) from e

    def _sprite_delete(self, remote_paths: list[str]) -> None:
        """Delete remote files with ``rm -f`` over exec.

        Sprites 2.0 has no ``fs/delete``. Files written by ``fs/write`` are
        owned by root, while commands on the base image run as ``sprite``, so
        a failed ``rm`` is retried once with ``sudo -n``.

        Missing files are benign (``rm -f``); any OTHER failure must propagate
        so ``FileSyncManager`` rolls back and retries on the next cycle.
        Swallowing errors here would falsely commit the deletion — the manager
        drops the path from its synced set and never retries, which can leave
        stale credential material in a durable Sprite permanently.
        """
        for batch in _delete_batches(remote_paths):
            self._exec_or_raise(
                _delete_script(batch), _DELETE_TIMEOUT,
                f"deleting {len(batch)} synced file(s)",
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking SDK call."""
        sprite = self._sprite
        from sprites.exceptions import ExitError, TimeoutError as SpritesTimeout

        if login:
            shell_cmd = ["bash", "-l", "-c", cmd_string]
        else:
            shell_cmd = ["bash", "-c", cmd_string]

        # Bound the client wait, including when no deadline was supplied.
        # SDK cancellation closes the WebSocket, but does not guarantee the
        # remote process is killed (cancel_fn=None below). Absent/nonpositive
        # timeouts get a fallback deadline; explicit positive values pass through.
        cmd_timeout = float(timeout) if timeout and timeout > 0 else 3600.0

        cmd_kwargs = self._command_kwargs()

        def run():
            # After a token switch, self._sprite is bound to the new client.
            current = self._sprite or sprite
            return current.command(*shell_cmd, timeout=cmd_timeout, **cmd_kwargs).combined_output()

        def exec_fn() -> tuple[str, int]:
            try:
                output = self._with_auth_retry(run)
                return (output.decode("utf-8", errors="replace"), 0)
            except ExitError as e:
                # ``e.stdout`` carries the combined output when raised from
                # combined_output(); ``e.stderr`` is empty in that path.
                buf = (e.stdout or b"") + (e.stderr or b"")
                return (buf.decode("utf-8", errors="replace"),
                        e.exit_code() if callable(getattr(e, "exit_code", None)) else 1)
            except (SpritesTimeout, TimeoutError):
                # sprites-py has matching async and synchronous deadlines.
                # Its Future.result() can raise built-in TimeoutError before
                # the coroutine translates expiry to the SDK's exception.
                return (f"command timed out after {cmd_timeout}s\n", 124)

        # No external cancel: the SDK does not expose a kill hook on a
        # running Cmd. The deadline above is the cancellation path.
        return _ThreadedProcessHandle(exec_fn, cancel_fn=None)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        with self._lock:
            if self._sprite is None:
                return

            # No sync_back: the Sprite's persistent ext4 filesystem IS the
            # authoritative store. Files the agent touched stay in the Sprite
            # and are visible again on the next session that resumes by the
            # same task_id. (For ephemeral runs with persistent=False, the
            # Sprite is intentionally deleted with its filesystem.)

            try:
                if self._persistent:
                    logger.info(
                        "Sprites: leaving sprite %s running (persistent)",
                        self._sprite.name,
                    )
                else:
                    self._with_auth_retry(lambda: self._sprite.delete())
                    logger.info("Sprites: deleted sprite %s", self._sprite.name)
            except Exception as e:
                logger.warning(
                    "Sprites: cleanup failed for %s: %s", self._sprite_name, e
                )
            finally:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._sprite = None
