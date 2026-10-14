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
File sync therefore uploads with ``fs/write`` alone and deletes through exec.
"""

import hashlib
import logging
import re
import shlex
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote as _url_quote

from tools.environments.base import BaseEnvironment, EnvironmentConnectionError

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

#: Hermes's default deadline for one tool call (``agent/tool_executor.py``
#: ``_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S``, also the sequential default). The
#: Sprite is created inside the terminal tool call, so a create that outlives
#: the deadline returns ``tool_timeout`` while the constructor keeps running
#: and holds the task's creation lock.
_TOOL_DEADLINE_DEFAULT = 420.0

#: Part of the tool deadline kept for the rest of construction ($HOME probe,
#: file sync, session init), or for the re-GET after a create that timed out.
_CREATE_DEADLINE_MARGIN = 120.0

#: Bounds on the create wait. Pulling an image the platform has not cached
#: yet can take minutes (a first pull of a ~GB image measured 73 s; cached,
#: under 5 s), past the SDK's 120 s create timeout. 600 s is the old fixed
#: wait, kept when Hermes's tool deadline is disabled.
_CREATE_WAIT_MIN = 30.0
_CREATE_WAIT_MAX = 600.0

#: Connect, write and pool limits for a bounded request, so the read wait
#: dominates its total time.
_CREATE_PHASE_TIMEOUT = 15.0

#: Extra time after the create budget for the GET of a created Sprite or the
#: re-GET after a failed create. It comes out of the deadline margin, never
#: more than half the budget.
_RECOVERY_GET_TIMEOUT = 15.0

#: Least create budget left for the POST to be worth sending (capped at a
#: quarter of the create wait for very short deadlines).
_CREATE_MIN_POST = 5.0

#: Client wait for the exec that deletes synced files.
_DELETE_TIMEOUT = 60.0

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


def _tool_deadline() -> float | None:
    """Deadline (seconds) Hermes puts on the tool call creating the Sprite.

    Read from the running Hermes when ``agent.tool_executor`` is loaded
    (``timeouts.tools.sequential_call`` / ``concurrent_batch``; the terminal
    tool runs under either, so the shorter applies). ``None`` means both are
    disabled. Falls back to the 420 s default when it cannot be read.
    """
    executor = sys.modules.get("agent.tool_executor")
    resolvers = [
        getattr(executor, name, None)
        for name in ("_resolve_sequential_tool_timeout", "_resolve_concurrent_tool_timeout")
    ]
    if executor is None or not all(callable(fn) for fn in resolvers):
        return _TOOL_DEADLINE_DEFAULT
    try:
        values = [fn() for fn in resolvers]
    except Exception:
        return _TOOL_DEADLINE_DEFAULT
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            return _TOOL_DEADLINE_DEFAULT
        if value == value and value > 0:  # NaN and non-positive mean "no deadline"
            finite.append(value)
    return min(finite) if finite else None


def _create_wait() -> float:
    """Budget for looking up and creating a Sprite with ``config``.

    It starts when construction starts and stays below the tool deadline.

    300 s under the default 420 s deadline. A shorter configured deadline
    shrinks it (to 30 s, or half the deadline when that is shorter still); a
    disabled deadline restores the old 600 s.
    """
    deadline = _tool_deadline()
    if deadline is None:
        return _CREATE_WAIT_MAX
    floor = min(_CREATE_WAIT_MIN, deadline / 2)
    return min(_CREATE_WAIT_MAX, max(floor, deadline - _CREATE_DEADLINE_MARGIN))


def _bounded_timeout(total: float):
    """httpx timeout whose connect and read phases together fit in ``total``."""
    import httpx
    phase = min(_CREATE_PHASE_TIMEOUT, total / 4)
    return httpx.Timeout(total - phase, connect=phase, write=phase, pool=phase)


class _CallExpired(Exception):
    """A Sprites API call did not return within its share of the budget."""


def _call_within(seconds: float, fn, *args):
    """Return ``fn(*args)``, or raise ``_CallExpired`` after ``seconds``.

    httpx timeouts bound each connect or read, not a whole request: a server
    that keeps sending bytes slowly can hold a request open indefinitely. The
    call runs on a daemon thread and the caller stops waiting at ``seconds``,
    so construction (and the task's creation lock) ends in time. Callers
    close the abandoned call's client (see ``_bounded_call``) so it stops
    too; its result is discarded.
    """
    import contextvars
    result: dict = {}
    done = threading.Event()
    context = contextvars.copy_context()

    def _run():
        try:
            result["value"] = context.run(fn, *args)
        except BaseException as e:  # handed to the waiting caller
            result["error"] = e
        finally:
            done.set()

    threading.Thread(target=_run, name="sprites-create-call", daemon=True).start()
    if not done.wait(max(0.0, seconds)):
        raise _CallExpired()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _is_name_taken(err: BaseException) -> bool:
    """True for a create rejected because the name exists (HTTP 409).

    Both the SDK and the direct create format failures as
    ``Failed create sprite (status <code>): <body>``.
    """
    return "(status 409)" in str(err)


def _still_creating_error(sprite_name: str, create_config: dict, *,
                          waited: float | None = None, name_taken: bool = False,
                          persistent: bool = True) -> EnvironmentConnectionError:
    """Retryable error for a Sprite whose create has not finished yet.

    Hermes's terminal tool reports it as ``status: degraded`` with the retry
    hint and caches nothing, so the next call starts over. For a persistent
    Sprite its GET resumes the Sprite once the platform has finished it.
    """
    image = create_config.get("image_url")
    after = f" after {waited:.0f}s" if waited is not None else ""
    pulling = f" (the first pull of image {image} can take a few minutes)" if image else ""
    reason = f"Sprite {sprite_name} is still being created{after}{pulling}"
    if name_taken:
        # Fixed text: the raw 409 body is not passed to the model.
        reason += "; Sprites reports the name as taken (HTTP 409) but does not list it yet"
    if persistent:
        next_step = ("the next attempt uses this Sprite once the platform has "
                     "finished creating it.")
    else:
        next_step = "the next attempt creates a new Sprite."
    return EnvironmentConnectionError(
        reason,
        retry_hint=("The Sprite is not ready yet; this is not a command failure. "
                    f"Retry the same command shortly; {next_step}"),
    )


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
        # Hermes's tool deadline started before this call; the create budget
        # (used only with create settings) counts from here.
        started = time.monotonic()
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
        from sprites.exceptions import (
            AuthenticationError, NetworkError, NotFoundError, SpriteError,
        )

        self._NotFoundError = NotFoundError
        self._SpriteError = SpriteError

        from agent.secret_scope import get_secret
        token = get_secret("SPRITES_TOKEN") or get_secret("SPRITE_TOKEN")
        if not token:
            raise ValueError(
                "Sprites backend requires SPRITES_TOKEN. "
                "Run `hermes setup terminal` or set SPRITES_TOKEN in .env."
            )
        self._client = SpritesClient(
            token=token,
            timeout=max(30.0, float(timeout)),
        )
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._name_prefix = prefix
        self._scope = scope
        self._create_config = create_config
        # With create settings, the GET and create share one budget (see
        # _create_wait); the GET of the created Sprite and the re-GET after a
        # failed create may run up to a short recovery time past it. Every
        # such call is cut off at its limit (_call_within), so all of them end
        # within the tool deadline less the time kept for the rest of
        # construction. Without create settings the calls are the plain SDK
        # ones with the SDK's own timeouts.
        self._create_wait = _create_wait() if create_config else _CREATE_WAIT_MAX
        self._create_deadline = started + self._create_wait
        self._recovery_deadline = self._create_deadline + min(
            _RECOVERY_GET_TIMEOUT, self._create_wait / 2)
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
                self._sprite = self._get_sprite_within(
                    sprite_name, self._create_deadline)
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
                # A create that outlived its wait (EnvironmentConnectionError)
                # takes the same path: the platform may have finished it, and
                # a later attempt's GET resumes this same name. A persistent
                # Sprite (session or instance scope) is never deleted or
                # renamed here.
                try:
                    self._sprite = self._create_sprite(sprite_name)
                    logger.info(
                        "Sprites: created sprite %s for task %s",
                        self._sprite.name, task_id,
                    )
                except (SpriteError, EnvironmentConnectionError) as create_err:
                    try:
                        self._sprite = self._get_sprite_within(
                            sprite_name, self._recovery_deadline)
                        logger.info(
                            "Sprites: adopted sprite %s created concurrently "
                            "or by an earlier attempt (task %s)",
                            self._sprite.name, task_id,
                        )
                    except (SpriteError, EnvironmentConnectionError) as get_err:
                        if isinstance(get_err, AuthenticationError):
                            raise  # a bad token is not "still creating"
                        if isinstance(create_err, EnvironmentConnectionError):
                            # The re-GET failed too (slow or unreachable
                            # API); the retryable result still applies.
                            raise create_err
                        if _is_name_taken(create_err) and (
                                isinstance(get_err, (NotFoundError, NetworkError,
                                                     EnvironmentConnectionError))):
                            # Name taken but not visible (or not checkable)
                            # yet: an earlier attempt's create is still in
                            # progress.
                            logger.debug("Sprites: create of %s returned: %s",
                                         sprite_name, create_err)
                            raise _still_creating_error(
                                sprite_name, self._create_config, name_taken=True,
                            ) from create_err
                        if isinstance(get_err, NotFoundError):
                            raise create_err
                        raise
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
            cmd = self._sprite.command("bash", "-c", "echo $HOME", timeout=15)
            home = cmd.combined_output().decode().strip()
            if home:
                self._remote_home = home
                if requested_cwd in {"~", "/root"}:
                    self.cwd = home
            else:
                self._apply_home_fallback()
        except Exception:
            pass

        self._fs = self._sprite.filesystem("/")
        self._sync_manager = _SharedSpriteSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._sprite_upload,
            delete_fn=self._sprite_delete,
            shared_lock=_shared_sync_lock(sprite_name),
        )
        self._sync_manager.sync(force=True)
        self.init_session()

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
        """
        if not self._create_config:
            return self._client.create_sprite(sprite_name)

        import httpx
        from sprites.exceptions import AuthenticationError, NetworkError

        body = {"name": sprite_name, "config": dict(self._create_config)}
        # The create gets what is left of the construction budget, which
        # stays below Hermes's tool deadline: past it the user gets
        # tool_timeout while this constructor runs on and holds the task's
        # creation lock, so every retry would queue behind it.
        wait = self._create_deadline - time.monotonic()
        if wait < min(_CREATE_MIN_POST, self._create_wait / 4):
            # The lookup used the budget; nothing was sent, so the retry's
            # GET finds no Sprite and creates it then.
            raise EnvironmentConnectionError(
                f"Sprite {sprite_name} was not created: the Sprites API was "
                "too slow to answer before the tool deadline",
                retry_hint="This is not a command failure. Retry the same command shortly.",
            )
        try:
            response = self._bounded_call(
                wait, lambda client: client.http_client.post(
                    f"{client.base_url}/v1/sprites",
                    json=body,
                    timeout=_bounded_timeout(wait),
                ))
        except (httpx.ReadTimeout, _CallExpired) as e:
            # The request reached the platform, which may still finish the
            # create. Say so in a retryable result instead of a stack trace.
            self._warn_ephemeral_left_behind(sprite_name)
            raise _still_creating_error(
                sprite_name, self._create_config, waited=wait,
                persistent=self._persistent,
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
        try:
            return self._get_sprite_within(sprite_name, self._recovery_deadline)
        except Exception:
            # Created, but this attempt cannot use it.
            self._warn_ephemeral_left_behind(sprite_name)
            raise

    def _warn_ephemeral_left_behind(self, sprite_name: str) -> None:
        """Name an ephemeral Sprite this run may leave behind."""
        if not self._persistent:
            # Ephemeral mode never adopts, so if the platform finishes this
            # unique name nothing will use or delete it.
            logger.warning(
                "Sprites: ephemeral sprite %s may exist or appear later without "
                "this run using it; delete it by hand", sprite_name,
            )

    def _get_sprite_within(self, sprite_name: str, deadline: float):
        """``get_sprite``, cut off at ``deadline`` (monotonic) with create settings.

        Raises the retryable ``EnvironmentConnectionError`` if the API has
        not answered by then. With no create settings this is the plain SDK
        call, unchanged.
        """
        if not self._create_config:
            return self._client.get_sprite(sprite_name)
        seconds = deadline - time.monotonic()
        try:
            sprite = self._bounded_call(
                seconds, lambda client: client.get_sprite(sprite_name))
        except _CallExpired:
            raise EnvironmentConnectionError(
                f"Sprite {sprite_name}: the Sprites API did not answer before "
                "the tool deadline",
                retry_hint="This is not a command failure. Retry the same command shortly.",
            ) from None
        # The per-call client is closed; run the Sprite on this
        # environment's client like any other.
        sprite.client = self._client
        return sprite

    def _bounded_call(self, seconds: float, fn):
        """``fn(client)`` on a fresh SDK client, cut off at ``seconds``.

        The client is closed afterwards in every case. On a cut-off that
        ends the abandoned request at its next read, or at the latest when
        its own per-read timeout (also ``seconds``) fires, so retries do not
        pile up open requests.
        """
        from sprites import SpritesClient
        client = SpritesClient(
            token=self._client.token,
            base_url=self._client.base_url,
            timeout=max(1.0, seconds),
        )
        try:
            return _call_within(seconds, fn, client)
        finally:
            client.close()

    def _apply_home_fallback(self) -> None:
        """Handle a Sprite whose commands run with ``$HOME`` unset.

        Imported images can run commands as root with no ``HOME``. When the
        user is root, use ``/root`` as the sync target and pass ``HOME`` on
        every later exec so ``~`` resolves there. Otherwise keep the
        ``/root`` fallback without changing exec.
        """
        try:
            cmd = self._sprite.command("sh", "-c", "id -u", timeout=15)
            uid = cmd.combined_output().decode().strip()
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
        """Upload a single file with ``fs/write``.

        ``mkdir_parents=True`` makes the write create missing parent
        directories, so no separate mkdir (which needs ``fs/list``, absent on
        the Sprites 2.0 runtime) is made. Mode and parent creation are the
        sprites-py defaults the plugin has always sent.
        """
        data = Path(host_path).read_bytes()
        remote = self._fs / remote_path
        remote.write_bytes(data, mode=0o644, mkdir_parents=True)

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
        from sprites.exceptions import ExitError

        for batch in _delete_batches(remote_paths):
            script = _delete_script(batch)
            cmd = self._sprite.command(
                "sh", "-c", script, timeout=_DELETE_TIMEOUT,
                **self._command_kwargs(),
            )
            try:
                cmd.combined_output()
            except ExitError as e:
                output = ((e.stdout or b"") + (e.stderr or b"")).decode(
                    "utf-8", errors="replace"
                ).strip()
                code = e.exit_code() if callable(getattr(e, "exit_code", None)) else "?"
                raise RuntimeError(
                    f"Sprites: deleting {len(batch)} synced file(s) failed "
                    f"(exit {code}): {output}"
                ) from e

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

        def exec_fn() -> tuple[str, int]:
            cmd = sprite.command(*shell_cmd, timeout=cmd_timeout, **cmd_kwargs)
            try:
                output = cmd.combined_output()
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
                    self._sprite.delete()
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
