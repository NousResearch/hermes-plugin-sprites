# hermes-plugin-sprites

Sprites terminal backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — run agent shell commands in [Sprites](https://sprites.dev), stateful cloud sandboxes on Fly.io with checkpoint & restore.

Built on the pluggable terminal-backend extension point (hermes-agent PR #94400): this plugin registers `terminal.backend: sprites` as a first-class backend across dispatch, the setup wizard, `hermes status` / `hermes doctor`, the dashboard picker, approval policy, path handling, and subprocess secret stripping — with zero changes to hermes-agent core.

## Install

```bash
# 1. Copy/clone this repo into the Hermes plugins dir
git clone git@github.com:NousResearch/hermes-plugin-sprites.git ~/.hermes/plugins/sprites

# 2. Install the SDK using Hermes's Python environment, not an unrelated Python
pip install 'sprites-py>=0.5.0,<0.6'

# 3. Enable + select
hermes plugins enable sprites
hermes config set terminal.backend sprites

# 4. Create a token in your organization's Sprites settings in the Fly.io
#    dashboard, then save SPRITES_TOKEN=... in ~/.hermes/.env using your editor.
#    `sprite login` alone does not supply the token to Hermes.
chmod 600 ~/.hermes/.env
```

`hermes doctor` checks token/SDK availability, not command execution. Compatibility
is tested against Hermes `v2026.8.31`, `v2026.9.7` and `v0.21.7` using real plugin
discovery and terminal dispatch with mocked SDK transport.

## Behavior

- **Persistent by default.** Each Sprite outlives the session and is resumed via a deterministic, profile-scoped name (`hermes-{task}` on the default profile, `hermes-{display}-{digest12}` on named profiles). Its ext4 filesystem is the authoritative store — no sync-back.
- **Profile isolation is fail-closed.** Identity digests are collision-resistant across component boundaries and DNS-bounded; profile-resolution failure refuses to fall back to the default profile's Sprite.
- **Ephemeral mode** (`terminal.container_persistent: false`): run-unique Sprite names (never adopted, deleted on cleanup), with per-session sandbox identities so two ephemeral runs can never attach the same live VM.
- **Client-side exec timeouts.** SDK calls without a positive timeout use a 3600s client timeout. This is not a guarantee that the remote process stops at that instant; the plugin has no explicit kill hook for an in-flight command.
- **Sharing model:** gateway/WebUI sessions each get their own Sprite; `delegate_task` children share their parent's; key-less flows (CLI, cron) share the profile-default Sprite.
- `SPRITES_TOKEN` / `SPRITE_TOKEN` are stripped from every subprocess the agent spawns.

## Settings

Optional settings live in the plugin's own namespace in `config.yaml`,
`plugins.entries.sprites.settings` (they also appear in the Plugins hub). With
none set, behaviour, Sprite names and API requests are exactly as described
above.

| Key | Values | Effect |
|---|---|---|
| `name_prefix` | lowercase letters, digits and single hyphens; max 40 chars | Every Sprite name starts with `<prefix>-` instead of `hermes-`: `<prefix>-{task}` on the default profile, `<prefix>-{display}-{digest12}` on named profiles, `<prefix>-eph-{task}-{nonce}` for ephemeral runs. The prefix is never truncated, and the plugin refuses to create or resume any Sprite outside it. |
| `image` | OCI reference with an explicit tag (not `latest`) or a digest | Image for Sprites this plugin creates. Prefer a digest pin (`name@sha256:...`) so every new Sprite gets the same bytes. Unset: the Sprites base image (Ubuntu, `sprite` user, sudo). |
| `max_cpus` | 1 to 128 | vCPUs for newly created Sprites. Unset: Sprites default. |
| `max_ram_gb` | 1 to 1024 | Memory (GB) for newly created Sprites. Unset: Sprites default. |
| `max_disk_gb` | 1 to 1024, no smaller than the image | Root filesystem (GB) for newly created Sprites. Unset: Sprites default. |
| `scope` | `session` (default) \| `instance` | `instance` makes every task id on a profile (gateway sessions, cron jobs, `delegate_task` children, CLI) resume the same persistent Sprite: `<prefix>-default` (or `hermes-default` without a prefix) on the default profile, the usual digest-scoped `-default` name on a named profile. Each session still gets its own shell; they share the VM and filesystem (file sync into the Sprite is serialised across sessions in one process). |

An invalid value for either key is an error, not a fallback: the backend
refuses to start and `hermes doctor` shows the reason on its "Sprites naming"
row. The same happens when config.yaml itself cannot be parsed, because Hermes
then runs on a fallback copy of the config that may not contain the settings
you just wrote. Ignoring a bad prefix would create Sprites outside the namespace you meant
to confine this install to; ignoring a misspelt scope would quietly start
creating one Sprite per session.

`image`, `max_cpus`, `max_ram_gb` and `max_disk_gb` apply only when the plugin
creates a Sprite. An existing Sprite is resumed as it is; the plugin never
updates its config (on Sprites 2.0 the image and disk size are fixed at
creation). To move to a new image or size, delete the Sprite (or change
`name_prefix`) so the next session creates a fresh one. Creating from an image
Sprites has not pulled before can take a minute or more. The plugin waits for
that create up to 2 minutes less than Hermes's tool-call deadline (300 s under
the default 420 s `timeouts.tools.sequential_call` / `concurrent_batch`; 600 s
when both are disabled). If the create is still running then, the command
returns a retryable "still being created" result; retry it shortly and it uses
the Sprite once Sprites has finished creating it (an ephemeral run instead
deletes the unfinished Sprite and the retry creates a new one). Most non-base images run commands as root, and some
leave `$HOME` unset: the plugin then uses `/root` as the home directory and
passes `HOME=/root` to its commands.

```bash
hermes config set plugins.entries.sprites.settings.image 'nousresearch/hermes-sandbox@sha256:<digest>'
hermes config set plugins.entries.sprites.settings.max_cpus 2
hermes config set plugins.entries.sprites.settings.max_ram_gb 4
hermes config set plugins.entries.sprites.settings.max_disk_gb 20
```

`scope` only affects persistent Sprites. Ephemeral mode
(`terminal.container_persistent: false`) keeps its run-unique names under either
scope, so two runs never share a VM.

### One Sprite per hosted instance

When several Hermes installs share one Sprites organization (for example one
hosted instance per customer), give each install its own prefix and the
`instance` scope, so each install owns exactly one Sprite and can never resume
another install's:

```bash
hermes config set plugins.entries.sprites.settings.name_prefix hc-<instance-id>
hermes config set plugins.entries.sprites.settings.scope instance
```

That install's only Sprite is then `hc-<instance-id>-default`, so its
lifecycle (resume after a restart, delete when the instance is deleted) is one
deterministic name. Notes:

- Use a prefix like `hc-<id>`. `hermes config set` parses YAML scalars, so
  values such as `123`, `1e3`, `true` or `off` are stored as numbers/booleans
  (the plugin rejects them), and `null` / `none` clear the setting (the
  default `hermes-*` names come back).
- A prefix is a plain string prefix, so in `session` scope install `hc-a` can
  produce a name inside another install's `hc-a-b-` namespace (task id
  `b-...`). Use fixed-length ids, or `instance` scope (whose only name is
  `<prefix>-default`), to keep installs disjoint.
- The prefix confines what this plugin does; it is not an access control. The
  `SPRITES_TOKEN` on the machine decides what the machine can reach, so scope
  the token to the same prefix where Sprites lets you.
- Changing either setting changes which Sprite is resumed. The previous
  Sprite is not deleted; remove it with the Sprites CLI or API if you no longer
  need its files.

## Sprites 2.0 runtime

Sprites now run on the 2.0 runtime, behind the same `https://api.sprites.dev/v1`
paths. Of the filesystem API it keeps only `fs/write`, which creates missing
parent directories itself. The plugin's file sync (skills, credential files and
caches copied into `~/.hermes` on the Sprite before each command) therefore:

- uploads each file with one `fs/write` (`mode=0644`, `mkdirParents=true`, the
  same request as before, minus the separate directory check that needed
  `fs/list`);
- deletes files that disappeared on the host with one exec per batch,
  `rm -f -- <paths>`, retried as `sudo -n rm -f -- <paths>` because `fs/write`
  creates root-owned files while the base image runs commands as `sprite`. A
  missing file is fine; any other failure is retried on the next sync.

## Tests

```bash
# Unit (no token needed): use a Hermes checkout with its test dependencies installed.
# Run from that checkout, substituting the absolute path to this plugin:
cd /path/to/hermes-agent
scripts/run_tests.sh /path/to/hermes-plugin-sprites/tests/test_sprites_environment.py
```

Through `scripts/run_tests.sh`, the tests import Hermes from the checkout you
run them in (the runner clears the environment, so `HERMES_AGENT_REPO` does not
reach them). When invoking pytest some other way, `HERMES_AGENT_REPO` selects
the checkout if set; otherwise the current directory is used if it is a Hermes
checkout, then `~/.hermes/hermes-agent`.
Keep this plugin's clone outside the Hermes checkout and outside any other
clone of this repo, or pytest also loads that other copy's `conftest.py`.

The legacy `tests/test_sprites_terminal_live.py` still assumes the pre-plugin
Hermes layout and is not a supported standalone live-test recipe. Its rewrite
is deferred; do not run it against your normal Hermes profile.

## Attribution

The Sprites environment was authored by **Kyle McLaren** ([@kylemclaren](https://github.com/kylemclaren), Fly.io) as hermes-agent [PR #30112](https://github.com/NousResearch/hermes-agent/pull/30112) and hardened through three rounds of review in [PR #93523](https://github.com/NousResearch/hermes-agent/pull/93523) (identity digests, DNS bounds, fail-closed profile resolution, race-safe first-use create, bounded deadlines, sandboxed live test suite). Extracted to this standalone plugin repo per the hermes-agent policy that third-party service integrations ship as plugins rather than core code.

## License

MIT
