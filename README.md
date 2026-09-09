# hermes-plugin-sprites

Sprites terminal backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — run agent shell commands in [Sprites](https://sprites.dev), stateful cloud sandboxes on Fly.io with checkpoint & restore.

Built on the pluggable terminal-backend extension point (hermes-agent PR #94400): this plugin registers `terminal.backend: sprites` as a first-class backend across dispatch, the setup wizard, `hermes status` / `hermes doctor`, the dashboard picker, approval policy, path handling, and subprocess secret stripping — with zero changes to hermes-agent core.

## Hermes compatibility

The offline suite is tested against Hermes `v2026.8.31` and `v2026.9.7`.
The plugin supports both the original environment module layout and the
September 2026 split. It uses the new process-handle module when available,
falling back only when that module is absent in an older release.
The SDK must be installed into Hermes's Python environment; this standalone
plugin does not use Hermes's built-in lazy-dependency allowlist.

`hermes doctor` checks token and SDK availability; it does not execute a command
or establish end-to-end backend compatibility. The offline tests below load the
plugin through Hermes discovery and exercise real terminal dispatch with a
mocked Sprites SDK, including persistent and ephemeral cleanup. Live testing is
separate and creates billable resources.

## Install

```bash
# 1. Copy/clone this repo into the Hermes plugins dir
git clone git@github.com:NousResearch/hermes-plugin-sprites.git ~/.hermes/plugins/sprites

# 2. Install the SDK using Hermes's Python environment, not an unrelated Python
pip install 'sprites-py>=0.5.0,<0.6'

# 3. Enable + select
hermes plugins enable sprites
hermes config set terminal.backend sprites

# 4. Use your editor to save SPRITES_TOKEN in the selected profile's .env.
#    Create the token in your organization's Sprites token settings in
#    the Fly.io dashboard. `sprite login` alone does not supply it to Hermes.
chmod 600 ~/.hermes/.env
```

`hermes doctor` reports token/SDK status once `terminal.backend: sprites` is active.
For a named profile, install the plugin under that profile's `plugins/sprites`
directory and use `hermes -p <profile>` for enable/config/chat commands. Keep its
token in that profile's `.env` rather than the default profile's file.

## Behavior

- **Persistent by default.** Each Sprite outlives the session and is resumed via a deterministic, profile-scoped name (`hermes-{task}` on the default profile, `hermes-{display}-{digest12}` on named profiles). Its ext4 filesystem is the authoritative store — no sync-back.
- **Profile isolation is fail-closed.** Identity digests are collision-resistant across component boundaries and DNS-bounded; profile-resolution failure refuses to fall back to the default profile's Sprite.
- **Ephemeral mode** (`terminal.container_persistent: false`): run-unique Sprite names (never adopted, deleted on cleanup), with per-session sandbox identities so two ephemeral runs can never attach the same live VM.
- **Client-side exec timeouts.** SDK calls without a positive timeout use a 3600s client timeout. This is not a guarantee that the remote process stops at that instant; the plugin has no explicit kill hook for an in-flight command.
- **Sharing model:** gateway/WebUI sessions each get their own Sprite; `delegate_task` children share their parent's; key-less flows (CLI, cron) share the profile-default Sprite.
- `SPRITES_TOKEN` / `SPRITE_TOKEN` are stripped from every subprocess the agent spawns.
- Hermes's remote-file sync uploads selected credential files, skills, and cache
  files. Review the profile's contents; token environment stripping does not
  mean no credentials leave the host. Local repositories are not automatically
  uploaded, and remote edits are not synced back to the host.

## Tests

```bash
# Unit (no token needed): use a Hermes checkout with its test dependencies installed.
# Run from that checkout, substituting the absolute path to this plugin:
cd /path/to/hermes-agent
scripts/run_tests.sh /path/to/hermes-plugin-sprites/tests/test_sprites_environment.py
```

Repeat against each Hermes release being supported. Keep the real discovery and
terminal-dispatch checks enabled: a registration-only test will miss an import
failure that happens when the first environment is created.

### Opt-in live tests

The live suite requires Hermes `v2026.9.7`, its test dependencies, and the real
SDK (`sprites-py` `0.5.1` was used for validation). It exercises plugin discovery
and terminal dispatch, stdout/stderr and exit codes, shell state, skill-file
upload/deletion, persistence, profile isolation, timeout/interrupt results, and
ephemeral cleanup. It does not call an LLM.

**This creates billable resources and deletes their files.** Use an appropriate
test organization. Put its Sprites token in a private file (`chmod 600`) using
your editor, outside source control. The file's explicit path enables the test;
without it, the test skips before importing the SDK or making API calls.

```bash
cd /path/to/hermes-agent
scripts/run_tests.sh /path/to/hermes-plugin-sprites/tests/test_sprites_terminal_live.py \
  --file-timeout 600 --file-retries 0 -- \
  --sprites-live-token-file /private/path/to/test-token
```

The test isolates Hermes's home, installs the plugin into temporary profiles,
and permits only its run-unique Sprite names. It refuses to adopt existing
Sprites, adds a unique ownership label at creation, checks identity before
deleting, and verifies absence after cleanup. Persistent test Sprites are also
removed. Review cleanup output after failures; forcibly killing the test process
can prevent teardown and leave billable resources behind. Remove your temporary
token file when finished.

Timeout and interrupt checks use short, bounded remote workloads. Live testing
observed both workloads continue remotely after the client returned; neither
result should be treated as confirmation of a remote kill.

## Attribution

The Sprites environment was authored by **Kyle McLaren** ([@kylemclaren](https://github.com/kylemclaren), Fly.io) as hermes-agent [PR #30112](https://github.com/NousResearch/hermes-agent/pull/30112) and hardened through three rounds of review in [PR #93523](https://github.com/NousResearch/hermes-agent/pull/93523) (identity digests, DNS bounds, fail-closed profile resolution, race-safe first-use create, bounded deadlines, sandboxed live test suite). Extracted to this standalone plugin repo per the hermes-agent policy that third-party service integrations ship as plugins rather than core code.

## License

MIT
