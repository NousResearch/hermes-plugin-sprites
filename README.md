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
is tested against Hermes `v2026.8.31` and `v2026.9.7` using real plugin discovery
and terminal dispatch with mocked SDK transport.

## Behavior

- **Persistent by default.** Each Sprite outlives the session and is resumed via a deterministic, profile-scoped name (`hermes-{task}` on the default profile, `hermes-{display}-{digest12}` on named profiles). Its ext4 filesystem is the authoritative store — no sync-back.
- **Profile isolation is fail-closed.** Identity digests are collision-resistant across component boundaries and DNS-bounded; profile-resolution failure refuses to fall back to the default profile's Sprite.
- **Ephemeral mode** (`terminal.container_persistent: false`): run-unique Sprite names (never adopted, deleted on cleanup), with per-session sandbox identities so two ephemeral runs can never attach the same live VM.
- **Client-side exec timeouts.** SDK calls without a positive timeout use a 3600s client timeout. This is not a guarantee that the remote process stops at that instant; the plugin has no explicit kill hook for an in-flight command.
- **Sharing model:** gateway/WebUI sessions each get their own Sprite; `delegate_task` children share their parent's; key-less flows (CLI, cron) share the profile-default Sprite.
- `SPRITES_TOKEN` / `SPRITE_TOKEN` are stripped from every subprocess the agent spawns.

## Tests

```bash
# Unit (no token needed): use a Hermes checkout with its test dependencies installed.
# Run from that checkout, substituting the absolute path to this plugin:
cd /path/to/hermes-agent
scripts/run_tests.sh /path/to/hermes-plugin-sprites/tests/test_sprites_environment.py
```

The legacy `tests/test_sprites_terminal_live.py` still assumes the pre-plugin
Hermes layout and is not a supported standalone live-test recipe. Its rewrite
is deferred; do not run it against your normal Hermes profile.

## Attribution

The Sprites environment was authored by **Kyle McLaren** ([@kylemclaren](https://github.com/kylemclaren), Fly.io) as hermes-agent [PR #30112](https://github.com/NousResearch/hermes-agent/pull/30112) and hardened through three rounds of review in [PR #93523](https://github.com/NousResearch/hermes-agent/pull/93523) (identity digests, DNS bounds, fail-closed profile resolution, race-safe first-use create, bounded deadlines, sandboxed live test suite). Extracted to this standalone plugin repo per the hermes-agent policy that third-party service integrations ship as plugins rather than core code.

## License

MIT
