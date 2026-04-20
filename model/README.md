# Wakeword Models

This directory now uses `lightwake` as the primary wakeword lane for firmware + app integration.

## Active lane

- `wakewords.manifest`: source of truth for protocol/audio defaults and the `.lww` model path
- `lightwake/`: self-written Rust crate for a small template-based wakeword detector
- `lightwake/voice_commands.lww`: binary model consumed by the firmware bridge

`lightwake/`:

- reads mono 16-bit PCM WAV
- builds a compact `.lww` binary model
- detects by cosine similarity on a fixed-size embedding

See [lightwake/README.md](F:/Code/NCKH_EE_2026/model/lightwake/README.md) for commands.
