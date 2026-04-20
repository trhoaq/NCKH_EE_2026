# lightwake

`lightwake` is a very small Rust wakeword detector based on MFCC templates.

It is designed for low-resource targets where a full speech pipeline is too heavy:

- no external dependencies
- fixed-size MFCC embeddings
- binary model format
- simple MFCC distance inference

## Commands

Build a model from directories of WAV files:

```powershell
cargo run --manifest-path .\model\lightwake\Cargo.toml --bin lightwake-cli -- `
  build `
  --keyword on --positive .\model\data\on `
  --keyword off --positive .\model\data\off `
  --negative .\model\data\negative `
  --output .\model\lightwake\voice_commands.lww
```

`build` now auto-normalizes every input WAV with `ffmpeg` before training:

- mono
- 16 kHz
- PCM 16-bit little-endian

`detect` cũng sẽ tự normalize nếu WAV đầu vào không phải PCM 16-bit mono 16 kHz.

Ngoài bước convert, pipeline nội bộ hiện có thêm:

- energy-based VAD / silence trimming trước khi trích đặc trưng
- DC offset removal
- soft noise gate
- RMS/peak stabilization
- cepstral mean/variance normalization
- delta + delta-delta features
- nhiều template cho mỗi keyword thay vì một centroid duy nhất
- threshold theo percentile của positive / hard-negative distances
- augmentation đơn giản cho positive sample trong lúc build
- hard-negative selection trong lúc build model

So bạn không cần gọi thêm lệnh convert riêng nữa, nhưng `ffmpeg` phải có trên `PATH`.

Run detection on a WAV file:

```powershell
cargo run --manifest-path .\model\lightwake\Cargo.toml --bin lightwake-cli -- `
  detect `
  --model .\model\lightwake\voice_commands.lww `
  --input .\model\samples\test_on.wav
```

Nếu bạn đổi pipeline preprocessing hoặc thresholding, hãy rebuild lại `.lww` để model khớp detector mới.

Convert mixed WAV formats with `ffmpeg` and train in one step:

```powershell
.\model\lightwake\train_with_ffmpeg.ps1 `
  -Keyword on `
  -PositiveDir .\model\data\data_on\positive `
  -NegativeDir .\model\data\data_on\negative `
  -OutputModel .\model\lightwake\on_only.lww
```

## WAV requirements

- PCM 16-bit little-endian
- mono
- recommended sample rate: 16000 Hz
- short utterances, around 1-2 seconds
