use std::env;
use std::fs;
use std::path::PathBuf;

fn manifest_path() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"))
        .join("../../../model/wakewords.manifest")
}

fn model_root() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing CARGO_MANIFEST_DIR"))
        .join("../../../model")
}

fn main() {
    let manifest_path = manifest_path();
    println!("cargo:rerun-if-changed={}", manifest_path.display());

    let manifest_source = fs::read_to_string(&manifest_path).unwrap_or_else(|error| {
        panic!(
            "failed to read manifest {}: {}",
            manifest_path.display(),
            error
        )
    });

    let mut protocol_version = 1u32;
    let mut keyword_set_version = 1u32;
    let mut audio_sample_rate = 16_000u32;
    let mut audio_channels = 1u16;
    let mut bits_per_sample = 16u16;
    let mut frame_ms = 30u32;
    let mut engine_name = String::from("lightwake");
    let mut model_file = String::from("lightwake/voice_commands.lww");

    for raw_line in manifest_source.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(value) = line.strip_prefix("protocol_version=") {
            protocol_version = value.parse().expect("invalid protocol_version");
        } else if let Some(value) = line.strip_prefix("keyword_set_version=") {
            keyword_set_version = value.parse().expect("invalid keyword_set_version");
        } else if let Some(value) = line.strip_prefix("audio_sample_rate=") {
            audio_sample_rate = value.parse().expect("invalid audio_sample_rate");
        } else if let Some(value) = line.strip_prefix("audio_channels=") {
            audio_channels = value.parse().expect("invalid audio_channels");
        } else if let Some(value) = line.strip_prefix("bits_per_sample=") {
            bits_per_sample = value.parse().expect("invalid bits_per_sample");
        } else if let Some(value) = line.strip_prefix("frame_ms=") {
            frame_ms = value.parse().expect("invalid frame_ms");
        } else if let Some(value) = line.strip_prefix("engine=") {
            engine_name = value.to_string();
        } else if let Some(value) = line.strip_prefix("lightwake_model=") {
            model_file = value.to_string();
        }
    }

    let model_path = model_root().join(&model_file);
    println!("cargo:rerun-if-changed={}", model_path.display());

    let mut generated_source = String::new();
    generated_source.push_str("pub const PROTOCOL_VERSION: u32 = ");
    generated_source.push_str(&protocol_version.to_string());
    generated_source.push_str(";\n");
    generated_source.push_str("pub const KEYWORD_SET_VERSION: u32 = ");
    generated_source.push_str(&keyword_set_version.to_string());
    generated_source.push_str(";\n");
    generated_source.push_str("pub const SAMPLE_RATE_FALLBACK: u32 = ");
    generated_source.push_str(&audio_sample_rate.to_string());
    generated_source.push_str(";\n");
    generated_source.push_str("pub const CHANNELS: u16 = ");
    generated_source.push_str(&audio_channels.to_string());
    generated_source.push_str(";\n");
    generated_source.push_str("pub const BITS_PER_SAMPLE: u16 = ");
    generated_source.push_str(&bits_per_sample.to_string());
    generated_source.push_str(";\n");
    generated_source.push_str("pub const FRAME_MS_FALLBACK: u32 = ");
    generated_source.push_str(&frame_ms.to_string());
    generated_source.push_str(";\n");
    generated_source.push_str("pub const ENGINE_NAME: &str = \"");
    generated_source.push_str(&engine_name);
    generated_source.push_str("\";\n");

    if model_path.exists() {
        generated_source.push_str("pub const MODEL_BYTES: &[u8] = include_bytes!(\"");
        generated_source.push_str(&model_path.to_string_lossy().replace('\\', "\\\\"));
        generated_source.push_str("\");\n");
    } else {
        println!(
            "cargo:warning=lightwake model file missing and detector will boot in not-ready mode: {}",
            model_path.display()
        );
        generated_source.push_str("pub const MODEL_BYTES: &[u8] = &[];\n");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing OUT_DIR"));
    fs::write(out_dir.join("generated_lightwake.rs"), generated_source)
        .expect("failed to write generated lightwake source");
}
