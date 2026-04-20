use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use lightwake::LightwakeError;

pub struct NormalizedDataset {
    pub positive_dirs: Vec<PathBuf>,
    pub negative_dir: PathBuf,
    temp_root: PathBuf,
}

impl Drop for NormalizedDataset {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.temp_root);
    }
}

pub struct NormalizedInput {
    wav_path: PathBuf,
    temp_root: PathBuf,
}

impl NormalizedInput {
    pub fn wav_path(&self) -> &Path {
        &self.wav_path
    }
}

impl Drop for NormalizedInput {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.temp_root);
    }
}

pub fn normalize_dataset(
    positive_specs: &[(String, PathBuf)],
    negative_dir: &Path,
    sample_rate: u32,
) -> Result<NormalizedDataset, LightwakeError> {
    ensure_ffmpeg_available()?;

    let temp_root = build_temp_root()?;
    fs::create_dir_all(&temp_root)?;

    let mut normalized_positive_dirs = Vec::<PathBuf>::new();
    for (keyword, source_dir) in positive_specs {
        let destination_dir = temp_root.join(format!("positive_{}", sanitize_name(keyword)));
        normalize_directory(source_dir, &destination_dir, sample_rate)?;
        normalized_positive_dirs.push(destination_dir);
    }

    let normalized_negative_dir = temp_root.join("negative");
    normalize_directory(negative_dir, &normalized_negative_dir, sample_rate)?;

    Ok(NormalizedDataset {
        positive_dirs: normalized_positive_dirs,
        negative_dir: normalized_negative_dir,
        temp_root,
    })
}

pub fn normalize_input_wav(
    source_path: &Path,
    sample_rate: u32,
) -> Result<NormalizedInput, LightwakeError> {
    ensure_ffmpeg_available()?;

    let temp_root = build_temp_root()?;
    fs::create_dir_all(&temp_root)?;

    let destination_path = temp_root.join("normalized_input.wav");
    normalize_file(source_path, &destination_path, sample_rate)?;

    Ok(NormalizedInput {
        wav_path: destination_path,
        temp_root,
    })
}

fn ensure_ffmpeg_available() -> Result<(), LightwakeError> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map_err(|error| {
            LightwakeError::InvalidArgument(format!(
                "ffmpeg is required for auto-normalization but could not be launched: {error}"
            ))
        })?;

    if output.status.success() {
        Ok(())
    } else {
        Err(LightwakeError::InvalidArgument(
            "ffmpeg is installed but returned a non-zero status for `ffmpeg -version`".to_string(),
        ))
    }
}

fn build_temp_root() -> Result<PathBuf, LightwakeError> {
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| {
            LightwakeError::InvalidArgument("system clock is before unix epoch".to_string())
        })?
        .as_millis();
    Ok(std::env::temp_dir().join(format!("lightwake-normalized-{unix_ms}")))
}

fn normalize_directory(
    source_dir: &Path,
    destination_dir: &Path,
    sample_rate: u32,
) -> Result<(), LightwakeError> {
    let files = collect_wav_files(source_dir)?;
    fs::create_dir_all(destination_dir)?;

    for source_path in files {
        let filename = source_path.file_name().ok_or_else(|| {
            LightwakeError::InvalidArgument(format!(
                "source file has no name: {}",
                source_path.display()
            ))
        })?;
        let destination_path = destination_dir.join(filename);
        normalize_file(&source_path, &destination_path, sample_rate)?;
    }

    Ok(())
}

fn collect_wav_files(source_dir: &Path) -> Result<Vec<PathBuf>, LightwakeError> {
    let mut paths = fs::read_dir(source_dir)?
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| {
            path.extension()
                .map(|extension| extension.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();

    if paths.is_empty() {
        return Err(LightwakeError::InvalidArgument(format!(
            "no .wav files found in {}",
            source_dir.display()
        )));
    }

    Ok(paths)
}

fn normalize_file(
    source_path: &Path,
    destination_path: &Path,
    sample_rate: u32,
) -> Result<(), LightwakeError> {
    let status = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(source_path)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(sample_rate.to_string())
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(destination_path)
        .status()
        .map_err(|error| {
            LightwakeError::InvalidArgument(format!(
                "failed to run ffmpeg for {}: {error}",
                source_path.display()
            ))
        })?;

    if status.success() {
        Ok(())
    } else {
        Err(LightwakeError::InvalidArgument(format!(
            "ffmpeg failed while converting {}",
            source_path.display()
        )))
    }
}

fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
