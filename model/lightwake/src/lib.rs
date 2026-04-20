use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

const MODEL_MAGIC: [u8; 4] = *b"LWW1";
const DEFAULT_SAMPLE_RATE: u32 = 16_000;
const TRANSPORT_FRAME_SIZE: usize = 320;
const MFCC_FRAME_SIZE: usize = 256;
const MFCC_HOP_SIZE: usize = 128;
const MEL_FILTER_COUNT: usize = 20;
const MFCC_COEFF_COUNT: usize = 8;
const TARGET_STEPS: usize = 16;
const EMBEDDING_DIM: usize = TARGET_STEPS * MFCC_COEFF_COUNT;

#[derive(Debug)]
pub enum LightwakeError {
    Io(std::io::Error),
    InvalidArgument(String),
    InvalidWav(String),
    InvalidModel(String),
}

impl Display for LightwakeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LightwakeError::Io(error) => write!(f, "io error: {error}"),
            LightwakeError::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            LightwakeError::InvalidWav(message) => write!(f, "invalid wav: {message}"),
            LightwakeError::InvalidModel(message) => write!(f, "invalid model: {message}"),
        }
    }
}

impl std::error::Error for LightwakeError {}

impl From<std::io::Error> for LightwakeError {
    fn from(value: std::io::Error) -> Self {
        LightwakeError::Io(value)
    }
}

#[derive(Clone, Debug)]
pub struct Example {
    pub keyword: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct BuildConfig {
    pub sample_rate: u32,
    pub threshold_margin: f32,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            threshold_margin: 0.08,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KeywordTemplate {
    pub keyword: String,
    pub action_id: u8,
    pub threshold: f32,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct LightwakeModel {
    pub sample_rate: u32,
    pub frame_size: u16,
    pub target_steps: u16,
    pub templates: Vec<KeywordTemplate>,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub keyword: String,
    pub action_id: u8,
    pub score: f32,
}

impl LightwakeModel {
    pub fn detect_pcm(&self, samples: &[i16], sample_rate: u32) -> Option<Detection> {
        let resampled = if sample_rate == self.sample_rate {
            samples.to_vec()
        } else {
            resample_linear(samples, sample_rate, self.sample_rate)
        };

        let embedding = compute_mfcc_embedding(&resampled, self.sample_rate);
        let mut best: Option<Detection> = None;

        for template in &self.templates {
            let distance = euclidean_distance(&embedding, &template.embedding);
            if distance <= template.threshold {
                let score = (1.0 - (distance / (template.threshold + 1e-6))).clamp(0.0, 1.0);
                match &best {
                    Some(existing) if existing.score >= score => {}
                    _ => {
                        best = Some(Detection {
                            keyword: template.keyword.clone(),
                            action_id: template.action_id,
                            score,
                        });
                    }
                }
            }
        }

        best
    }

    pub fn save(&self, path: &Path) -> Result<(), LightwakeError> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend_from_slice(&MODEL_MAGIC);
        buffer.extend_from_slice(&self.sample_rate.to_le_bytes());
        buffer.extend_from_slice(&self.frame_size.to_le_bytes());
        buffer.extend_from_slice(&self.target_steps.to_le_bytes());
        buffer.extend_from_slice(&(self.templates.len() as u16).to_le_bytes());

        for template in &self.templates {
            let keyword_bytes = template.keyword.as_bytes();
            if keyword_bytes.len() > u8::MAX as usize {
                return Err(LightwakeError::InvalidModel(
                    "keyword too long for binary format".to_string(),
                ));
            }
            buffer.push(keyword_bytes.len() as u8);
            buffer.extend_from_slice(keyword_bytes);
            buffer.push(template.action_id);
            buffer.extend_from_slice(&template.threshold.to_le_bytes());
            buffer.extend_from_slice(&(template.embedding.len() as u16).to_le_bytes());
            for value in &template.embedding {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        }

        fs::write(path, buffer)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, LightwakeError> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LightwakeError> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader.take_exact(4)?;
        if magic != MODEL_MAGIC {
            return Err(LightwakeError::InvalidModel("bad magic".to_string()));
        }

        let sample_rate = reader.read_u32()?;
        let frame_size = reader.read_u16()?;
        let target_steps = reader.read_u16()?;
        let template_count = reader.read_u16()? as usize;
        let mut templates = Vec::with_capacity(template_count);

        for _ in 0..template_count {
            let keyword_len = reader.read_u8()? as usize;
            let keyword =
                String::from_utf8(reader.take_exact(keyword_len)?.to_vec()).map_err(|_| {
                    LightwakeError::InvalidModel("keyword is not valid utf8".to_string())
                })?;
            let action_id = reader.read_u8()?;
            let threshold = reader.read_f32()?;
            let embedding_len = reader.read_u16()? as usize;
            let mut embedding = Vec::with_capacity(embedding_len);
            for _ in 0..embedding_len {
                embedding.push(reader.read_f32()?);
            }

            templates.push(KeywordTemplate {
                keyword,
                action_id,
                threshold,
                embedding,
            });
        }

        Ok(Self {
            sample_rate,
            frame_size,
            target_steps,
            templates,
        })
    }
}

pub fn build_model(
    keyword_examples: &[Example],
    negative_examples: &[PathBuf],
    config: &BuildConfig,
) -> Result<LightwakeModel, LightwakeError> {
    if keyword_examples.is_empty() {
        return Err(LightwakeError::InvalidArgument(
            "at least one positive example is required".to_string(),
        ));
    }

    let mut grouped = std::collections::BTreeMap::<String, Vec<Vec<f32>>>::new();
    for example in keyword_examples {
        let wav = read_wav_i16(&example.path)?;
        let samples = if wav.sample_rate == config.sample_rate {
            wav.samples
        } else {
            resample_linear(&wav.samples, wav.sample_rate, config.sample_rate)
        };
        grouped
            .entry(example.keyword.clone())
            .or_default()
            .push(compute_mfcc_embedding(&samples, config.sample_rate));
    }

    let negative_embeddings = negative_examples
        .iter()
        .map(|path| {
            let wav = read_wav_i16(path)?;
            let samples = if wav.sample_rate == config.sample_rate {
                wav.samples
            } else {
                resample_linear(&wav.samples, wav.sample_rate, config.sample_rate)
            };
            Ok(compute_mfcc_embedding(&samples, config.sample_rate))
        })
        .collect::<Result<Vec<_>, LightwakeError>>()?;

    let mut templates = Vec::<KeywordTemplate>::new();
    for (keyword, embeddings) in grouped {
        let centroid = average_embedding(&embeddings)?;
        let positive_distances = embeddings
            .iter()
            .map(|embedding| euclidean_distance(&centroid, embedding))
            .collect::<Vec<_>>();
        let negative_distances = negative_embeddings
            .iter()
            .map(|embedding| euclidean_distance(&centroid, embedding))
            .collect::<Vec<_>>();

        let max_positive = positive_distances
            .iter()
            .copied()
            .fold(0.0_f32, |accumulator, value| accumulator.max(value));
        let min_negative = if negative_distances.is_empty() {
            max_positive + config.threshold_margin + 0.15
        } else {
            negative_distances
                .iter()
                .copied()
                .fold(f32::MAX, |accumulator, value| accumulator.min(value))
        };

        let midpoint = (max_positive + min_negative) * 0.5;
        let threshold = (midpoint + config.threshold_margin * 0.5)
            .max(max_positive + 0.01)
            .clamp(0.05, 2.5);

        templates.push(KeywordTemplate {
            keyword: keyword.clone(),
            action_id: action_id_for_keyword(&keyword),
            threshold,
            embedding: centroid,
        });
    }

    Ok(LightwakeModel {
        sample_rate: config.sample_rate,
        frame_size: TRANSPORT_FRAME_SIZE as u16,
        target_steps: TARGET_STEPS as u16,
        templates,
    })
}

pub fn read_wav_i16(path: &Path) -> Result<WavData, LightwakeError> {
    let bytes = fs::read(path)?;
    if bytes.len() < 44 {
        return Err(LightwakeError::InvalidWav(format!(
            "{} is too small to be a pcm wav",
            path.display()
        )));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(LightwakeError::InvalidWav(format!(
            "{} is not a RIFF/WAVE file",
            path.display()
        )));
    }

    let mut offset = 12usize;
    let mut fmt_channels = None;
    let mut fmt_sample_rate = None;
    let mut fmt_bits_per_sample = None;
    let mut pcm_data = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;

        if offset + chunk_size > bytes.len() {
            return Err(LightwakeError::InvalidWav(format!(
                "{} has a truncated chunk",
                path.display()
            )));
        }

        let chunk = &bytes[offset..offset + chunk_size];
        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err(LightwakeError::InvalidWav(
                    "fmt chunk is too small".to_string(),
                ));
            }
            let audio_format = u16::from_le_bytes([chunk[0], chunk[1]]);
            let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
            let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let bits_per_sample = u16::from_le_bytes([chunk[14], chunk[15]]);
            if audio_format != 1 {
                return Err(LightwakeError::InvalidWav(
                    "only PCM wav files are supported".to_string(),
                ));
            }
            fmt_channels = Some(channels);
            fmt_sample_rate = Some(sample_rate);
            fmt_bits_per_sample = Some(bits_per_sample);
        } else if chunk_id == b"data" {
            pcm_data = Some(chunk.to_vec());
        }

        offset += chunk_size;
        if chunk_size % 2 == 1 {
            offset += 1;
        }
    }

    let channels =
        fmt_channels.ok_or_else(|| LightwakeError::InvalidWav("missing fmt chunk".to_string()))?;
    let sample_rate = fmt_sample_rate
        .ok_or_else(|| LightwakeError::InvalidWav("missing sample rate".to_string()))?;
    let bits_per_sample = fmt_bits_per_sample
        .ok_or_else(|| LightwakeError::InvalidWav("missing bits per sample".to_string()))?;
    let pcm_data =
        pcm_data.ok_or_else(|| LightwakeError::InvalidWav("missing data chunk".to_string()))?;

    if channels != 1 {
        return Err(LightwakeError::InvalidWav(
            "only mono wav files are supported".to_string(),
        ));
    }
    if bits_per_sample != 16 {
        return Err(LightwakeError::InvalidWav(
            "only 16-bit pcm wav files are supported".to_string(),
        ));
    }
    if pcm_data.len() % 2 != 0 {
        return Err(LightwakeError::InvalidWav(
            "pcm data length is not aligned to 16-bit samples".to_string(),
        ));
    }

    let samples = pcm_data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();

    Ok(WavData {
        sample_rate,
        samples,
    })
}

#[derive(Clone, Debug)]
pub struct WavData {
    pub sample_rate: u32,
    pub samples: Vec<i16>,
}

fn compute_mfcc_embedding(samples: &[i16], sample_rate: u32) -> Vec<f32> {
    let emphasized = pre_emphasize(samples);
    let frames = split_into_overlapping_frames(&emphasized, MFCC_FRAME_SIZE, MFCC_HOP_SIZE);
    let frame_mfcc = if frames.is_empty() {
        vec![vec![0.0; MFCC_COEFF_COUNT]]
    } else {
        let filter_bank = build_mel_filter_bank(sample_rate, MFCC_FRAME_SIZE, MEL_FILTER_COUNT);
        frames
            .iter()
            .map(|frame| mfcc_for_frame(frame, &filter_bank))
            .collect::<Vec<_>>()
    };

    let embedding = resample_feature_sequence(&frame_mfcc, TARGET_STEPS);
    debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
    embedding
}

fn pre_emphasize(samples: &[i16]) -> Vec<f32> {
    let normalized = normalize_samples(samples);
    let mut emphasized = Vec::with_capacity(normalized.len());
    let mut previous = 0.0_f32;
    for sample in normalized {
        emphasized.push(sample - 0.97 * previous);
        previous = sample;
    }
    emphasized
}

fn normalize_samples(samples: &[i16]) -> Vec<f32> {
    let peak = samples
        .iter()
        .map(|sample| sample.abs() as f32)
        .fold(1.0_f32, f32::max);
    samples
        .iter()
        .map(|sample| (*sample as f32) / peak)
        .collect::<Vec<_>>()
}

fn split_into_overlapping_frames(
    samples: &[f32],
    frame_size: usize,
    hop_size: usize,
) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut frames = Vec::<Vec<f32>>::new();
    let mut index = 0usize;
    while index < samples.len() {
        let mut frame = vec![0.0_f32; frame_size];
        let available = (samples.len() - index).min(frame_size);
        frame[..available].copy_from_slice(&samples[index..index + available]);
        frames.push(frame);

        if index + frame_size >= samples.len() {
            break;
        }
        index += hop_size.max(1);
    }
    frames
}

fn mfcc_for_frame(frame: &[f32], filter_bank: &[Vec<f32>]) -> Vec<f32> {
    let windowed = apply_hamming(frame);
    let power_spectrum = power_spectrum_naive(&windowed);
    let mel_energies = filter_bank
        .iter()
        .map(|filter| {
            let mut energy = 0.0_f32;
            for (weight, power) in filter.iter().zip(power_spectrum.iter()) {
                energy += weight * power;
            }
            energy.max(1e-8).ln()
        })
        .collect::<Vec<_>>();

    dct_type_ii(&mel_energies, MFCC_COEFF_COUNT, 1)
}

fn apply_hamming(frame: &[f32]) -> Vec<f32> {
    let length = frame.len().max(1) as f32;
    frame
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let phase = (2.0 * std::f32::consts::PI * index as f32) / (length - 1.0).max(1.0);
            let window = 0.54 - 0.46 * phase.cos();
            sample * window
        })
        .collect()
}

fn power_spectrum_naive(frame: &[f32]) -> Vec<f32> {
    let frame_len = frame.len();
    let mut spectrum = vec![0.0_f32; frame_len / 2 + 1];

    for (bin, value) in spectrum.iter_mut().enumerate() {
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;
        for (index, sample) in frame.iter().enumerate() {
            let angle = (2.0 * std::f32::consts::PI * bin as f32 * index as f32) / frame_len as f32;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        *value = (real * real + imag * imag) / frame_len as f32;
    }

    spectrum
}

fn build_mel_filter_bank(
    sample_rate: u32,
    frame_size: usize,
    filter_count: usize,
) -> Vec<Vec<f32>> {
    let spectrum_bins = frame_size / 2 + 1;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_points = (0..filter_count + 2)
        .map(|index| mel_min + (mel_max - mel_min) * index as f32 / (filter_count + 1) as f32)
        .collect::<Vec<_>>();
    let hz_points = mel_points
        .iter()
        .map(|mel| mel_to_hz(*mel))
        .collect::<Vec<_>>();
    let bins = hz_points
        .iter()
        .map(|hz| ((frame_size as f32 + 1.0) * hz / sample_rate as f32).floor() as usize)
        .collect::<Vec<_>>();

    let mut filters = vec![vec![0.0_f32; spectrum_bins]; filter_count];
    for filter_index in 0..filter_count {
        let left = bins[filter_index].min(spectrum_bins - 1);
        let center = bins[filter_index + 1].min(spectrum_bins - 1);
        let right = bins[filter_index + 2].min(spectrum_bins - 1);

        if center > left {
            for bin in left..center {
                filters[filter_index][bin] = (bin - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for bin in center..right {
                filters[filter_index][bin] = (right - bin) as f32 / (right - center) as f32;
            }
        }
    }

    filters
}

fn dct_type_ii(values: &[f32], coefficient_count: usize, skip_coefficients: usize) -> Vec<f32> {
    let n = values.len().max(1) as f32;
    (skip_coefficients..skip_coefficients + coefficient_count)
        .map(|coefficient| {
            let mut sum = 0.0_f32;
            for (index, value) in values.iter().enumerate() {
                let angle = std::f32::consts::PI * coefficient as f32 * (index as f32 + 0.5) / n;
                sum += value * angle.cos();
            }
            sum
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn resample_feature_sequence(frame_features: &[Vec<f32>], target_steps: usize) -> Vec<f32> {
    let feature_dim = frame_features.first().map(|frame| frame.len()).unwrap_or(0);
    let mut output = vec![0.0_f32; target_steps * feature_dim];

    for step in 0..target_steps {
        let position = if target_steps == 1 {
            0.0
        } else {
            step as f32 * (frame_features.len().saturating_sub(1) as f32)
                / (target_steps.saturating_sub(1) as f32)
        };
        let left_index = position.floor() as usize;
        let right_index = position.ceil() as usize;
        let alpha = position - left_index as f32;

        let left = &frame_features[left_index.min(frame_features.len() - 1)];
        let right = &frame_features[right_index.min(frame_features.len() - 1)];
        for feature_index in 0..feature_dim {
            let value = left[feature_index] * (1.0 - alpha) + right[feature_index] * alpha;
            output[step * feature_dim + feature_index] = value;
        }
    }

    l2_normalize(&mut output);
    output
}

fn average_embedding(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, LightwakeError> {
    if embeddings.is_empty() {
        return Err(LightwakeError::InvalidArgument(
            "cannot average zero embeddings".to_string(),
        ));
    }

    let mut average = vec![0.0_f32; embeddings[0].len()];
    for embedding in embeddings {
        if embedding.len() != average.len() {
            return Err(LightwakeError::InvalidModel(
                "embedding dimensions do not match".to_string(),
            ));
        }
        for (index, value) in embedding.iter().enumerate() {
            average[index] += value;
        }
    }
    for value in &mut average {
        *value /= embeddings.len() as f32;
    }
    l2_normalize(&mut average);
    Ok(average)
}

fn euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return f32::MAX;
    }

    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| {
            let delta = left_value - right_value;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for value in values {
        *value /= norm;
    }
}

fn resample_linear(input: &[i16], input_rate: u32, output_rate: u32) -> Vec<i16> {
    if input.is_empty() || input_rate == output_rate {
        return input.to_vec();
    }

    let ratio = output_rate as f32 / input_rate as f32;
    let output_len = ((input.len() as f32) * ratio).round().max(1.0) as usize;
    let mut output = Vec::<i16>::with_capacity(output_len);

    for output_index in 0..output_len {
        let source_position = output_index as f32 / ratio;
        let left_index = source_position.floor() as usize;
        let right_index = (left_index + 1).min(input.len() - 1);
        let alpha = source_position - left_index as f32;
        let left = input[left_index] as f32;
        let right = input[right_index] as f32;
        let interpolated = left * (1.0 - alpha) + right * alpha;
        output.push(interpolated.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }

    output
}

fn action_id_for_keyword(keyword: &str) -> u8 {
    match keyword.to_ascii_lowercase().as_str() {
        "on" | "bat" | "bật" => 1,
        "off" | "tat" | "tắt" => 2,
        _ => 0,
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take_exact(&mut self, length: usize) -> Result<&'a [u8], LightwakeError> {
        if self.offset + length > self.bytes.len() {
            return Err(LightwakeError::InvalidModel(
                "unexpected end of binary model".to_string(),
            ));
        }
        let slice = &self.bytes[self.offset..self.offset + length];
        self.offset += length;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, LightwakeError> {
        Ok(self.take_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, LightwakeError> {
        let bytes = self.take_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, LightwakeError> {
        let bytes = self.take_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(&mut self) -> Result<f32, LightwakeError> {
        let bytes = self.take_exact(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(sample_rate: u32, frequency_hz: f32, samples: usize) -> Vec<i16> {
        (0..samples)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (f32::sin(t * frequency_hz * std::f32::consts::TAU) * 24_000.0) as i16
            })
            .collect()
    }

    #[test]
    fn embedding_is_fixed_size() {
        let embedding = compute_mfcc_embedding(&sine_wave(16_000, 440.0, 8_000), 16_000);
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn model_roundtrip_binary() {
        let model = LightwakeModel {
            sample_rate: 16_000,
            frame_size: TRANSPORT_FRAME_SIZE as u16,
            target_steps: TARGET_STEPS as u16,
            templates: vec![KeywordTemplate {
                keyword: "on".to_string(),
                action_id: 1,
                threshold: 0.7,
                embedding: vec![0.1; EMBEDDING_DIM],
            }],
        };

        let mut bytes = Vec::<u8>::new();
        model
            .save(Path::new("target/test-lightwake-model.lww"))
            .ok();
        bytes.extend_from_slice(&MODEL_MAGIC);
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&(TRANSPORT_FRAME_SIZE as u16).to_le_bytes());
        bytes.extend_from_slice(&(TARGET_STEPS as u16).to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.push(2);
        bytes.extend_from_slice(b"on");
        bytes.push(1);
        bytes.extend_from_slice(&0.7f32.to_le_bytes());
        bytes.extend_from_slice(&(EMBEDDING_DIM as u16).to_le_bytes());
        for _ in 0..EMBEDDING_DIM {
            bytes.extend_from_slice(&0.1f32.to_le_bytes());
        }

        let parsed = LightwakeModel::from_bytes(&bytes).expect("model should parse");
        assert_eq!(parsed.templates.len(), 1);
        assert_eq!(parsed.templates[0].keyword, "on");
    }

    #[test]
    fn detect_matches_same_signal() {
        let samples = sine_wave(16_000, 440.0, 8_000);
        let embedding = compute_mfcc_embedding(&samples, 16_000);
        let model = LightwakeModel {
            sample_rate: 16_000,
            frame_size: TRANSPORT_FRAME_SIZE as u16,
            target_steps: TARGET_STEPS as u16,
            templates: vec![KeywordTemplate {
                keyword: "on".to_string(),
                action_id: 1,
                threshold: 0.45,
                embedding,
            }],
        };

        let detection = model.detect_pcm(&samples, 16_000);
        assert!(detection.is_some());
        assert_eq!(detection.expect("detection").action_id, 1);
    }
}
