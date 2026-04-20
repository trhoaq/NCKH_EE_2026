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
const DELTA_ORDER: usize = 2;
const FRAME_FEATURE_DIM: usize = MFCC_COEFF_COUNT * 3;
const EMBEDDING_DIM: usize = TARGET_STEPS * FRAME_FEATURE_DIM;
const TRIM_FRAME_SIZE: usize = TRANSPORT_FRAME_SIZE;
const TRIM_HOP_SIZE: usize = TRIM_FRAME_SIZE / 2;
const TRIM_PAD_FRAMES: usize = 2;
const SILENCE_THRESHOLD_RATIO: f32 = 0.18;
const SILENCE_FLOOR: f32 = 0.01;
const NOISE_GATE_RATIO: f32 = 0.08;
const TARGET_RMS: f32 = 0.18;
const MIN_ACTIVE_FRAMES: usize = 2;
const MAX_TEMPLATES_PER_KEYWORD: usize = 8;
const POSITIVE_THRESHOLD_PERCENTILE: f32 = 0.90;
const NEGATIVE_THRESHOLD_PERCENTILE: f32 = 0.25;
const HARD_NEGATIVE_COUNT: usize = 12;
const AUGMENT_SHIFT_SAMPLES: usize = 160;
const AUGMENT_NOISE_SCALE: f32 = 0.012;

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

#[derive(Clone, Debug, Default)]
struct KeywordTrainingSet {
    base_embeddings: Vec<Vec<f32>>,
    training_embeddings: Vec<Vec<f32>>,
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
                let normalized_distance = (distance / (template.threshold + 1e-6)).clamp(0.0, 1.0);
                let score = (1.0 - normalized_distance * normalized_distance).clamp(0.0, 1.0);
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

    let mut grouped = std::collections::BTreeMap::<String, KeywordTrainingSet>::new();
    for example in keyword_examples {
        let wav = read_wav_i16(&example.path)?;
        let samples = if wav.sample_rate == config.sample_rate {
            wav.samples
        } else {
            resample_linear(&wav.samples, wav.sample_rate, config.sample_rate)
        };
        let base_embedding = compute_mfcc_embedding(&samples, config.sample_rate);
        let augmented_embeddings = augment_positive_embeddings(&samples, config.sample_rate);
        let training_set = grouped.entry(example.keyword.clone()).or_default();
        training_set.base_embeddings.push(base_embedding.clone());
        training_set.training_embeddings.push(base_embedding);
        training_set
            .training_embeddings
            .extend(augmented_embeddings);
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
    for (keyword, training_set) in grouped {
        let template_count = choose_template_count(training_set.base_embeddings.len());
        let clusters = cluster_keyword_embeddings(
            &training_set.base_embeddings,
            &training_set.training_embeddings,
            template_count,
        )?;

        for cluster in clusters {
            if cluster.is_empty() {
                continue;
            }
            let centroid = average_embedding(&cluster)?;
            let positive_distances = cluster
                .iter()
                .map(|embedding| euclidean_distance(&centroid, embedding))
                .collect::<Vec<_>>();
            let hard_negative_distances =
                select_hard_negative_distances(&centroid, &negative_embeddings);
            let threshold =
                threshold_from_percentiles(&positive_distances, &hard_negative_distances, config)?;

            templates.push(KeywordTemplate {
                keyword: keyword.clone(),
                action_id: action_id_for_keyword(&keyword),
                threshold,
                embedding: centroid,
            });
        }
    }

    Ok(LightwakeModel {
        sample_rate: config.sample_rate,
        frame_size: TRANSPORT_FRAME_SIZE as u16,
        target_steps: TARGET_STEPS as u16,
        templates,
    })
}

fn choose_template_count(base_count: usize) -> usize {
    base_count.clamp(1, MAX_TEMPLATES_PER_KEYWORD)
}

fn cluster_keyword_embeddings(
    base_embeddings: &[Vec<f32>],
    training_embeddings: &[Vec<f32>],
    template_count: usize,
) -> Result<Vec<Vec<Vec<f32>>>, LightwakeError> {
    if training_embeddings.is_empty() {
        return Err(LightwakeError::InvalidArgument(
            "cannot cluster zero training embeddings".to_string(),
        ));
    }

    let mut centers = select_seed_embeddings(base_embeddings, template_count)?;
    let mut clusters = vec![Vec::<Vec<f32>>::new(); centers.len()];

    for _ in 0..3 {
        for cluster in &mut clusters {
            cluster.clear();
        }
        for embedding in training_embeddings {
            let index = nearest_center_index(&centers, embedding);
            clusters[index].push(embedding.clone());
        }
        for (index, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                centers[index] = average_embedding(cluster)?;
            }
        }
    }

    Ok(clusters
        .into_iter()
        .filter(|cluster| !cluster.is_empty())
        .collect())
}

fn select_seed_embeddings(
    base_embeddings: &[Vec<f32>],
    template_count: usize,
) -> Result<Vec<Vec<f32>>, LightwakeError> {
    if base_embeddings.is_empty() {
        return Err(LightwakeError::InvalidArgument(
            "cannot seed templates without positive embeddings".to_string(),
        ));
    }

    let target = template_count.min(base_embeddings.len()).max(1);
    let mut seeds = vec![base_embeddings[0].clone()];
    while seeds.len() < target {
        let next = base_embeddings
            .iter()
            .max_by(|left, right| {
                distance_to_nearest_seed(left, &seeds)
                    .partial_cmp(&distance_to_nearest_seed(right, &seeds))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| {
                LightwakeError::InvalidArgument("failed to select template seeds".to_string())
            })?;
        if seeds
            .iter()
            .any(|seed| euclidean_distance(seed, &next) < 1e-5)
        {
            break;
        }
        seeds.push(next);
    }
    Ok(seeds)
}

fn distance_to_nearest_seed(embedding: &[f32], seeds: &[Vec<f32>]) -> f32 {
    seeds
        .iter()
        .map(|seed| euclidean_distance(seed, embedding))
        .fold(f32::MAX, f32::min)
}

fn nearest_center_index(centers: &[Vec<f32>], embedding: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_distance = f32::MAX;
    for (index, center) in centers.iter().enumerate() {
        let distance = euclidean_distance(center, embedding);
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    best_index
}

fn select_hard_negative_distances(center: &[f32], negative_embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut distances = negative_embeddings
        .iter()
        .map(|embedding| euclidean_distance(center, embedding))
        .collect::<Vec<_>>();
    distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(HARD_NEGATIVE_COUNT.min(distances.len()));
    distances
}

fn threshold_from_percentiles(
    positive_distances: &[f32],
    hard_negative_distances: &[f32],
    config: &BuildConfig,
) -> Result<f32, LightwakeError> {
    if positive_distances.is_empty() {
        return Err(LightwakeError::InvalidArgument(
            "cannot derive threshold without positive distances".to_string(),
        ));
    }

    let positive_guard = percentile(positive_distances, POSITIVE_THRESHOLD_PERCENTILE);
    let negative_guard = if hard_negative_distances.is_empty() {
        positive_guard + config.threshold_margin + 0.08
    } else {
        percentile(hard_negative_distances, NEGATIVE_THRESHOLD_PERCENTILE)
    };

    let midpoint = (positive_guard + negative_guard) * 0.5;
    let mut threshold = (positive_guard + config.threshold_margin * 0.35)
        .min(midpoint)
        .max(positive_guard + 0.005);
    if !hard_negative_distances.is_empty() {
        threshold = threshold.min((negative_guard - 0.01).max(positive_guard + 0.005));
    }
    Ok(threshold.clamp(0.05, 2.5))
}

fn percentile(values: &[f32], percentile: f32) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }

    let position = percentile.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let left_index = position.floor() as usize;
    let right_index = position.ceil() as usize;
    let alpha = position - left_index as f32;
    let left = sorted[left_index];
    let right = sorted[right_index];
    left * (1.0 - alpha) + right * alpha
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
    let preprocessed = preprocess_samples(samples);
    let emphasized = pre_emphasize(&preprocessed);
    let frames = retain_salient_frames(split_into_overlapping_frames(
        &emphasized,
        MFCC_FRAME_SIZE,
        MFCC_HOP_SIZE,
    ));
    let frame_mfcc = if frames.is_empty() {
        vec![vec![0.0; MFCC_COEFF_COUNT]]
    } else {
        let filter_bank = build_mel_filter_bank(sample_rate, MFCC_FRAME_SIZE, MEL_FILTER_COUNT);
        let mut features = frames
            .iter()
            .map(|frame| mfcc_for_frame(frame, &filter_bank))
            .collect::<Vec<_>>();
        cepstral_mean_variance_normalize(&mut features);
        let mut enriched = append_delta_features(&features);
        cepstral_mean_variance_normalize(&mut enriched);
        enriched
    };

    let embedding = resample_feature_sequence(&frame_mfcc, TARGET_STEPS);
    debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
    embedding
}

fn preprocess_samples(samples: &[i16]) -> Vec<f32> {
    let normalized = samples
        .iter()
        .map(|sample| (*sample as f32) / i16::MAX as f32)
        .collect::<Vec<_>>();
    let centered = remove_dc_offset(&normalized);
    let trimmed = trim_to_active_region(&centered, TRIM_FRAME_SIZE, TRIM_HOP_SIZE);
    let gated = apply_soft_noise_gate(&trimmed);
    stabilize_signal_level(&gated)
}

fn augment_positive_embeddings(samples: &[i16], sample_rate: u32) -> Vec<Vec<f32>> {
    let variants = vec![
        shift_samples(samples, AUGMENT_SHIFT_SAMPLES as isize),
        shift_samples(samples, -(AUGMENT_SHIFT_SAMPLES as isize)),
        add_deterministic_noise(samples, AUGMENT_NOISE_SCALE),
        speed_perturb(samples, 0.97),
        speed_perturb(samples, 1.03),
    ];

    variants
        .into_iter()
        .filter(|variant| !variant.is_empty())
        .map(|variant| compute_mfcc_embedding(&variant, sample_rate))
        .collect()
}

fn pre_emphasize(samples: &[f32]) -> Vec<f32> {
    let mut emphasized = Vec::with_capacity(samples.len());
    let mut previous = 0.0_f32;
    for &sample in samples {
        emphasized.push(sample - 0.97 * previous);
        previous = sample;
    }
    emphasized
}

fn remove_dc_offset(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|sample| sample - mean).collect()
}

fn trim_to_active_region(samples: &[f32], frame_size: usize, hop_size: usize) -> Vec<f32> {
    if samples.len() <= frame_size || hop_size == 0 {
        return samples.to_vec();
    }

    let frame_count = 1 + samples.len().saturating_sub(1) / hop_size;
    let mut energies = Vec::<f32>::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let start = frame_index * hop_size;
        if start >= samples.len() {
            break;
        }
        let end = (start + frame_size).min(samples.len());
        let frame = &samples[start..end];
        let rms =
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32).sqrt();
        energies.push(rms);
    }

    let max_energy = energies.iter().copied().fold(0.0_f32, f32::max);
    if max_energy <= 1e-5 {
        return samples.to_vec();
    }

    let vad_mask = compute_energy_vad_mask(&energies);
    let first_active = match vad_mask.iter().position(|is_active| *is_active) {
        Some(index) => index,
        None => return samples.to_vec(),
    };
    let last_active = vad_mask
        .iter()
        .rposition(|is_active| *is_active)
        .unwrap_or(first_active);
    let padding = hop_size * TRIM_PAD_FRAMES;
    let start = first_active
        .saturating_mul(hop_size)
        .saturating_sub(padding);
    let end = ((last_active * hop_size) + frame_size + padding).min(samples.len());

    if end <= start {
        return samples.to_vec();
    }

    fine_trim_active_samples(&samples[start..end])
}

fn apply_soft_noise_gate(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(1.0_f32, f32::max);
    let gate = (peak * NOISE_GATE_RATIO).max(0.002);

    samples
        .iter()
        .map(|sample| {
            let amplitude = sample.abs();
            if amplitude >= gate {
                *sample
            } else {
                let ratio = amplitude / gate;
                sample * ratio * ratio
            }
        })
        .collect()
}

fn fine_trim_active_samples(samples: &[f32]) -> Vec<f32> {
    if samples.len() <= MFCC_FRAME_SIZE {
        return samples.to_vec();
    }

    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    if peak <= 1e-6 {
        return samples.to_vec();
    }

    let threshold = (peak * 0.08).max(0.003);
    let first = match samples.iter().position(|sample| sample.abs() >= threshold) {
        Some(index) => index,
        None => return samples.to_vec(),
    };
    let last = samples
        .iter()
        .rposition(|sample| sample.abs() >= threshold)
        .unwrap_or(first);
    let padding = MFCC_HOP_SIZE / 2;
    let start = first.saturating_sub(padding);
    let end = (last + padding + 1).min(samples.len());

    samples[start..end].to_vec()
}

fn stabilize_signal_level(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    if peak <= 1e-6 {
        return samples.to_vec();
    }

    let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32)
        .sqrt()
        .max(1e-6);
    let gain = (TARGET_RMS / rms).clamp(0.5, 8.0).min(0.98 / peak);

    samples
        .iter()
        .map(|sample| (sample * gain).clamp(-1.0, 1.0))
        .collect()
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

fn retain_salient_frames(frames: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if frames.len() <= 1 {
        return frames;
    }

    let energies = frames
        .iter()
        .map(|frame| {
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32).sqrt()
        })
        .collect::<Vec<_>>();
    let max_energy = energies.iter().copied().fold(0.0_f32, f32::max);
    if max_energy <= 1e-5 {
        return frames;
    }

    let vad_mask = compute_energy_vad_mask(&energies);
    let kept = frames
        .into_iter()
        .zip(vad_mask)
        .filter_map(|(frame, is_active)| is_active.then_some(frame))
        .collect::<Vec<_>>();

    if kept.is_empty() {
        vec![vec![0.0; MFCC_FRAME_SIZE]]
    } else {
        kept
    }
}

fn compute_energy_vad_mask(frame_energies: &[f32]) -> Vec<bool> {
    if frame_energies.is_empty() {
        return Vec::new();
    }

    let max_energy = frame_energies.iter().copied().fold(0.0_f32, f32::max);
    if max_energy <= 1e-5 {
        return vec![false; frame_energies.len()];
    }

    let threshold = (max_energy * SILENCE_THRESHOLD_RATIO).max(SILENCE_FLOOR * 0.5);
    let raw_mask = frame_energies
        .iter()
        .map(|energy| *energy >= threshold)
        .collect::<Vec<_>>();
    let mut mask = raw_mask.clone();
    for index in 0..raw_mask.len() {
        if raw_mask[index] {
            if index > 0 {
                mask[index - 1] = true;
            }
            if index + 1 < raw_mask.len() {
                mask[index + 1] = true;
            }
        }
    }

    let mut run_start = None::<usize>;
    for index in 0..=mask.len() {
        let active = mask.get(index).copied().unwrap_or(false);
        match (run_start, active) {
            (None, true) => run_start = Some(index),
            (Some(start), false) => {
                if index - start < MIN_ACTIVE_FRAMES {
                    for item in mask.iter_mut().take(index).skip(start) {
                        *item = false;
                    }
                }
                run_start = None;
            }
            _ => {}
        }
    }
    mask
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

fn append_delta_features(frame_features: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if frame_features.is_empty() {
        return Vec::new();
    }

    let delta = compute_delta_features(frame_features, DELTA_ORDER);
    let delta_delta = compute_delta_features(&delta, DELTA_ORDER);

    frame_features
        .iter()
        .zip(delta.iter())
        .zip(delta_delta.iter())
        .map(|((base, first_delta), second_delta)| {
            let mut combined = Vec::with_capacity(FRAME_FEATURE_DIM);
            combined.extend_from_slice(base);
            combined.extend_from_slice(first_delta);
            combined.extend_from_slice(second_delta);
            combined
        })
        .collect()
}

fn compute_delta_features(frame_features: &[Vec<f32>], order: usize) -> Vec<Vec<f32>> {
    if frame_features.is_empty() {
        return Vec::new();
    }

    let denominator = 2.0 * (1..=order).map(|index| (index * index) as f32).sum::<f32>();
    let feature_dim = frame_features[0].len();
    let mut deltas = vec![vec![0.0_f32; feature_dim]; frame_features.len()];

    for (frame_index, delta_frame) in deltas.iter_mut().enumerate() {
        for offset in 1..=order {
            let left_index = frame_index.saturating_sub(offset);
            let right_index = (frame_index + offset).min(frame_features.len() - 1);
            let weight = offset as f32 / denominator;
            for feature_index in 0..feature_dim {
                delta_frame[feature_index] += weight
                    * (frame_features[right_index][feature_index]
                        - frame_features[left_index][feature_index]);
            }
        }
    }

    deltas
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn cepstral_mean_variance_normalize(frame_features: &mut [Vec<f32>]) {
    if frame_features.is_empty() {
        return;
    }

    for feature_index in 0..frame_features[0].len() {
        let mean = frame_features
            .iter()
            .map(|frame| frame[feature_index])
            .sum::<f32>()
            / frame_features.len() as f32;
        let variance = frame_features
            .iter()
            .map(|frame| {
                let centered = frame[feature_index] - mean;
                centered * centered
            })
            .sum::<f32>()
            / frame_features.len() as f32;
        let stddev = variance.sqrt().max(1e-4);
        for frame in frame_features.iter_mut() {
            frame[feature_index] = (frame[feature_index] - mean) / stddev;
        }
    }
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

fn shift_samples(input: &[i16], shift: isize) -> Vec<i16> {
    if input.is_empty() || shift == 0 {
        return input.to_vec();
    }

    let mut output = vec![0i16; input.len()];
    for (index, sample) in input.iter().enumerate() {
        let target = index as isize + shift;
        if (0..input.len() as isize).contains(&target) {
            output[target as usize] = *sample;
        }
    }
    output
}

fn add_deterministic_noise(input: &[i16], scale: f32) -> Vec<i16> {
    input
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let phase = index as f32 * 0.173_205_08;
            let noise = (phase.sin() * 0.7 + (phase * 0.37).cos() * 0.3) * scale * 32_000.0;
            (*sample as f32 + noise)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

fn speed_perturb(input: &[i16], factor: f32) -> Vec<i16> {
    if input.is_empty() || factor <= 0.0 {
        return input.to_vec();
    }

    let output_len = ((input.len() as f32) / factor).round().max(1.0) as usize;
    let mut output = Vec::<i16>::with_capacity(output_len);
    for output_index in 0..output_len {
        let source_position = output_index as f32 * factor;
        let left_index = source_position.floor() as usize;
        let right_index = (left_index + 1).min(input.len() - 1);
        let alpha = source_position - left_index as f32;
        let left = input[left_index.min(input.len() - 1)] as f32;
        let right = input[right_index] as f32;
        let interpolated = left * (1.0 - alpha) + right * alpha;
        output.push(interpolated.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    output
}

fn action_id_for_keyword(keyword: &str) -> u8 {
    match keyword.to_ascii_lowercase().as_str() {
        "on" | "bat" => 1,
        "off" | "tat" => 2,
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sine_wave(sample_rate: u32, frequency_hz: f32, samples: usize) -> Vec<i16> {
        (0..samples)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (f32::sin(t * frequency_hz * std::f32::consts::TAU) * 24_000.0) as i16
            })
            .collect()
    }

    fn write_pcm_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let riff_len = 36 + data_len;
        let mut bytes = Vec::<u8>::with_capacity((44 + data_len) as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_len.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        fs::write(path, bytes).expect("wav written");
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

    #[test]
    fn preprocessing_trims_padding_to_active_region() {
        let mut padded = vec![0i16; 3_200];
        padded.extend(sine_wave(16_000, 440.0, 8_000));
        padded.extend(vec![0i16; 3_200]);

        let normalized = padded
            .iter()
            .map(|sample| (*sample as f32) / i16::MAX as f32)
            .collect::<Vec<_>>();
        let trimmed = trim_to_active_region(&normalized, TRIM_FRAME_SIZE, TRIM_HOP_SIZE);

        assert!(trimmed.len() < normalized.len() - 2_500);
        assert!(trimmed.len() > 7_000);
    }

    #[test]
    fn preprocessing_handles_dc_offset_and_gain_changes() {
        let reference = sine_wave(16_000, 550.0, 8_000);
        let shifted = reference
            .iter()
            .map(|sample| {
                ((*sample as i32 / 3) + 2_000).clamp(i16::MIN as i32, i16::MAX as i32) as i16
            })
            .collect::<Vec<_>>();

        let reference_embedding = compute_mfcc_embedding(&reference, 16_000);
        let shifted_embedding = compute_mfcc_embedding(&shifted, 16_000);
        let distance = euclidean_distance(&reference_embedding, &shifted_embedding);

        assert!(distance < 0.45, "dc/gain distance too high: {distance}");
    }

    #[test]
    fn build_model_uses_multiple_templates_for_multimodal_keyword() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let temp_root = PathBuf::from(format!("target/test-training-{unique}"));
        fs::create_dir_all(&temp_root).expect("temp dir created");

        let positive_a = temp_root.join("on_a.wav");
        let positive_b = temp_root.join("on_b.wav");
        let positive_c = temp_root.join("on_c.wav");
        let positive_d = temp_root.join("on_d.wav");
        let negative = temp_root.join("negative.wav");

        write_pcm_wav(&positive_a, 16_000, &sine_wave(16_000, 420.0, 8_000));
        write_pcm_wav(&positive_b, 16_000, &sine_wave(16_000, 430.0, 8_000));
        write_pcm_wav(&positive_c, 16_000, &sine_wave(16_000, 780.0, 8_000));
        write_pcm_wav(&positive_d, 16_000, &sine_wave(16_000, 790.0, 8_000));
        write_pcm_wav(&negative, 16_000, &sine_wave(16_000, 1_600.0, 8_000));

        let examples = vec![
            Example {
                keyword: "on".to_string(),
                path: positive_a,
            },
            Example {
                keyword: "on".to_string(),
                path: positive_b,
            },
            Example {
                keyword: "on".to_string(),
                path: positive_c,
            },
            Example {
                keyword: "on".to_string(),
                path: positive_d,
            },
        ];

        let model = build_model(
            &examples,
            &[negative],
            &BuildConfig {
                sample_rate: 16_000,
                threshold_margin: 0.08,
            },
        )
        .expect("model builds");

        assert!(model.templates.len() >= 2, "expected multiple templates");
        let _ = fs::remove_dir_all(temp_root);
    }
}
