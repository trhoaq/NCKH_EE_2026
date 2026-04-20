use std::cmp::min;
use std::sync::{Mutex, OnceLock};

use lightwake::LightwakeModel;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct rust_bridge_audio_config_t {
    pub sample_rate: u32,
    pub frame_samples: u32,
    pub bits_per_sample: u16,
    pub channels: u16,
    pub wakeword_count: u32,
    pub keyword_set_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct rust_bridge_detection_t {
    pub detected: u8,
    pub action_id: u8,
    pub reserved: u16,
    pub score_milli: u32,
    pub keyword: [u8; 32],
}

impl Default for rust_bridge_detection_t {
    fn default() -> Self {
        Self {
            detected: 0,
            action_id: 0,
            reserved: 0,
            score_milli: 0,
            keyword: [0; 32],
        }
    }
}

struct EngineState {
    model: LightwakeModel,
    audio_config: rust_bridge_audio_config_t,
}

static ENGINE: OnceLock<Mutex<Option<EngineState>>> = OnceLock::new();

#[allow(dead_code)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated_lightwake.rs"));
}

fn engine_slot() -> &'static Mutex<Option<EngineState>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

fn fallback_frame_samples() -> u32 {
    ((generated::SAMPLE_RATE_FALLBACK as u64 * generated::FRAME_MS_FALLBACK as u64) / 1000u64)
        as u32
}

fn build_audio_config(
    sample_rate: u32,
    frame_samples: u32,
    wakeword_count: u32,
) -> rust_bridge_audio_config_t {
    rust_bridge_audio_config_t {
        sample_rate,
        frame_samples,
        bits_per_sample: generated::BITS_PER_SAMPLE,
        channels: generated::CHANNELS,
        wakeword_count,
        keyword_set_version: generated::KEYWORD_SET_VERSION,
    }
}

#[no_mangle]
pub extern "C" fn wakeword_bridge_init(out_audio_config: *mut rust_bridge_audio_config_t) -> i32 {
    let fallback = build_audio_config(generated::SAMPLE_RATE_FALLBACK, fallback_frame_samples(), 0);

    if generated::MODEL_BYTES.is_empty() {
        unsafe {
            if !out_audio_config.is_null() {
                *out_audio_config = fallback;
            }
        }
        *engine_slot().lock().expect("engine mutex poisoned") = None;
        return 1;
    }

    let model = match LightwakeModel::from_bytes(generated::MODEL_BYTES) {
        Ok(model) => model,
        Err(_) => {
            unsafe {
                if !out_audio_config.is_null() {
                    *out_audio_config = fallback;
                }
            }
            *engine_slot().lock().expect("engine mutex poisoned") = None;
            return -1;
        }
    };

    let audio_config = build_audio_config(
        model.sample_rate,
        u32::from(model.frame_size),
        model.templates.len() as u32,
    );

    unsafe {
        if !out_audio_config.is_null() {
            *out_audio_config = audio_config;
        }
    }

    *engine_slot().lock().expect("engine mutex poisoned") = Some(EngineState {
        model,
        audio_config,
    });
    0
}

#[no_mangle]
pub extern "C" fn wakeword_bridge_reset_session() -> i32 {
    if engine_slot()
        .lock()
        .expect("engine mutex poisoned")
        .is_some()
    {
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn wakeword_bridge_process_samples(
    samples: *const i16,
    sample_count: usize,
    out_detection: *mut rust_bridge_detection_t,
) -> i32 {
    if samples.is_null() || out_detection.is_null() {
        return -1;
    }

    let guard = engine_slot().lock().expect("engine mutex poisoned");
    let state = match guard.as_ref() {
        Some(state) => state,
        None => return -1,
    };

    let slice = unsafe { std::slice::from_raw_parts(samples, sample_count) };
    let detection = state
        .model
        .detect_pcm(slice, state.audio_config.sample_rate);

    let mut native_detection = rust_bridge_detection_t::default();
    if let Some(detection) = detection {
        native_detection.detected = 1;
        native_detection.action_id = detection.action_id;
        native_detection.score_milli = (detection.score.clamp(0.0, 1.0) * 1000.0) as u32;
        let keyword_bytes = detection.keyword.as_bytes();
        let copy_length = min(
            keyword_bytes.len(),
            native_detection.keyword.len().saturating_sub(1),
        );
        native_detection.keyword[..copy_length].copy_from_slice(&keyword_bytes[..copy_length]);
    }

    unsafe {
        *out_detection = native_detection;
    }
    0
}
