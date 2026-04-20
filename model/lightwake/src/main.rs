mod ffmpeg_normalize;

use std::env;
use std::path::{Path, PathBuf};

use lightwake::{build_model, read_wav_i16, BuildConfig, Example, LightwakeError, LightwakeModel};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), LightwakeError> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| {
        LightwakeError::InvalidArgument("missing command: use build or detect".to_string())
    })?;

    match command.as_str() {
        "build" => run_build(args.collect()),
        "detect" => run_detect(args.collect()),
        _ => Err(LightwakeError::InvalidArgument(format!(
            "unknown command {command}, use build or detect"
        ))),
    }
}

fn run_build(arguments: Vec<String>) -> Result<(), LightwakeError> {
    let mut keyword = None::<String>;
    let mut positive_specs = Vec::<(String, PathBuf)>::new();
    let mut positives = Vec::<Example>::new();
    let mut negative_source_dir = None::<PathBuf>;
    let mut output = None::<PathBuf>;
    let mut sample_rate = 16_000u32;
    let mut threshold_margin = BuildConfig::default().threshold_margin;

    let mut index = 0usize;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--keyword" => {
                keyword = Some(next_value(&arguments, &mut index, "--keyword")?);
            }
            "--positive" => {
                let directory = PathBuf::from(next_value(&arguments, &mut index, "--positive")?);
                let current_keyword = keyword.clone().ok_or_else(|| {
                    LightwakeError::InvalidArgument(
                        "--positive must come after --keyword so files can be labeled".to_string(),
                    )
                })?;
                positive_specs.push((current_keyword, directory));
            }
            "--negative" => {
                negative_source_dir = Some(PathBuf::from(next_value(
                    &arguments,
                    &mut index,
                    "--negative",
                )?));
            }
            "--sample-rate" => {
                sample_rate = next_value(&arguments, &mut index, "--sample-rate")?
                    .parse()
                    .map_err(|_| {
                        LightwakeError::InvalidArgument(
                            "sample rate must be an integer".to_string(),
                        )
                    })?;
            }
            "--threshold-margin" => {
                threshold_margin = next_value(&arguments, &mut index, "--threshold-margin")?
                    .parse()
                    .map_err(|_| {
                        LightwakeError::InvalidArgument(
                            "threshold margin must be a number".to_string(),
                        )
                    })?;
            }
            "--output" => {
                output = Some(PathBuf::from(next_value(
                    &arguments, &mut index, "--output",
                )?));
            }
            flag => {
                return Err(LightwakeError::InvalidArgument(format!(
                    "unknown build flag {flag}"
                )));
            }
        }
        index += 1;
    }

    let output = output.ok_or_else(|| {
        LightwakeError::InvalidArgument("missing --output for build command".to_string())
    })?;
    if positive_specs.is_empty() {
        return Err(LightwakeError::InvalidArgument(
            "no positive wav files found".to_string(),
        ));
    }
    let negative_source_dir = negative_source_dir.ok_or_else(|| {
        LightwakeError::InvalidArgument("missing --negative for build command".to_string())
    })?;

    let normalized_dataset =
        ffmpeg_normalize::normalize_dataset(&positive_specs, &negative_source_dir, sample_rate)?;
    for ((keyword_name, _), normalized_dir) in positive_specs
        .iter()
        .zip(normalized_dataset.positive_dirs.iter())
    {
        positives.extend(read_wav_examples(normalized_dir, keyword_name)?);
    }
    let negatives = read_wav_paths(&normalized_dataset.negative_dir)?;

    let model = build_model(
        &positives,
        &negatives,
        &BuildConfig {
            sample_rate,
            threshold_margin,
        },
    )?;
    model.save(&output)?;
    println!(
        "saved {} templates to {} using ffmpeg-normalized wavs",
        model.templates.len(),
        output.display()
    );
    Ok(())
}

fn run_detect(arguments: Vec<String>) -> Result<(), LightwakeError> {
    let mut model_path = None::<PathBuf>;
    let mut input_path = None::<PathBuf>;

    let mut index = 0usize;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--model" => {
                model_path = Some(PathBuf::from(next_value(
                    &arguments, &mut index, "--model",
                )?))
            }
            "--input" => {
                input_path = Some(PathBuf::from(next_value(
                    &arguments, &mut index, "--input",
                )?))
            }
            flag => {
                return Err(LightwakeError::InvalidArgument(format!(
                    "unknown detect flag {flag}"
                )));
            }
        }
        index += 1;
    }

    let model_path = model_path.ok_or_else(|| {
        LightwakeError::InvalidArgument("missing --model for detect command".to_string())
    })?;
    let input_path = input_path.ok_or_else(|| {
        LightwakeError::InvalidArgument("missing --input for detect command".to_string())
    })?;

    let model = LightwakeModel::load(Path::new(&model_path))?;
    let (wav, _normalized_input) = match read_wav_i16(Path::new(&input_path)) {
        Ok(wav) => (wav, None),
        Err(LightwakeError::InvalidWav(_)) => {
            let normalized_input =
                ffmpeg_normalize::normalize_input_wav(Path::new(&input_path), model.sample_rate)?;
            let wav = read_wav_i16(normalized_input.wav_path())?;
            (wav, Some(normalized_input))
        }
        Err(error) => return Err(error),
    };

    match model.detect_pcm(&wav.samples, wav.sample_rate) {
        Some(detection) => {
            println!(
                "DETECTED keyword={} action_id={} score={:.3}",
                detection.keyword, detection.action_id, detection.score
            );
        }
        None => {
            println!("NO_DETECTION");
        }
    }

    Ok(())
}

fn next_value(
    arguments: &[String],
    index: &mut usize,
    flag: &str,
) -> Result<String, LightwakeError> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| LightwakeError::InvalidArgument(format!("missing value for {flag}")))
}

fn read_wav_examples(directory: &PathBuf, keyword: &str) -> Result<Vec<Example>, LightwakeError> {
    Ok(read_wav_paths(directory)?
        .into_iter()
        .map(|path| Example {
            keyword: keyword.to_string(),
            path,
        })
        .collect())
}

fn read_wav_paths(directory: &PathBuf) -> Result<Vec<PathBuf>, LightwakeError> {
    let mut paths = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| {
            path.extension()
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(LightwakeError::InvalidArgument(format!(
            "no .wav files found in {}",
            directory.display()
        )));
    }
    Ok(paths)
}
