# command_dl

Phone-side streaming command recognition pipeline for:

- `on`
- `off`
- `unknown`
- `silence`

The active lane uses TensorFlow Lite artifacts:

- `command_model.tflite`
- `command_model_meta.json`

## Train and export TFLite

```powershell
python .\model\command_dl\train_export_tflite.py `
  --on-dir .\model\data\data_on `
  --off-dir .\model\data\data_off `
  --unknown-dir .\model\data\unknown `
  --output-tflite .\app\mobile\src\main\assets\command_model.tflite `
  --output-meta .\app\mobile\src\main\assets\command_model.json `
  --epochs 6
```

`tensorflow` must be installed in your Python environment for this export path.

## Test one WAV file

```powershell
python .\model\command_dl\infer_wav.py `
  --model-tflite .\app\mobile\src\main\assets\command_model.tflite `
  --model-meta .\app\mobile\src\main\assets\command_model.json `
  --wav .\model\data\data_on\004ae714_nohash_0.wav
```
