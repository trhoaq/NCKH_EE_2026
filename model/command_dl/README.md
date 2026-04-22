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
  --on-dir .\model\data\data_on\positive `
  --off-dir .\model\data\data_off\positive `
  --unknown-dir .\model\data\data_on\negative `
  --unknown-dir .\model\data\data_off\negative `
  --output-tflite .\app\mobile\src\main\assets\command_model.tflite `
  --output-meta .\app\mobile\src\main\assets\command_model_meta.json `
  --epochs 6
```

`tensorflow` must be installed in your Python environment for this export path.

## Notes

- `infer_wav.py` is the older JSON-weight debug lane and is no longer the active mobile runtime path.
- The active mobile path expects:
  - `command_model.tflite`
  - `command_model_meta.json`
