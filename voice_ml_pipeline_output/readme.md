# Voice Pipeline — Output Artifacts

Each run of `voice-pca-pipeline-guided.ipynb` creates one timestamped subfolder here,
named `{Model}_{dd-Mon-yyyy}_{HH-MM-SS}/`.

**Latest run:** `LightGBM_23-May-2026_14-03-40/` — LightGBM, 99.50% test accuracy,
23 PCA features (`PC1`–`PC24`, excluding `PC17`), 6 classes
(Anxiety, Bipolar, Depression, Normal, Stress, Suicidal).

## Contents of a run folder

| File | Description |
|------|-------------|
| `best_model.joblib` | Trained classifier, ready for inference |
| `scaler.joblib` | RobustScaler (fit on training data only) |
| `label_encoder.joblib` | Target label encoder (class name ↔ integer) |
| `encoding_artifacts.joblib` | Encoding mappings (none needed — PCA-only input) |
| `outlier_transformers.joblib` | Per-column outlier smoothing transformers |
| `feature_names.json` | Ordered list of the selected PCA feature names |
| `model_metadata.json` | Best model name, params, all test metrics, class names |
| `pipeline_state.json` | Full pipeline execution state |

## Loading the model

```python
import joblib, json, pandas as pd

run = "LightGBM_23-May-2026_14-03-40"
model   = joblib.load(f"voice_ml_pipeline_output/{run}/best_model.joblib")
scaler  = joblib.load(f"voice_ml_pipeline_output/{run}/scaler.joblib")
encoder = joblib.load(f"voice_ml_pipeline_output/{run}/label_encoder.joblib")
features = json.load(open(f"voice_ml_pipeline_output/{run}/feature_names.json"))

# X must contain the PCA columns listed in features, in the same order.
X = scaler.transform(df[features])
pred = encoder.inverse_transform(model.predict(X))
```

See `demo-api-input-data-sample/` for example PCA inputs.
