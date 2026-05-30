# Text Pipeline — Output Artifacts

Each run of `text-ml-pipeline.ipynb` creates one timestamped subfolder here, named
`{Model}_{dd-Mon-yyyy}_{HH-MM-SS}/`.

**Latest run:** `Extra_Trees_18-May-2026_12-11-08/` — Extra Trees, 99.15% test accuracy,
20 features, 6 classes (ANXIETY, BIPOLAR, DEPRESSION, NORMAL, STRESS, SUICIDAL).

## Contents of a run folder

| File | Description |
|------|-------------|
| `best_model.joblib` | Trained classifier, ready for inference |
| `scaler.joblib` | RobustScaler (fit on training data only) |
| `label_encoder.joblib` | Target label encoder (class name ↔ integer) |
| `encoding_artifacts.joblib` | Categorical encoding mappings |
| `outlier_transformers.joblib` | Per-column outlier smoothing transformers |
| `feature_names.json` | Ordered list of the selected feature names |
| `model_metadata.json` | Best model name, params, all test metrics, class names |
| `pipeline_state.json` | Full pipeline execution state |
| `shap_*.png` | SHAP global importance, per-class, and waterfall plots |

## Loading the model

```python
import joblib, json, pandas as pd

run = "Extra_Trees_18-May-2026_12-11-08"
model   = joblib.load(f"text_ml_pipeline_output/{run}/best_model.joblib")
scaler  = joblib.load(f"text_ml_pipeline_output/{run}/scaler.joblib")
encoder = joblib.load(f"text_ml_pipeline_output/{run}/label_encoder.joblib")
features = json.load(open(f"text_ml_pipeline_output/{run}/feature_names.json"))

# X must contain the columns listed in features, in the same order.
X = scaler.transform(df[features])
pred = encoder.inverse_transform(model.predict(X))
```

> Apply the same outlier transforms and feature ordering used at training time
> (see `outlier_transformers.joblib` and `feature_names.json`) before scaling.
