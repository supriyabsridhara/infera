# Infera Project Guide

This guide describes the data pipeline, Bayesian Network, evaluation process,
and dashboard behavior implemented in the repository. All commands are
intended to run from the project root because the scripts use paths relative to
the current working directory.

## 1. System overview

Infera has four processing stages:

1. **Generate observations** with `generate_dataset.py`.
2. **Fit and serialize the model** with `train_model.py`.
3. **Compare classifiers and create plots** with `evaluate.py`.
4. **Explore predictions and artifacts** with the Flask or Streamlit dashboard.

The main artifact flow is:

```text
generate_dataset.py
        |
        v
data/sensor_data.csv
        |
        +----------------------+
        |                      |
        v                      v
train_model.py             evaluate.py
        |                      |
        v                      +--> data/evaluation_metrics.csv
data/bayesian_model.pkl          +--> data/plots/*.png
        |                      ^
        +----------------------+
        |
        +--> app_flask.py / app.py
```

`evaluate.py` reads both the sensor CSV and the serialized model. It does not
train or save a model itself.

## 2. Synthetic dataset

### Generation

`generate_dataset.py` creates 20,000 rows by default. It first samples the
`entry_status` class using these prior probabilities:

| Class | Probability |
| --- | ---: |
| `no_entry` | 0.50 |
| `authorized` | 0.35 |
| `intrusion` | 0.15 |

It then samples each binary sensor independently, conditioned on the sampled
class. The configured probabilities of a triggered sensor (`1`) are:

| Entry status | Door | Motion | Glass-break |
| --- | ---: | ---: | ---: |
| `no_entry` | 0.03 | 0.05 | 0.01 |
| `authorized` | 0.90 | 0.85 | 0.04 |
| `intrusion` | 0.80 | 0.90 | 0.75 |

The script seeds NumPy's random generator with `42` at module load. Running the
script from a fresh process therefore generates the same dataset for the
default configuration.

### CSV schema

`data/sensor_data.csv` contains the model columns and display-friendly labels:

| Column | Meaning |
| --- | --- |
| `entry_status` | Target class: `no_entry`, `authorized`, or `intrusion` |
| `door_sensor` | Door state: `0` not triggered, `1` triggered |
| `motion_sensor` | Motion state: `0` not triggered, `1` triggered |
| `glass_sensor` | Glass-break state: `0` not triggered, `1` triggered |
| `door_sensor_label` | Text form of the door state: `No` or `Yes` |
| `motion_sensor_label` | Text form of the motion state: `No` or `Yes` |
| `glass_sensor_label` | Text form of the glass state: `No` or `Yes` |

`train_model.load_data()` intentionally selects only the target and three
integer sensor columns. The display label columns are not model inputs.

## 3. Bayesian Network and inference

### Structure and fitting

`train_model.build_and_train()` defines this directed acyclic graph:

```text
entry_status --> door_sensor
entry_status --> motion_sensor
entry_status --> glass_sensor
```

The class is the parent of all three sensors. There are no edges between
sensors, so the model represents them as conditionally independent given
`entry_status`. The graph structure is specified in code; the data is used to
estimate the discrete conditional probability distributions with pgmpy's
`DiscreteMLE` estimator. A model validity check runs before returning the
fitted network.

The trained network is saved with Python pickle to
`data/bayesian_model.pkl`. The dashboard and evaluator load this artifact
rather than fitting a new network on each request.

### Query

`train_model.infer(model, door, motion, glass)` passes the sensor states as
evidence to pgmpy's variable elimination algorithm. The queried variable is
`entry_status`. The returned mapping contains class probabilities plus:

- `risk_level`: `Low`, `Moderate`, or `High`
- `recommended_action`: corresponding plain-language guidance

Risk is calculated from `P(intrusion | evidence)` using the thresholds in
`train_model.py`:

```text
P(intrusion) < 0.30       -> Low
0.30 <= P(intrusion) < 0.70 -> Moderate
P(intrusion) >= 0.70      -> High
```

The recommendations currently say to continue monitoring for low risk, send a
guard to verify moderate risk, and notify security personnel and law
enforcement for high risk. These are illustrative strings, not operational
response rules.

## 4. Model evaluation

`evaluate.py` compares two classifiers:

### Rule-based baseline

The code applies these checks in order:

1. If door and glass-break sensors are triggered, predict `intrusion`.
2. Otherwise, if door and motion sensors are triggered, predict
   `authorized`.
3. Otherwise, predict `no_entry`.

The order matters. For example, a triggered door, motion, and glass sensor is
classified as intrusion because the first rule matches.

### Bayesian Network classifier

For each evaluation row, the code queries the saved model with the three sensor
states and chooses the class with the highest posterior probability.

### Split and metrics

The evaluator uses a stratified 80/20 split with `random_state=42` and
calculates:

- **Accuracy:** fraction of all predictions that are correct.
- **Precision, recall, and F1-score:** weighted averages across the three
  classes, reported as percentages.
- **False positive rate (FPR):** one-vs-rest FPR for each class, then the
  unweighted macro-average of those three rates, reported as a percentage.

The metrics are written to `data/evaluation_metrics.csv`. The script also
prints per-class classification reports and generates:

- `data/plots/comparison_metrics.png`
- `data/plots/confusion_matrices.png`
- `data/plots/class_distribution.png`

### Evaluation caveat

The current workflow fits `bayesian_model.pkl` using the full generated CSV
before `evaluate.py` creates its 80/20 split. Consequently, the Bayesian model
has already seen the records later used as the test set. Its reported test
metrics are therefore not an independent holdout estimate and may be
optimistic. The split still determines the rows used to calculate the metrics,
but it does not isolate those rows from model fitting. A rigorous evaluation
would fit a separate Bayesian Network using only the training partition.

The dataset itself is simulated from simple class-conditional probabilities,
so performance on it does not demonstrate performance on real sensor data.

## 5. Flask website

Start the Flask interface from the repository root:

```bash
python app_flask.py
```

The default local address is `http://127.0.0.1:5000`.

### `GET /`

The index route loads:

- The serialized Bayesian model at application startup.
- Evaluation rows from `data/evaluation_metrics.csv`, if present.
- Dataset class counts from `data/sensor_data.csv`, if present.
- Existing plot files from `data/plots/`, if present.
- The model's CPTs.

It renders `templates/index.html` with these values. The page includes the
sensor controls, a prediction panel, evaluation tabs, dataset distribution,
and CPT display. Missing optional metrics, data, or images are handled by the
template's existing empty states.

### `POST /predict`

The browser sends JSON containing integer sensor values:

```json
{
  "door": 1,
  "motion": 1,
  "glass": 0
}
```

The response contains the class probabilities as percentages, the risk level,
the recommendation, and a base64-encoded PNG chart:

```json
{
  "no_entry": 1.2,
  "authorized": 96.4,
  "intrusion": 2.4,
  "risk_level": "Low",
  "recommended_action": "Continue normal monitoring. No immediate action required.",
  "chart_b64": "<base64-encoded PNG>"
}
```

The page's JavaScript updates the existing result panel with those values.
This endpoint and the visible sensor controls use the same three binary
observations.

## 6. Streamlit dashboard

Start the Streamlit interface with:

```bash
streamlit run app.py
```

It uses the same serialized model and `train_model.infer()` function. The
dashboard provides:

- Sensor radio controls and a predict button.
- Probability metrics, a risk recommendation, and a probability chart.
- Evaluation metrics, comparison chart, and confusion matrices.
- Training dataset distribution and a sample of records.
- An expandable CPT viewer.

Streamlit caches the loaded model as a resource for the current process. The
Flask and Streamlit dashboards are alternative front ends; they do not call
each other.

## 7. Other scripts and files

| File | Responsibility |
| --- | --- |
| `generate_dataset.py` | Generate and save the synthetic CSV |
| `train_model.py` | Load data, build/fit the network, save it, print CPTs, and run a sample query |
| `evaluate.py` | Compare the baseline and saved model; write metrics and plots |
| `sanity_check.py` | Print predictions for four representative sensor combinations |
| `app_flask.py` | Load the model, implement Flask routes, and provide prediction charts |
| `app.py` | Render the Streamlit dashboard |
| `templates/index.html` | Render and interact with the Flask dashboard |
| `requirements.txt` | Declare Python package dependencies |

All data and model paths are relative to the current working directory. Run the
commands from the project root, and preserve the generated files between
pipeline stages.

## 8. Troubleshooting

- **Model file missing:** run `python generate_dataset.py`, then
  `python train_model.py`.
- **Evaluation artifacts missing:** run `python evaluate.py` after the dataset
  and model have been created.
- **Flask cannot be imported:** install the packages with
  `python -m pip install -r requirements.txt`.
- **Unexpected missing files:** confirm the current directory is the project
  root; the scripts use relative `data/` paths.
- **Model import or API mismatch:** use a pgmpy release that supports
  `DiscreteBayesianNetwork`, `DiscreteMLE`, and `VariableElimination`.

## 9. Limitations and responsible use

- Data is synthetic and encodes the assumptions specified in the generator.
- The graph encodes conditional independence between sensor readings given
  entry status.
- The evaluator's saved-model test metrics have training/test leakage, as
  described above.
- Sensor readings have no timestamps, locations, identities, or missing-value
  handling.
- The application has no authentication, authorization, audit trail, or
  operational alert delivery.
- `app_flask.py` runs Flask with debug mode enabled. Keep it on a trusted local
  machine; do not expose this development server to a network.
- Python pickle files must only be loaded from trusted sources.

Treat the interface as a demonstration of probabilistic classification, not as
a substitute for tested safety or security procedures.
