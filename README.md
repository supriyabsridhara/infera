# Infera

Infera is a small, locally run building-entry monitoring prototype. It uses a
Bayesian Network to estimate the probabilities of **no entry**, **authorized
entry**, and **intrusion** from three binary sensor observations:

- Door sensor
- Motion sensor
- Glass-break sensor

The repository includes a synthetic data generator, model training and
inference, a comparison against a rule-based classifier, and two dashboard
interfaces: a Flask website and a Streamlit app.

> **Scope:** This is an educational prototype built with simulated data. It is
> not a validated alarm system and should not be used to make real security or
> emergency-response decisions.

## Features

- Generate a reproducible 20,000-row sensor dataset.
- Fit a discrete Bayesian Network and inspect its conditional probability
  tables (CPTs).
- Submit sensor states and view posterior class probabilities and a risk level.
- Compare the Bayesian Network with a simple rule-based baseline.
- View evaluation metrics, confusion matrices, class distribution, and dataset
  summaries in either dashboard.

## Requirements

- Python 3.9 or later
- Packages listed in [`requirements.txt`](requirements.txt)

## Quick start

Run commands from the repository root.

### 1. Create an environment and install packages

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### 2. Generate the data

```bash
python generate_dataset.py
```

This writes `data/sensor_data.csv`.

### 3. Train and save the Bayesian Network

```bash
python train_model.py
```

This reads the generated CSV and writes `data/bayesian_model.pkl`.

### 4. Evaluate the models and generate charts

```bash
python evaluate.py
```

This writes `data/evaluation_metrics.csv` and the charts in `data/plots/`.

### 5. Start a dashboard

Choose either interface:

```bash
# Flask website at http://127.0.0.1:5000
python app_flask.py

# Or Streamlit dashboard at the URL printed by Streamlit
streamlit run app.py
```

The dashboards need the trained model. The evaluation and dataset sections also
use the generated CSV, metrics, and plot files when they are available.

## Project structure

```text
infera/
├── app.py                  # Streamlit dashboard
├── app_flask.py            # Flask routes and prediction API
├── evaluate.py             # Baseline comparison, metrics, and plots
├── generate_dataset.py     # Synthetic sensor data
├── sanity_check.py         # Example inference scenarios
├── train_model.py          # Bayesian Network training and inference
├── requirements.txt
├── README.md
├── docs/
│   └── PROJECT_GUIDE.md    # Detailed architecture and implementation guide
├── templates/
│   └── index.html          # Flask dashboard template
└── data/
    ├── sensor_data.csv
    ├── bayesian_model.pkl
    ├── evaluation_metrics.csv
    └── plots/
        ├── class_distribution.png
        ├── comparison_metrics.png
        └── confusion_matrices.png
```

Generated files under `data/` are used by later stages. Re-run the stages in
order after changing the generator or model:

```text
generate_dataset.py -> train_model.py -> evaluate.py -> dashboard
```

## How predictions work

The model's graph is:

```text
                 +--> door_sensor
entry_status ----+--> motion_sensor
                 +--> glass_sensor
```

Training estimates the prior distribution of `entry_status` and the conditional
probabilities of each sensor given that class. At prediction time, the three
observations are evidence, and variable elimination calculates
`P(entry_status | door_sensor, motion_sensor, glass_sensor)`.

The displayed risk level is based on the posterior probability of the
`intrusion` class:

| Intrusion probability | Risk level |
| --- | --- |
| Below 30% | Low |
| 30% through below 70% | Moderate |
| 70% or higher | High |

For implementation details, data definitions, evaluation methodology, API
fields, and known limitations, see the
[project guide](docs/PROJECT_GUIDE.md).

## Sanity checks

After generating the data and training the model, run:

```bash
python sanity_check.py
```

The script prints probabilities for several fixed sensor combinations. It is a
quick smoke check, not a comprehensive automated test suite.
