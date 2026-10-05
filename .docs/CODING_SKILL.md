# CODING_SKILL.md

Style guide for writing Python code in AI/ML projects. Mimic the conventions below exactly. This file defines **form only**: structure, naming, spacing, comments, docstrings, logging, error handling. Never copy concrete values, names, paths, or domain logic from examples. Placeholders look like `<this>`.

---

## 1. Core Principles

1. Python is the primary language. Target Python 3.10+.
2. Config-driven: every tunable value (paths, hyperparameters, thresholds, column lists, seeds, URIs) comes from the config file or a constants module. No magic numbers or literals inline.
3. One class = one responsibility. One main class per file.
4. Small, single-purpose methods. Orchestrators call them in sequence.
5. Every method that can fail is wrapped in `try/except`, logs, and re-raises.
6. Log generously; use `print` never.
7. Dependencies are injected through `__init__`, not created ad hoc inside methods.

---

## 2. Project Layout

Layered, one package per concern. Every package has an `__init__.py`.

```
src/
├── config/              # config loader + yaml
├── data/                # reader, preprocessor
├── evaluation/          # metrics
├── logger/              # logger setup (in __init__.py)
├── model/               # model selection / registry logic
├── path_constants/      # all path constants (in __init__.py)
├── pipelines/           # pipeline generators, feature engineering, orchestrators
├── training/            # model trainer
├── utils/               # artifact + experiment-tracking helpers
├── visualization/
└── __init__.py
```

Rules:
- Files: `snake_case.py`, named after the noun they implement (`data_reader.py`, `model_training.py`).
- Classes: `PascalCase`, noun or noun-phrase (`DataReader`, `ModelTrainer`, `CalculateMetrics`).
- Methods/variables: `snake_case`, verb-first for methods (`load_data`, `save_model`, `pick_best_model`).
- Constants: `UPPER_SNAKE_CASE`, defined only in the path/constants module.
- `__init__.py` is empty unless it hosts shared singletons (logger, path constants).
- Imports always absolute from the package root (`from src.x.y import Z`). Never relative.

---

## 3. Imports

Sorted isort-style, three groups separated by one blank line:

1. Standard library
2. Third-party
3. Local (`src.*`)

```python
import json
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config_loader
from src.logger import logging
from src.utils.artifacts_setup import ArtifactsSetup
```

Rules:
- `import x` lines first, then `from x import y` lines, alphabetical within each group.
- Multiple names from one module: wrap in parentheses, aligned under the first name, no trailing comma.
- A single long import that overflows uses a backslash continuation.

```python
from src.path_constants import (PROCESSED_TEST_DATA_PATH,
                                PROCESSED_TRAIN_DATA_PATH)
from src.pipelines.some_long_module_name import \
    SomeVeryLongClassName
```

- Import the project logger as `from src.logger import logging` and use `logging.<level>(...)` everywhere.
- Import config via the module (`from src.config import config_loader`) and call `config_loader.load_config()`.
- No wildcard imports. No unused imports.

---

## 4. Whitespace and Layout

| Situation | Rule |
|---|---|
| Top-level definitions (classes, functions) | 2 blank lines between |
| Methods inside a class | 1 blank line between |
| After class docstring / class line | 1 blank line before first method |
| Inside methods | 1 blank line between logical blocks |
| After imports | 2 blank lines before first definition |
| End of file | exactly 1 trailing newline, no trailing whitespace on any line |
| Indentation | 4 spaces, never tabs |
| Max line length | ~100 chars; break long calls over lines |
| Quotes | double quotes `"` for all strings |

Multi-line calls: one argument per line, 4-space hanging indent, closing paren on its own line at the call's indent, no trailing comma.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=stratify_column
)
```

Keyword arguments: `key=value`, no spaces around `=`.
Operators: single space around binary operators and `=` in assignments.
Dict literals: one entry per line when more than ~2 entries, `"key": value`.

```python
self.metrics = {
    "<metric_a>": value_a,
    "<metric_b>": value_b,
    "<metric_c>": value_c
}
```

Tuple returns are parenthesized:

```python
return (X_train, X_test, y_train, y_test)
```

---

## 5. Type Hints

- Annotate all parameters and return types on public methods.
- Use `pd.DataFrame`, `pd.Series`, `Path`, `dict`, `float`, `str`, `bool`.
- Use `typing.Tuple` / `Optional` for compound types.
- Dependency parameters are annotated with their class (`artifacts_setup: ArtifactsSetup`).
- Defaults go after the annotation: `test_size: float = 0.2`.
- Multi-parameter signatures wrap one parameter per line when too long.

```python
def data_train_test_split(
        self,
        data: pd.DataFrame,
        target_column_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
```

(Defaults shown here only demonstrate syntax. In real code, pass the value from config.)

---

## 6. Docstrings

Triple double quotes. Placed immediately under the `class` / `def` line.

**Class docstring:** one or two lines describing purpose.

```python
class <ClassName>:
    """
    <One-line description of what the class does>
    """
```

**Method docstring, simple:** single descriptive sentence, starts with a verb.

```python
def <method>(self):
    """
        Function to <do something> and then returns <what>
    """
```

**Method docstring, rich (for public APIs with params):** summary line, blank line, `Args:` and `Returns:` blocks.

```python
def <method>(self, X: pd.DataFrame, dataset: str = "<name>") -> pd.Series:
    """
    <Summary sentence>

    Args:
        X: <what it is>
        dataset: <what it is used for>

    Returns:
        <what is returned>
    """
```

Rules:
- Every class and every public method has a docstring.
- Describe *what* it does and what it returns; do not narrate implementation.
- Private/trivial helpers may skip it.

---

## 7. Comments

- Inline comments: two spaces after code, `# lowercase or sentence case short phrase`.
- Block comments: on their own line above the code they explain, same indent.
- Explain **why** or label a **stage**, not what obvious code does.
- Stage banners in orchestrators use a consistent label:

```python
# STAGE 1: <Stage Name>
logging.info("\n[STAGE 1] <Stage Name>")
```

- Short intent comments before non-obvious blocks:

```python
# Remove last element (no threshold for last point)
# Condition: <rule being enforced>
# fallback to <alternative> if condition not met
# Convert numpy types to native python types
# Overwrite <file> in current run folder
```

- No commented-out dead code. No TODO without context.

---

## 8. Class Design Pattern

Standard shape for any component class:

```python
class <ComponentName>:
    """
    <Purpose>
    """

    def __init__(self, artifacts_setup: ArtifactsSetup):
        config = config_loader.load_config()

        self.<value_a> = config["<section>"]["<key_a>"]
        self.<value_b> = config["<section>"]["<key_b>"]

        self.artifacts_setup = artifacts_setup
        self.artifacts_path = self.artifacts_setup.artifact_path

    def <action>(self, <args>):
        """
        <What it does>
        """

        try:
            logging.info("<Starting action>")

            <logic>

            logging.info("<Successfully finished action>")
            return <result>

        except Exception as e:
            logging.error(f"<Could not do action> due to error: {e}")
            raise
```

Rules:
- Read config once in `__init__`, store each value as a `self.<name>` attribute, use the attribute later.
- Share one artifacts/context object across components by passing it into constructors.
- Initialization logs a `logging.debug("Initializing <thing>")` line when the setup is non-trivial.
- Fail fast in `__init__` for missing prerequisites (e.g. raise `FileNotFoundError` with a clear message).
- Path handling uses `pathlib.Path` and the `/` operator, never string concatenation.

---

## 9. Error Handling

- Wrap the **entire body** of each public method in one `try/except Exception as e`.
- Log with `logging.error(...)` including the exception, then `raise` (bare, preserve traceback).
- Orchestrator top-level catch logs with `exc_info=True`.
- Message templates (keep wording style):
  - `f"Error during <operation>: {e}"`
  - `f"Could not <operation> due to error: {e}"`
  - `f"Failed to <operation>: {e}"`
  - `f"Error while <operation> to {path}: {e}"`
- Non-critical operations (e.g. optional artifact save) may log and not re-raise; critical ones always re-raise.
- Bad-but-skippable data: `logging.warning(...)` then `continue`.
- Raise specific exceptions with explanatory f-string messages (`ValueError`, `FileNotFoundError`).
- Prefer `except Exception as e`; avoid bare `except:`.

---

## 10. Logging

- One shared logger module configures root logger with console + rotating file handler, formatter and level taken from config.
- Level semantics:
  - `info`: stage starts, major actions, final results, selected choices
  - `debug`: shapes, intermediate values, sub-steps, config values
  - `warning`: skipped/corrupt/incompatible items
  - `error`: inside `except` blocks
- Message style: sentence case, no trailing period required, present-continuous for start, "Successfully ..." for completion.
- Always f-strings with values; format floats explicitly: `{value:.6f}`, `{elapsed:.2f}`.
- Prefix dataset-scoped logs with a tag: `f"[{dataset.upper()}] Metric: {value:.6f}"`.
- Log shape after every data transformation: `logging.debug(f"<name> shape: {df.shape}")`.
- Log timing for expensive steps:

```python
start_time = time.time()
logging.info("<Step> started")
<work>
elapsed_time = time.time() - start_time
logging.info(f"<Step> completed in {elapsed_time:.2f} seconds")
```

---

## 11. Configuration Conventions

- Single YAML config, loaded via a tiny loader function with a docstring.
- Top-level sections by concern: project, seed, schema, logging, features, training, model, metrics, tracking, inference.
- Access pattern: `config["section"]["key"]`, nested for params.
- All paths live in a constants module built from a `ROOT_DIR` `Path`, then composed with `/`.
- A global seed from config is passed to every random component (`random_state=...`).
- Hyperparameters are never written in Python; they are read from config and passed through.

---

## 12. AI/ML Specific Conventions

### Variable naming
- Features uppercase `X`, target lowercase `y`.
- Suffixes for splits: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`.
- Transformed data gets `_t` suffix: `X_train_t`.
- Predictions: `y_pred`, probabilities: `y_probabilities` / `y_<split>_prob`.
- Threshold variable: `threshold` / `best_threshold`.

### Data
- Loading, splitting, and preprocessing are separate classes.
- Split with stratification controlled by config; split twice (train/test, then train/validation).
- Always `.drop(columns=..., errors="ignore")` for configured drop lists.
- Preprocessor class exposes `fit_transform` (train only) and `transform` (val/test), and returns a `pd.DataFrame` with proper column names and `index=X.index`, cast with `.astype(float)`.
- Persist interim and processed data and the fitted preprocessor as artifacts.

### Sklearn pipelines
- Build pipelines in a dedicated generator class with one method per sub-pipeline (`generate_<kind>_pipeline`) plus a final `generate_pipeline`.
- Use `Pipeline(steps=[(...), (...)])` with named, snake_case step names.
- Use `ColumnTransformer(transformers=[...], remainder="drop")`.
- Custom transformers subclass `BaseEstimator, TransformerMixin`:

```python
class <FeatureGenerator>(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["<new_col>"] = <expression>

        return X
```

### Model training
- Trainer class wraps the estimator, builds it in `__init__` from config values, and exposes `fit`, `predict`, `predict_probability`, `get_params`, `get_model`, `save_model`, `save_params`.
- Probability method returns positive-class column: `predict_proba(X)[:, 1]`.

### Evaluation
- Dedicated metrics class: `calculate_thresholds`, `evaluate`, `save_metrics_plots`.
- Thresholds are tuned on **validation**, applied to **test**.
- `evaluate` returns a dict of scalar metrics plus confusion matrix as `.tolist()`.
- Plots saved with matplotlib: `plt.figure(figsize=(...))`, labelled axes, title, legend, `plt.tight_layout()`, `plt.savefig(path)`, `plt.close()`.
- Convert numpy scalars to native Python before JSON/MLflow:

```python
serializable_metrics = {}
for k, v in metrics.items():
    if isinstance(v, (np.floating, np.integer)):
        serializable_metrics[k] = v.item()
    else:
        serializable_metrics[k] = v
```

### Artifacts and experiment tracking
- Each run gets a timestamped artifact directory: `artifacts/<model_name>/<timestamp>/` with a `plots/` subfolder.
- Artifact helper class owns all save operations (`save_pipeline`, `save_model`, `save_json`, `save_csv_artifact`) using `joblib` for binaries and `json.dump(..., indent=4)` for JSON.
- Tracking helper class wraps MLflow: `get_run_name`, `start_run`, `log_params`, `log_metrics`, `log_artifacts`, `log_model`, each in try/except with debug logs.
- Runs execute inside `with <tracker>.start_run(run_name):`.
- Metrics files carry a schema/version field so incompatible runs can be skipped when selecting the best model.
- Best-model selection iterates run folders, skips missing/empty/corrupt/incompatible, applies a config-defined metric, copies winners to a stable `models/` location.
- Secrets/env via `python-dotenv` `load_dotenv()` at module top; never inline credentials.

---

## 13. Pipeline Orchestrator Pattern

- One orchestrator class constructs all components in `__init__` and stores them as attributes.
- Each stage is a small method (`split_data`, `preprocess_data`, `train_model`, `tune_threshold`, `evaluate_model`).
- `run_pipeline` calls stages in order with numbered stage banners, timing, a final summary, and a single outer `try/except` that logs with `exc_info=True` and re-raises.

```python
def run_pipeline(self):
    try:
        pipeline_start_time = time.time()
        logging.info("<PIPELINE NAME> STARTED")

        # STAGE 1: <Name>
        logging.info("\n[STAGE 1] <Name>")
        <call>

        # STAGE 2: <Name>
        logging.info("\n[STAGE 2] <Name>")
        <call>

        pipeline_elapsed_time = time.time() - pipeline_start_time
        logging.info(f"Total Pipeline Time: {pipeline_elapsed_time:.2f} seconds")
        logging.info("<PIPELINE NAME> COMPLETED SUCCESSFULLY")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise
```

---

## 14. Checklist Before Finishing Any File

- [ ] Imports grouped (stdlib / third-party / local), alphabetized, absolute
- [ ] 2 blank lines between top-level defs, 1 between methods
- [ ] No trailing whitespace, single trailing newline
- [ ] Double quotes everywhere, f-strings for formatted text
- [ ] Type hints on public signatures
- [ ] Class and public-method docstrings present
- [ ] Whole method body in `try/except` → `logging.error` → `raise`
- [ ] Appropriate `info` / `debug` / `warning` logging, floats formatted
- [ ] Every value from config or constants, nothing hardcoded
- [ ] `pathlib.Path` for paths
- [ ] Naming conventions: `X`/`y`, `_t` suffix, snake_case methods, PascalCase classes
- [ ] No dead code, no `print`, no wildcard imports
