# Social Media Platform Classification

**Contributors:** Marcus Hedlund, Milo Shan

## Project Overview

This project studies whether a post's text alone is enough to identify which
social media platform it came from. We compare posts from Twitter, Reddit, and
Hacker News and train three supervised models to predict the source platform.

The project intentionally avoids using platform-specific metadata such as
author IDs, likes, subreddit names, timestamps as predictive inputs, or other
source fields that would make the task easier for the wrong reasons. Instead,
the models focus on text content, punctuation, visible platform conventions,
and engineered text statistics.

## Main Questions

- How separable are Twitter, Reddit, and Hacker News posts using text alone?
- Which stylistic or structural signals seem to matter most?
- How much do performance and interpretability change across a linear baseline,
  a tree-based embedding model, and a neural network?

## Repository Map

- `notebooks/1_load_and_explore_data.ipynb`
  Source-level inspection of the raw datasets and the assumptions that drive
  later cleaning decisions.
- `notebooks/2_create_datasets.ipynb`
  Dataset construction pipeline, cleaning, balancing, HTML audit, and
  train/validation/test split generation.
- `notebooks/3_eda_pca_clustering.ipynb`
  Main EDA notebook with platform summaries, stylistic-marker analysis,
  temporal analysis, and PCA on engineered text statistics.
- `notebooks/4_logistic_regression_baseline.ipynb`
  Interpretable TF-IDF + text-stat logistic-regression baseline.
- `notebooks/5_xgboost_baseline.ipynb`
  XGBoost model trained on MiniLM sentence embeddings, including example-based
  interpretation of important latent dimensions.
- `notebooks/6_neural_network_model.ipynb`
  MLP classifier using MiniLM sentence embeddings and engineered text
  statistics, with early stopping and a lightweight tuning sweep.
- `dashboard.py`
  Dash application used for the final project demo.
- `outputs/eda/`
  Exported plots and tables used by the dashboard.
- `outputs/modeling/`
  Saved metrics, plots, and model artifacts from the modeling notebooks.

## Data Artifacts

The processed dataset artifacts used by the modeling notebooks live under:

- `data/processed/final_dataset_local_duckdb/social_platform_dataset_300k.parquet`
- `data/processed/final_dataset_local_duckdb/train_210k.parquet`
- `data/processed/final_dataset_local_duckdb/val_45k.parquet`
- `data/processed/final_dataset_local_duckdb/test_45k.parquet`

The dataset-construction notebook is the canonical place to explain how those
files are produced and cleaned.

## Models

- **Logistic Regression**
  Sparse TF-IDF baseline with engineered text statistics. This is the most
  interpretable model.
- **XGBoost**
  Nonlinear classifier on MiniLM sentence embeddings. This captures richer
  semantic and stylistic structure than the sparse baseline.
- **Neural Network**
  MLP over MiniLM embeddings plus engineered text statistics. This is the most
  flexible model in the project.

Each model notebook now includes a small validation-based hyperparameter sweep
before the final training run.

## Dashboard

An interactive project dashboard is available in `dashboard.py`. It covers the project overview, EDA, model comparison, coefficient-based interpretation, and real post examples from the dataset.

Suggested demo flow:

1. Start with **Overview** for the objective, rubric coverage, and headline results.
2. Use **EDA Explorer** to filter platforms and show how text length, hashtags, mentions, URLs, and time coverage differ.
3. Use **Modeling** to compare Logistic Regression, XGBoost, and the Neural Network across accuracy, F1, precision, and recall.
4. Use **Feature Lens** to click through Logistic Regression coefficients or XGBoost embedding dimensions and inspect matching real posts.

### Run

1. Install dependencies:
   - `python3 -m pip install -r requirements.txt`
2. Start the dashboard:
   - `python3 dashboard.py`
3. Open:
   - [http://127.0.0.1:8050](http://127.0.0.1:8050)
