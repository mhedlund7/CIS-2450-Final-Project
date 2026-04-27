"""
Final project dashboard for CIS 2450.

Run with:
    python dashboard.py
"""

from __future__ import annotations
import base64
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "modeling"
NN_OUTPUT_DIR = OUTPUT_ROOT / "neural_network"
LOGREG_OUTPUT_DIR = OUTPUT_ROOT / "logistic_regression"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


MODEL_METRICS = [
    {
        "model": "Logistic Regression (TF-IDF + text stats)",
        "validation_accuracy": 0.8835,
        "test_accuracy": 0.8844,
        "macro_f1": 0.89,
    },
    {
        "model": "XGBoost (MiniLM embeddings + text stats)",
        "validation_accuracy": 0.8942,
        "test_accuracy": 0.8946,
        "macro_f1": 0.89,
    },
    {
        "model": "Neural Network (MiniLM embeddings + text stats)",
        "validation_accuracy": 0.9478,
        "test_accuracy": 0.9466,
        "macro_f1": 0.95,
    },
]


PER_CLASS_METRICS = pd.DataFrame(
    [
        # Approximate values from each notebook's classification report based on the notebook outputs.
        {"model": "Logistic Regression", "platform": "hackernews", "precision": 0.88, "recall": 0.89, "f1_score": 0.89},
        {"model": "Logistic Regression", "platform": "reddit", "precision": 0.87, "recall": 0.86, "f1_score": 0.86},
        {"model": "Logistic Regression", "platform": "twitter", "precision": 0.91, "recall": 0.90, "f1_score": 0.90},
        {"model": "XGBoost", "platform": "hackernews", "precision": 0.89, "recall": 0.90, "f1_score": 0.89},
        {"model": "XGBoost", "platform": "reddit", "precision": 0.88, "recall": 0.88, "f1_score": 0.88},
        {"model": "XGBoost", "platform": "twitter", "precision": 0.92, "recall": 0.90, "f1_score": 0.91},
        {"model": "Neural Network", "platform": "hackernews", "precision": 0.94, "recall": 0.95, "f1_score": 0.95},
        {"model": "Neural Network", "platform": "reddit", "precision": 0.95, "recall": 0.94, "f1_score": 0.95},
        {"model": "Neural Network", "platform": "twitter", "precision": 0.95, "recall": 0.95, "f1_score": 0.95},
    ]
)


def load_nn_metrics() -> dict:
    metrics_path = NN_OUTPUT_DIR / "nn_embedding_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_nn_history() -> pd.DataFrame:
    history_path = NN_OUTPUT_DIR / "nn_training_history.csv"
    if not history_path.exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


def load_logreg_metrics() -> dict:
    metrics_path = LOGREG_OUTPUT_DIR / "logreg_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_date_range_summary() -> pd.DataFrame:
    summary_path = EDA_OUTPUT_DIR / "date_range_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame([{"status": "Run the EDA notebook to export date_range_summary.csv"}])
    return pd.read_csv(summary_path)


def make_metric_cards(df: pd.DataFrame) -> list[html.Div]:
    cards: list[html.Div] = []
    for _, row in df.iterrows():
        cards.append(
            html.Div(
                [
                    html.H3(row["model"], style={"marginBottom": "8px"}),
                    html.Div(f"Validation Accuracy: {row['validation_accuracy']:.4f}"),
                    html.Div(f"Test Accuracy: {row['test_accuracy']:.4f}"),
                    html.Div(f"Macro F1: {row['macro_f1']:.2f}"),
                ],
                className="metric-card",
            )
        )
    return cards


def image_card(image_name: str, title: str) -> html.Div:
    image_path = EDA_OUTPUT_DIR / image_name
    if not image_path.exists():
        return html.Div(
            [
                html.H3(title),
                html.P(f"Missing {image_name}. Re-run EDA notebook exports."),
            ],
            className="card",
        )
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    src = f"data:image/png;base64,{encoded}"
    return html.Div(
        [
            html.H3(title),
            html.Img(src=src, className="eda-image"),
        ],
        className="card",
    )


def insight_card(title: str, points: list[str]) -> html.Div:
    return html.Div(
        [
            html.H3(title, className="insight-title"),
            html.Ul([html.Li(point) for point in points], className="insight-list"),
        ],
        className="card insight-card",
    )


def build_app() -> Dash:
    app = Dash(__name__)
    metrics_df = pd.DataFrame(MODEL_METRICS)
    nn_metrics = load_nn_metrics()
    nn_history = load_nn_history()
    logreg_metrics = load_logreg_metrics()
    date_range_summary = load_date_range_summary()

    if logreg_metrics:
        metrics_df.loc[
            metrics_df["model"].str.startswith("Logistic Regression"),
            ["validation_accuracy", "test_accuracy", "macro_f1"],
        ] = [
            logreg_metrics.get("validation_accuracy", 0.8835),
            logreg_metrics.get("test_accuracy", 0.8844),
            logreg_metrics.get("macro_f1", 0.89),
        ]

    if nn_metrics:
        metrics_df.loc[
            metrics_df["model"].str.startswith("Neural Network"),
            ["validation_accuracy", "test_accuracy"],
        ] = [
            nn_metrics.get("validation_accuracy", 0.9478),
            nn_metrics.get("test_accuracy", 0.9466),
        ]

    comparison_df = metrics_df.melt(
        id_vars=["model"],
        value_vars=["validation_accuracy", "test_accuracy", "macro_f1"],
        var_name="metric",
        value_name="value",
    )
    comparison_fig = px.bar(
        comparison_df,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model Comparison: Accuracy and Macro F1",
        text_auto=".3f",
    )
    comparison_fig.update_layout(yaxis_title="Score", xaxis_title="")
    comparison_fig.update_yaxes(range=[0.80, 1.0])

    if not nn_history.empty:
        nn_history_fig = px.line(
            nn_history,
            x="epoch",
            y=["train_accuracy", "val_accuracy"],
            title="Neural Network Training Curve",
            markers=True,
        )
        nn_history_fig.update_layout(yaxis_title="Accuracy", xaxis_title="Epoch")
    else:
        nn_history_fig = px.scatter(title="Neural Network history file not found")

    heatmap_df = PER_CLASS_METRICS.pivot_table(
        index=["model", "platform"], values="f1_score", aggfunc="mean"
    ).reset_index()
    per_class_fig = px.density_heatmap(
        heatmap_df,
        x="platform",
        y="model",
        z="f1_score",
        color_continuous_scale="Blues",
        text_auto=".2f",
        title="Per-Class F1 Score by Model",
    )
    per_class_fig.update_layout(xaxis_title="Platform", yaxis_title="")

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("CIS 2450 Final Project Dashboard", className="header-title"),
                    html.P(
                        "Classifying social-media post platform (Twitter, Reddit, HackerNews) "
                        "using Logistic Regression, XGBoost, and a Neural Network.",
                        className="header-subtitle",
                    ),
                ],
                className="header-card",
            ),
            html.Details(
                [
                    html.Summary("Overview", className="section-title"),
                    html.Div(
                        [
                            html.P(
                                "For our project we investigate the difference in the language used across various social media platforms."
                                "The objective is to build and compare several models to classify what social media "
                                "platform an input post comes from, as well as to try and distill what features or "
                                "qualities influence that decision. We analyze Logistic Regression, XGBoost, and Neural "
                                "Network models. This project is interesting because it can give insight into the "
                                "different vocabularies and linguistic styles that arise separately across different "
                                "platforms, and it's also just fun to look at the different post styles.",
                                className="overview-line",
                            ),
                        ],
                        className="card",
                    ),
                ],
                open=True,
                className="section-details overview-section",
            ),
            html.Details(
                [
                    html.Summary("EDA and Data", className="section-title"),
                    image_card("class_balance.png", "Class Balance by Platform"),
                    image_card(
                        "text_length_hist.png",
                        "Text Length Distributions (Capped at 2000, Evenly Spaced Bins)",
                    ),
                    image_card("style_avg_word_len_bar.png", "Average Word Length by Platform"),
                    image_card("style_stats_bar.png", "Average Stylistic Marker Counts by Platform"),
                    image_card("year_month_counts.png", "Posts by Year-Month and Platform"),
                    html.Div(
                        [
                            html.H3("Earliest and Latest Entry by Platform"),
                            dash_table.DataTable(
                                data=date_range_summary.to_dict("records"),
                                columns=[
                                    {"name": c.replace("_", " ").title(), "id": c}
                                    for c in date_range_summary.columns
                                ],
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left", "padding": "10px"},
                                style_header={"fontWeight": "bold"},
                            ),
                        ],
                        className="card",
                    ),
                ],
                open=True,
                className="section-details",
            ),
            html.Details(
                [
                    html.Summary("Model and Results", className="section-title"),
                    html.Div(
                        make_metric_cards(metrics_df),
                        className="metrics-grid",
                    ),
                    html.Div(
                        dcc.Graph(figure=comparison_fig, config={"displayModeBar": False}),
                        className="card",
                    ),
                    html.Div(
                        dcc.Graph(figure=per_class_fig, config={"displayModeBar": False}),
                        className="card",
                    ),
                    html.Div(
                        dcc.Graph(figure=nn_history_fig, config={"displayModeBar": False}),
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H3("Raw Model Metrics Table"),
                            dash_table.DataTable(
                                data=metrics_df.round(4).to_dict("records"),
                                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in metrics_df.columns],
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left", "padding": "10px"},
                                style_header={"fontWeight": "bold"},
                            ),
                        ],
                        className="card",
                    )
                ],
                open=True,
                className="section-details",
            ),
        ],
        className="dashboard-container",
    )
    return app


if __name__ == "__main__":
    app = build_app()
    app.run(debug=True, port=8050)
