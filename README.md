# Social Media Platform Classification

**Contributors:** Marcus Hedlund, Milo Shan

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
