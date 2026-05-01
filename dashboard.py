"""
Interactive final project dashboard for CIS 2450.

Run with:
    python3 dashboard.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset_local_duckdb"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling"
LOGREG_OUTPUT_DIR = MODEL_OUTPUT_DIR / "logistic_regression"
NN_OUTPUT_DIR = MODEL_OUTPUT_DIR / "neural_network"
XGB_METRICS_PATH = MODEL_OUTPUT_DIR / "xgboost_embedding_metrics.json"
XGB_TOP_DIMENSIONS_PATH = MODEL_OUTPUT_DIR / "xgboost_top_dimensions.csv"
XGB_DIMENSION_EXAMPLES_PATH = MODEL_OUTPUT_DIR / "xgboost_top_dimension_examples.csv"
LOGREG_TOP_FEATURES_PATH = LOGREG_OUTPUT_DIR / "logreg_top_features.csv"

PLATFORMS = ["hackernews", "reddit", "twitter"]
PLATFORM_LABELS = {
    "hackernews": "Hacker News",
    "reddit": "Reddit",
    "twitter": "Twitter",
}
PLATFORM_COLORS = {
    "hackernews": "#0f766e",
    "reddit": "#c2410c",
    "twitter": "#2563eb",
}

STYLE_METRICS = {
    "text_len": "Post length",
    "num_words": "Word count",
    "num_questions": "Question marks",
    "num_exclamations": "Exclamation marks",
    "num_hashtags": "Hashtags",
    "num_mentions": "Mentions",
    "num_urls": "URLs",
    "avg_word_len": "Average word length",
}

# Setup our model outputs from the notebooks for reference in the dashboard.


LOGREG_TOP_FEATURES = {'hackernews': {'com': 5.801167,
                'org': 4.76056,
                'article': 3.21432,
                'hn': 3.112926,
                'www': 2.962313,
                'probably': 2.802742,
                'github': 2.788735,
                'point': 2.761451,
                'people': 2.759692,
                'google': 2.611358,
                'doesn': 2.523546,
                'yes': 2.407572,
                'interesting': 2.353612,
                'yeah': 2.348904,
                'isn': 2.344044},
 'reddit': {'telegram': 3.014149,
            'looking': 2.904652,
            'help': 2.568728,
            'overview': 2.475501,
            'guys': 2.290458,
            'num_words': 2.249691,
            'girl': 2.189551,
            'poll': 2.044982,
            'reddit': 2.039501,
            'dm': 2.032964,
            'wondering': 2.003841,
            'wanna': 1.876439,
            'sub': 1.876273,
            'reddit com': 1.853424,
            'cum': 1.843836},
 'twitter': {'https': 14.157866,
             'http': 6.082246,
             'num_mentions': 2.765396,
             'la': 2.271574,
             '00': 2.143079,
             'que': 1.976868,
             'el': 1.526586,
             'today': 1.349722,
             'week': 1.272876,
             'le': 1.221238,
             'tonight': 1.178061,
             'في': 1.171919,
             'les': 1.168812,
             'おはようございます': 1.160835,
             'et': 1.142039}}

XGB_TOP_DIMENSIONS = [
    {"dimension": 250, "importance": 0.062896},
    {"dimension": 11, "importance": 0.031529},
    {"dimension": 176, "importance": 0.029031},
    {"dimension": 224, "importance": 0.024004},
    {"dimension": 223, "importance": 0.017113},
]

XGB_DIMENSION_SUMMARIES = {
    250: "High values look like short promotional or socially tagged posts with mentions, hashtags, and callout wording. Low values are longer explanatory Reddit or Hacker News posts.",
    11: "High values lean toward discussion-oriented Hacker News prose. Low values look more like title-style Reddit community posts, recruitment posts, or self-presentation.",
    176: "High values include concise update or event language, often with links or mentions. Low values are longer question-heavy Reddit discussions.",
    224: "High values capture casual help-seeking or everyday social text. Low values lean toward analytical or critical Hacker News commentary.",
    223: "This dimension is harder to summarize cleanly, which is a useful reminder that some latent embedding coordinates help XGBoost without mapping to one simple human label.",
}

XGB_DIMENSION_EXAMPLES = [{'dimension': 250,
  'direction': 'highest',
  'value': 0.268829,
  'platform': 'reddit',
  'text_preview': 'Instagram: "¥$: The Vultures Experience. RUN UP MY POST. @rory_mitch203 None'},
 {'dimension': 250,
  'direction': 'highest',
  'value': 0.258527,
  'platform': 'twitter',
  'text_preview': '@DKlarations @davewolfusa @TheFavoritist @NASASpaceflight @ChrisG_NSF and to the AWESOME editor who '
                  'pulled the video together!'},
 {'dimension': 250,
  'direction': 'highest',
  'value': 0.251255,
  'platform': 'twitter',
  'text_preview': 'Looking forward to the match @SUFC_tweets tonight with our guests from #littlehelper'},
 {'dimension': 250,
  'direction': 'lowest',
  'value': -0.252174,
  'platform': 'reddit',
  'text_preview': 'Why are there so many boulders in certain places I don’t understand why some mountains or other '
                  'hilly areas will have excessive amounts of boulders but others won’t, in some places it seems like '
                  'it’s an entire hill is made up of boulders. I would love to know why this is'},
 {'dimension': 250,
  'direction': 'lowest',
  'value': -0.242158,
  'platform': 'hackernews',
  'text_preview': 'The demand side became pretty proficient regarding load shedding as well, making literally millions '
                  'off of it.'},
 {'dimension': 250,
  'direction': 'lowest',
  'value': -0.240736,
  'platform': 'hackernews',
  'text_preview': 'In the Boston area many of the nearby towns are what I call "urban suburbs". Very family friendly, '
                  "many single- or two-family homes, but also very walkable and many shops throughout the town. I'm "
                  'talking about towns such as Brookline, Cambridge and Arlington. They are really nice towns to live '
                  'in and in very high demand.'},
 {'dimension': 11,
  'direction': 'highest',
  'value': 0.264749,
  'platform': 'hackernews',
  'text_preview': "Yep if you want to make it more attractive don't break the browser for anyone trying to read your "
                  "blog. Page Up/Down did nothing, there was no indication on how to navigate this page. I didn't read "
                  'it.'},
 {'dimension': 11,
  'direction': 'highest',
  'value': 0.236429,
  'platform': 'hackernews',
  'text_preview': "The market for used pre-2010 physical books will be huge. Personally I've stopped buying online "
                  'books because of this kind of shenanigans and reverting back to having local music and movies for '
                  "similar reasons. The idea that some remote working technocrats can change words of a book while I'm "
                  'reading it just creeps me out.'},
 {'dimension': 11,
  'direction': 'highest',
  'value': 0.234883,
  'platform': 'hackernews',
  'text_preview': "If anyone's using Firefox it'll mess with everyone's performance. They're working on it, should be "
                  'okay in the next month or so.'},
 {'dimension': 11,
  'direction': 'lowest',
  'value': -0.240438,
  'platform': 'reddit',
  'text_preview': "I Made Myself Sound Like A Darkin ( Rhaast's Race ) None"},
 {'dimension': 11,
  'direction': 'lowest',
  'value': -0.236251,
  'platform': 'reddit',
  'text_preview': '(Kik: UsernameDotCom69) Who’s ready to fall in love with a Kpop idol? | catfish None'},
 {'dimension': 11,
  'direction': 'lowest',
  'value': -0.232689,
  'platform': 'reddit',
  'text_preview': '[EUW] Tier II jungler looking for a team to join My preferred role is jungle, I can still also play '
                  'toplane and support if need be ( I can be a flexpick) my jungle mains are: Nunu, Skarner, Rammus '
                  'and Warwick. \u200b My summoner name is: Witlof Man \u200b I dont flame and Im pretty chill '
                  'overall'},
 {'dimension': 176,
  'direction': 'highest',
  'value': 0.303086,
  'platform': 'hackernews',
  'text_preview': 'From talking to Anton, it seems like GitHub integration is next up on the TODO list. Any feedback '
                  "on whether you'd find that useful and how you think it should be done would be really appreciated."},
 {'dimension': 176,
  'direction': 'highest',
  'value': 0.270384,
  'platform': 'twitter',
  'text_preview': '@ericamou signing copies for subscribers, pre-orders and website orders❤️ https://t.co/oGscNQaUTf'},
 {'dimension': 176,
  'direction': 'highest',
  'value': 0.260175,
  'platform': 'twitter',
  'text_preview': 'His Excellency President @CyrilRamaphosa arriving at the Sefako Makgatho Presidential Guesthouse in '
                  'Tshwane to receive #LettersofCredence from Heads of Mission-Designate #BetterAfricaBetterWorld 🌍 '
                  'https://t.co/mU83iyzFgv'},
 {'dimension': 176,
  'direction': 'lowest',
  'value': -0.250758,
  'platform': 'reddit',
  'text_preview': "Question about clear times Hey Hunters! I've noticed that quite a lot of you, when comparing the "
                  'power of weapons, talk about clear times. Specifically how long it takes you to hunt down and '
                  'kill/capture a large monster. I\'ve seen a lot of talk saying that "in order to get comparable '
                  'clear times...don\'t use X weapon or use X weapon and armor sets". I saw earlier about someone in G '
                  'Rank complaining that they felt their weapon was under powered because they killed at G Rank '
                  'monster in around 8 mins compared to the 4 mins it takes them with a different weapon. Why are '
                  "shorter times, to the point of absurdity, the standard? Shouldn't we be more interested in having "
                  "good hunts and fun fights? Maybe it's because I'm High Rank and I'm still finding new and tougher "
                  "fights but I just don't see the point that of hunting something and killing it in under the amount "
                  "of time that it takes to warm up a hot pocket. Just doesn't sound fun to me"},
 {'dimension': 176,
  'direction': 'lowest',
  'value': -0.24833,
  'platform': 'reddit',
  'text_preview': "PVP - Is just me or this skills are broken? First of all, i'm a noob in terms on pvp, i'm just "
                  'starting with it, but i noticed that many of my deaths came always from the same 2 skills: overload '
                  'light attacks and cristalshards. They are hitting beetwen 10-18k, ranged attacks that can almost '
                  "one-shoot me (i'm around 20-21k hp) without much i can do dosn't seems like a proper mechanic. I "
                  'heard that bows don\'t do that much damage to reward meele combat that is more "risky", but with '
                  "this things one-shooting people at range, that argument don't have too much sense."},
 {'dimension': 176,
  'direction': 'lowest',
  'value': -0.237194,
  'platform': 'reddit',
  'text_preview': 'Can you handle it? None'},
 {'dimension': 224,
  'direction': 'highest',
  'value': 0.218133,
  'platform': 'reddit',
  'text_preview': 'HELP: Rega Elicit MK5 + KEF R3 Meta Hello, how are you? I’m looking to buy an integrated amp to be '
                  'paired with my KEF R3 Meta Speakers. I really like the Rega Elicit MK5, but since the speakers are '
                  '4 ohm, I’m weary of the warning that comes in the amp’s user manual: “intended for use with 4-16 '
                  'ohm speakers. Sustained used into speakers of less than 8 ohm could make the unit run warmer than '
                  'normal”. Any advice on how this pairing would work in the long run? I never listen for extended '
                  'periods of time and never blast full volume. Option B for me would be the Yamaha AS-1200 for the '
                  'KEF. Thanks in advance!'},
 {'dimension': 224,
  'direction': 'highest',
  'value': 0.208572,
  'platform': 'reddit',
  'text_preview': 'Laundry I’m almost out of clean clothes, where/can I do laundry in the dorm? I’m a freshman in '
                  'Carlyle.'},
 {'dimension': 224,
  'direction': 'highest',
  'value': 0.207315,
  'platform': 'twitter',
  'text_preview': '@ohnahji Getting my school outfit ready!'},
 {'dimension': 224,
  'direction': 'lowest',
  'value': -0.249706,
  'platform': 'hackernews',
  'text_preview': 'Looks like the objectivists downvoted you. Such is the nature of Hacker News!'},
 {'dimension': 224,
  'direction': 'lowest',
  'value': -0.237909,
  'platform': 'hackernews',
  'text_preview': '>...Heck, most people on Hackers News don’t even have an actual business. >The subject matter of '
                  'the post drove businesses with real money to our site. ...from Hacker News? Anyone else find this '
                  'rather contradictory?'},
 {'dimension': 224,
  'direction': 'lowest',
  'value': -0.230471,
  'platform': 'hackernews',
  'text_preview': 'It\'s a classic example of someone who hasn\'t had to deal with "hard" statistics thinking that '
                  'just drawing a trendline (and dismissing biases out-of-hand) is all that statistics is. The author '
                  'is right that Chrome has a massive share, but everyone knows that. Any analysis more specific than '
                  'that requires more than just overplotting exponentials on a graph.'},
 {'dimension': 223,
  'direction': 'highest',
  'value': 1.814682e-32,
  'platform': 'hackernews',
  'text_preview': 'wow cool!!!!!!!!!!!!!!!!!!!!!!! :0'},
 {'dimension': 223,
  'direction': 'highest',
  'value': 1.483488e-32,
  'platform': 'reddit',
  'text_preview': "Economic Political Compass So I've seen a bunch of posts on this sub quibbling about definitions so "
                  'I thought I might try my hand at a solution: Using an Economic Compass that deals with who owns the '
                  'economy or the means of production\\*. The format for this compass only works on PC. Sorry. '
                  '(Clarification and examples at the bottom) '
                  '........................**Authoritarian**........................ '
                  '...................................||................................... '
                  '...................................||................................... '
                  '**Planned\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_||\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_Market** '
                  '...................................||.................................. '
                  '...................................||.................................. '
                  '..........................**Democratic**.......................... \\*These definitions can be '
                  'separate from the government organization in market economies. **My opinion, and therefore fact for '
                  'what falls in each zone:** Authoritarian/Market: United States, Russia Pre-Ukraine invasion. This '
                  'is Traditional Capitalism, where companies sell and buy resources and are either controlled by a '
                  'boss, board, or non-employee stockholders. Authoritarian/Planned: Soviet Union, Pre-Colombian Inca, '
                  'Maoist China. Sometimes called democratic centralism, this economic structure seeing an autocrat or '
                  'an unelected party control the means of production according to multiyear plans. '
                  "Democratic/Planned: UK's NHS, Cuba. These examples are a bit of stretch, as there aren't really "
                  'many good examples a purely planned economy coexisting with a democratic government. Cuba is '
                  'probably the best example of this model as most of its industries are nationalized and there is '
                  'democratic involvement in the government (although there are limits on press & political freedom). '
                  "Democratic/Market: Worker Coops, Tito's Yugoslavia, Hunter Gatherer Societies. This organizational "
                  'structure sees workers in control of their means of production to buy and sell what they produce. '
                  'This can exist through democratic control of businesses or individual production. Sorry for the '
                  "horrible quality of the compass, this sub won't let me post pictures and it formats wrong with "
                  'spaces, if anyone can post it more competently please do it.'},
 {'dimension': 223,
  'direction': 'highest',
  'value': 1.092257e-32,
  'platform': 'twitter',
  'text_preview': 'wow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'},
 {'dimension': 223,
  'direction': 'lowest',
  'value': -3.796831e-32,
  'platform': 'twitter',
  'text_preview': '@RHGSIG @karina_filusch @computerspinnt @ICS_AG @netalexx @dani_stoffers @SuVuK @der_fisch001 '
                  '@SteveJRitter @linuxkumpel @ASYSinghuber @Cultcoders @guidodeckstein @ArneBrandneu @aixzellent Die '
                  'gibt es bestimmt in der nächsten Ausgabe des @informatikradar Magazins 😎'},
 {'dimension': 223,
  'direction': 'lowest',
  'value': -3.636443e-32,
  'platform': 'twitter',
  'text_preview': '@Mieke35735862 @MriaHajzer @lizBeth_Hineni @Gianlui91580067 @Tzipor336 @Ruthtgn69 @TruthWins4ever '
                  '@LehiRed @james_anderssen @esp_anet @cemoimonica @zeque66 @NYCREMilton @shlomomeirisra3 @YaelSed1 '
                  '@Helenhun20 @JankeHansenM @AlanZionist @reilluminati @LaurentTS56 @Esther28007 @Nelby37482058 '
                  '@adelioalves3 @olgatorres540 @Wp7xtPXs0EaLPTn @Smalou1 @EvaDuranRamos @HeatherPlonski1 @coinabs '
                  '@LenGrunstein @DSDloveyou Trust and stay positive Dear Mieke⚘️ Know that All things are possible '
                  "when you believe 🌴💚 what your think in your mind tells you what will come. G'D's Shalom🕊 be with "
                  'you 🙏☘️💛🕊 https://t.co/fmsG11kkQu'},
 {'dimension': 223,
  'direction': 'lowest',
  'value': -3.425661e-32,
  'platform': 'twitter',
  'text_preview': '@AndiFrankfurt @wfritz56 @PStter @mabagus @Don_Slivovic @alex13wetter @C7_black @Nic2012 '
                  '@TscheijPieh @DerVatta_ @EintrachtEmma @RealdealOk @whatsnexteu @23oliver01 @AsiFinanz Traust du '
                  'dich nicht'}]



def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def load_project_data() -> pd.DataFrame:
    columns = [
        "platform",
        "post_id",
        "author",
        "text",
        "created_at",
        "text_len",
        "year_month",
        "score",
        "community",
        "likes",
        "retweets",
        "replies",
        "quotes",
        "num_comments",
        "upvote_ratio",
    ]
    data_path = DATA_DIR / "social_platform_dataset_300k.parquet"
    if not data_path.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_parquet(data_path, columns=columns)
    df["platform"] = df["platform"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["month_start"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")
    df["text_preview"] = df["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 260)
    return add_text_stats(df)


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("")
    out = df.copy()
    out["num_questions"] = text.str.count(r"\?")
    out["num_exclamations"] = text.str.count(r"!")
    out["num_hashtags"] = text.str.count(r"#\w+")
    out["num_mentions"] = text.str.count(r"@\w+")
    out["num_urls"] = text.str.count(r"https?://|www\.")
    out["num_digits"] = text.str.count(r"\d")
    out["num_allcaps_words"] = text.str.count(r"\b[A-Z]{2,}\b")
    out["num_words"] = text.str.count(r"\S+").clip(lower=1)
    out["avg_word_len"] = (out["text_len"] / out["num_words"]).round(2)
    return out


def build_model_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    logreg = read_json(LOGREG_OUTPUT_DIR / "logreg_metrics.json")
    xgb = read_json(XGB_METRICS_PATH)
    nn = read_json(NN_OUTPUT_DIR / "nn_embedding_metrics.json")

    rows = [
        {
            "model": "Logistic Regression",
            "feature_set": "TF-IDF 1-2 grams + text statistics",
            "validation_accuracy": logreg.get("validation_accuracy", 0.8763028646354202),
            "test_accuracy": logreg.get("test_accuracy", 0.8788835059335971),
            "macro_f1": logreg.get("macro_f1", 0.8800108619954073),
            "interpretation": "Best for explanation: coefficients show which words and style markers pushed each platform prediction.",
        },
        {
            "model": "XGBoost",
            "feature_set": "MiniLM sentence embeddings",
            "validation_accuracy": xgb.get("validation_accuracy", 0.8897704291397204),
            "test_accuracy": xgb.get("test_accuracy", 0.891684074847771),
            "macro_f1": xgb.get("macro_f1", 0.8916222041254639),
            "interpretation": "Adds semantic embeddings and improves over the baseline, but its embedding dimensions are harder to interpret directly.",
        },
        {
            "model": "Neural Network",
            "feature_set": "MiniLM embeddings + text statistics",
            "validation_accuracy": nn.get("validation_accuracy", 0.9425072782629953),
            "test_accuracy": nn.get("test_accuracy", 0.9429530201342282),
            "macro_f1": 0.94,
            "interpretation": "Best final model: compact neural network with dropout and early stopping captured both semantic and style signals.",
        },
    ]

    per_class_rows = []
    report_sources = {
        "Logistic Regression": logreg.get("classification_report", {}),
        "XGBoost": xgb.get("classification_report", {}),
    }
    for model_name, report_source in report_sources.items():
        for platform in PLATFORMS:
            report = report_source.get(platform, {})
            per_class_rows.append(
                {
                    "model": model_name,
                    "platform": platform,
                    "precision": report.get("precision", 0.0),
                    "recall": report.get("recall", 0.0),
                    "f1_score": report.get("f1-score", 0.0),
                }
            )

    per_class_rows.extend(
        [
            {"model": "Neural Network", "platform": "hackernews", "precision": 0.92, "recall": 0.95, "f1_score": 0.94},
            {"model": "Neural Network", "platform": "reddit", "precision": 0.94, "recall": 0.94, "f1_score": 0.94},
            {"model": "Neural Network", "platform": "twitter", "precision": 0.96, "recall": 0.94, "f1_score": 0.95},
        ]
    )
    return pd.DataFrame(rows), pd.DataFrame(per_class_rows)


def feature_table() -> pd.DataFrame:
    df = read_csv_or_empty(LOGREG_TOP_FEATURES_PATH, ["platform", "rank", "feature", "weight"])
    if df.empty:
        rows = []
        for platform, features in LOGREG_TOP_FEATURES.items():
            for feature, coefficient in features.items():
                rows.append({"platform": platform, "feature": feature, "coefficient": coefficient})
        return pd.DataFrame(rows)
    return df.rename(columns={"weight": "coefficient"})


def xgb_dimension_table() -> pd.DataFrame:
    df = read_csv_or_empty(XGB_TOP_DIMENSIONS_PATH, ["embedding_dim", "importance"])
    if df.empty:
        return pd.DataFrame(XGB_TOP_DIMENSIONS)
    return df.rename(columns={"embedding_dim": "dimension"})


def xgb_example_table() -> pd.DataFrame:
    df = read_csv_or_empty(XGB_DIMENSION_EXAMPLES_PATH, ["dimension", "direction", "value", "platform", "text_example"])
    if df.empty:
        return pd.DataFrame(XGB_DIMENSION_EXAMPLES)
    return df.rename(columns={"text_example": "text_preview"})


DATA_DF = load_project_data()
METRICS_DF, PER_CLASS_DF = build_model_metrics()
FEATURE_DF = feature_table()
XGB_DIM_DF = xgb_dimension_table()
XGB_EXAMPLE_DF = xgb_example_table()


def platform_options() -> list[dict[str, str]]:
    return [{"label": PLATFORM_LABELS[p], "value": p} for p in PLATFORMS]


def format_platform(value: str) -> str:
    return PLATFORM_LABELS.get(value, value.title())


def filtered_data(platforms: list[str] | None, length_max: int | None = None) -> pd.DataFrame:
    selected = platforms or PLATFORMS
    df = DATA_DF[DATA_DF["platform"].isin(selected)]
    if length_max is not None:
        df = df[df["text_len"] <= length_max]
    return df


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[{"text": "No rows match the current filters.", "showarrow": False}],
        template="plotly_white",
    )
    return fig


def make_metric_cards() -> list[html.Div]:
    cards = []
    for _, row in METRICS_DF.iterrows():
        cards.append(
            html.Div(
                [
                    html.Div(row["model"], className="metric-label"),
                    html.Div(f"{row['test_accuracy']:.3f}", className="metric-value"),
                    html.Div("test accuracy", className="metric-caption"),
                    html.P(row["interpretation"], className="compact-text"),
                ],
                className="metric-card",
            )
        )
    return cards


def make_model_comparison(metric: str = "test_accuracy") -> go.Figure:
    label = metric.replace("_", " ").title()
    fig = px.bar(
        METRICS_DF,
        x="model",
        y=metric,
        color="model",
        color_discrete_sequence=["#0f766e", "#c2410c", "#2563eb"],
        text=metric,
        hover_data=["feature_set", "validation_accuracy", "test_accuracy", "macro_f1"],
        title=f"Model comparison by {label.lower()}",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0.80, 1.0], title="Score")
    fig.update_xaxes(title="")
    fig.update_layout(showlegend=False, template="plotly_white", margin=dict(l=30, r=20, t=60, b=35))
    return fig


def make_per_class_heatmap(metric: str = "f1_score") -> go.Figure:
    heatmap = PER_CLASS_DF.pivot(index="model", columns="platform", values=metric).loc[
        ["Logistic Regression", "XGBoost", "Neural Network"]
    ]
    fig = px.imshow(
        heatmap,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Tealrose",
        zmin=0.80,
        zmax=1.0,
        title=f"Per-class {metric.replace('_', ' ')}",
    )
    fig.update_xaxes(title="Actual platform", ticktext=[format_platform(c) for c in heatmap.columns], tickvals=list(heatmap.columns))
    fig.update_yaxes(title="")
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=60, b=35))
    return fig


def make_feature_bar(platform: str) -> go.Figure:
    df = FEATURE_DF[FEATURE_DF["platform"] == platform].sort_values("coefficient", ascending=True)
    fig = px.bar(
        df,
        x="coefficient",
        y="feature",
        orientation="h",
        color_discrete_sequence=[PLATFORM_COLORS[platform]],
        title=f"Most positive logistic-regression features for {format_platform(platform)}",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Coefficient weight", yaxis_title="", margin=dict(l=25, r=20, t=60, b=35))
    return fig


def make_xgb_dimension_bar() -> go.Figure:
    df = XGB_DIM_DF.sort_values("importance", ascending=True).copy()
    df["dimension_label"] = "Dimension " + df["dimension"].astype(str)
    fig = px.bar(
        df,
        x="importance",
        y="dimension_label",
        orientation="h",
        color_discrete_sequence=["#c2410c"],
        title="Most important XGBoost MiniLM embedding dimensions",
    )
    fig.update_layout(template="plotly_white", xaxis_title="XGBoost feature importance", yaxis_title="", margin=dict(l=25, r=20, t=60, b=35))
    return fig


def summarize_rows(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("No rows match the current filters.", className="callout")

    platform_counts = df["platform"].value_counts().reindex(PLATFORMS, fill_value=0)
    mean_len = df.groupby("platform")["text_len"].mean().reindex(PLATFORMS).round(1)
    fragments = [
        f"{format_platform(p)}: {int(platform_counts[p]):,} posts, average length {mean_len[p]:.1f}"
        for p in PLATFORMS
        if platform_counts[p] > 0
    ]
    return html.Div("Current filter: " + " | ".join(fragments), className="callout")


def make_post_records(df: pd.DataFrame, limit: int = 12) -> list[dict]:
    if df.empty:
        return []
    cols = ["platform", "created_at", "text_len", "author", "community", "text_preview"]
    out = df.loc[:, cols].copy()
    out["platform"] = out["platform"].map(format_platform)
    out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%d")
    out["community"] = out["community"].fillna("")
    return out.head(limit).to_dict("records")


def table_columns() -> list[dict[str, str]]:
    return [
        {"name": "Platform", "id": "platform"},
        {"name": "Date", "id": "created_at"},
        {"name": "Length", "id": "text_len"},
        {"name": "Author", "id": "author"},
        {"name": "Community", "id": "community"},
        {"name": "Text Preview", "id": "text_preview"},
    ]


def xgb_table_columns() -> list[dict[str, str]]:
    return [
        {"name": "Dimension", "id": "dimension"},
        {"name": "Direction", "id": "direction"},
        {"name": "Embedding Value", "id": "value"},
        {"name": "Platform", "id": "platform"},
        {"name": "Text Preview", "id": "text_preview"},
    ]


def feature_post_matches(platform: str, feature: str) -> pd.DataFrame:
    df = DATA_DF[DATA_DF["platform"] == platform].copy()
    if feature in df.columns:
        matches = df[df[feature] > 0].sort_values(feature, ascending=False)
    else:
        pattern = re.escape(feature)
        matches = df[df["text"].str.contains(pattern, case=False, na=False, regex=True)]
        matches = matches.assign(_match_len=matches["text"].str.len()).sort_values("_match_len", ascending=False)
    return matches


def make_xgb_example_records(dimension: int) -> list[dict]:
    df = XGB_EXAMPLE_DF[XGB_EXAMPLE_DF["dimension"] == dimension].copy()
    if df.empty:
        return []
    df["platform"] = df["platform"].map(format_platform)
    df["value"] = df["value"].map(lambda value: f"{value:.6g}")
    return df.to_dict("records")


def build_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "CIS 2450 Social Platform Dashboard"

    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div("CIS 2450 Final Project", className="eyebrow"),
                    html.H1("Social Media Platform Classification"),
                    html.P(
                        "A dashboard for exploring how language and text style differ across Hacker News, Reddit, and Twitter, "
                        "and how those signals affected the final classification models.",
                        className="lead",
                    ),
                ],
                className="hero",
            ),
            dcc.Tabs(
                id="main-tabs",
                value="overview",
                className="tabs",
                children=[
                    dcc.Tab(label="Overview", value="overview"),
                    dcc.Tab(label="EDA Explorer", value="eda"),
                    dcc.Tab(label="Modeling", value="modeling"),
                    dcc.Tab(label="Feature Lens", value="features"),
                ],
            ),
            html.Main(id="tab-content", className="content"),
        ],
        className="dashboard-shell",
    )

    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def render_tab(tab: str):
        if tab == "overview":
            return overview_tab()
        if tab == "eda":
            return eda_tab()
        if tab == "modeling":
            return modeling_tab()
        return features_tab()

    @app.callback(
        Output("eda-summary", "children"),
        Output("length-hist", "figure"),
        Output("style-bar", "figure"),
        Output("monthly-line", "figure"),
        Input("eda-platform-filter", "value"),
        Input("eda-length-slider", "value"),
        Input("style-metric", "value"),
    )
    def update_eda(platforms, length_max, style_metric):
        df = filtered_data(platforms, length_max)
        if df.empty:
            return summarize_rows(df), empty_figure("Text length distribution"), empty_figure("Style metric by platform"), empty_figure("Posts over time")

        hist = px.histogram(
            df,
            x="text_len",
            color="platform",
            nbins=60,
            barmode="overlay",
            opacity=0.68,
            color_discrete_map=PLATFORM_COLORS,
            title=f"Text length distribution up to {length_max:,} characters",
            labels={"text_len": "Text length", "count": "Posts"},
        )
        hist.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=60, b=35))

        grouped = df.groupby("platform", as_index=False)[style_metric].mean()
        style = px.bar(
            grouped,
            x="platform",
            y=style_metric,
            color="platform",
            color_discrete_map=PLATFORM_COLORS,
            text=style_metric,
            title=f"Average {STYLE_METRICS[style_metric].lower()} by platform",
        )
        style.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        style.update_xaxes(title="")
        style.update_yaxes(title=STYLE_METRICS[style_metric])
        style.update_layout(showlegend=False, template="plotly_white", margin=dict(l=30, r=20, t=60, b=35))

        monthly = df.dropna(subset=["month_start"]).groupby(["month_start", "platform"], as_index=False).size()
        line = px.line(
            monthly,
            x="month_start",
            y="size",
            color="platform",
            color_discrete_map=PLATFORM_COLORS,
            title="Monthly post coverage by platform",
            labels={"month_start": "Month", "size": "Posts"},
        )
        line.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=60, b=35))
        return summarize_rows(df), hist, style, line

    @app.callback(
        Output("model-comparison", "figure"),
        Output("class-heatmap", "figure"),
        Output("model-takeaway", "children"),
        Input("model-metric", "value"),
        Input("class-metric", "value"),
    )
    def update_modeling(model_metric, class_metric):
        best = METRICS_DF.sort_values(model_metric, ascending=False).iloc[0]
        takeaway = (
            f"{best['model']} leads on {model_metric.replace('_', ' ')} with a score of {best[model_metric]:.3f}. "
            "The accuracy and macro-F1 values are close because the project dataset was intentionally balanced across the three classes."
        )
        return (
            make_model_comparison(model_metric),
            make_per_class_heatmap(class_metric),
            html.Div(takeaway, className="callout"),
        )

    @app.callback(
        Output("feature-bar", "figure"),
        Output("feature-select", "options"),
        Output("feature-select", "value"),
        Output("feature-select-label", "children"),
        Output("feature-platform-wrap", "style"),
        Input("feature-model", "value"),
        Input("feature-platform", "value"),
    )
    def update_feature_options(feature_model, platform):
        if feature_model == "xgboost":
            options = [
                {
                    "label": f"Embedding dimension {row.dimension} ({row.importance:.4f})",
                    "value": str(row.dimension),
                }
                for row in XGB_DIM_DF.sort_values("importance", ascending=False).itertuples()
            ]
            value = options[0]["value"] if options else None
            return make_xgb_dimension_bar(), options, value, "Embedding dimension", {"display": "none"}

        feats = FEATURE_DF[FEATURE_DF["platform"] == platform].sort_values("coefficient", ascending=False)
        options = [{"label": f"{r.feature} ({r.coefficient:.2f})", "value": r.feature} for r in feats.itertuples()]
        option_values = {option["value"] for option in options}
        value = "looking" if platform == "reddit" and "looking" in option_values else (options[0]["value"] if options else None)
        return make_feature_bar(platform), options, value, "Coefficient feature", {}

    @app.callback(
        Output("feature-posts", "data"),
        Output("feature-posts", "columns"),
        Output("feature-note", "children"),
        Input("feature-model", "value"),
        Input("feature-platform", "value"),
        Input("feature-select", "value"),
    )
    def update_feature_posts(feature_model, platform, feature):
        if not feature:
            return [], table_columns(), ""
        if feature_model == "xgboost":
            dimension = int(feature)
            note = (
                f"Showing posts with the highest and lowest values on MiniLM embedding dimension {dimension}. "
                f"{XGB_DIMENSION_SUMMARIES.get(dimension, '')}"
            )
            return make_xgb_example_records(dimension), xgb_table_columns(), note

        matches = feature_post_matches(platform, feature)
        note = (
            f"Showing real {format_platform(platform)} posts that contain or emphasize '{feature}'. "
        )
        return make_post_records(matches, limit=10), table_columns(), note

    return app


def overview_tab() -> html.Div:
    return html.Div(
        [
            html.Section(
                [
                    html.H2("Overview"),
                    html.P(
                        "Our project asks whether a post's language and style reveal which platform it came from. "
                        "The notebooks build a balanced dataset and intentionally exclude IDs, authors, dates, and platform-specific metadata from modeling, focusing only on text and text-derived features.",
                    ),
                    html.Div(
                        [
                            html.Div([html.H3("Objective"), html.P("Classify posts as Hacker News, Reddit, or Twitter using text-derived signals.")]),
                            html.Div([html.H3("EDA insight"), html.P("Length, questions, hashtags, mentions, URLs, and word patterns differ sharply across platforms.")]),
                            html.Div([html.H3("Modeling insight"), html.P("Embeddings improve accuracy, and the neural network performs best while the logistic baseline remains the most explainable.")]),
                        ],
                        className="three-col",
                    ),
                ],
                className="section-block",
            ),
            html.Section(
                [
                    html.H2("Data Choices"),
                    html.P(
                        "The source review and EDA shaped the dataset pipeline before any model was trained.",
                        className="section-intro",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Text-only modeling"),
                                    html.P(
                                        "The platforms expose different metadata, so likes, communities, parent IDs, dates, authors, and platform-specific fields are excluded from the model inputs."
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H3("Balanced dataset"),
                                    html.P(
                                        "The final dataset keeps 100,000 posts per platform, which makes accuracy more meaningful and prevents one class from dominating the result."
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H3("Leakage control"),
                                    html.P(
                                        "Time coverage and source formatting differ by platform, so dates are used for auditing but not modeling, and HTML-like markup is stripped before splitting."
                                    ),
                                ]
                            ),
                        ],
                        className="three-col",
                    ),
                ],
                className="section-block",
            ),
            html.Section([html.H2("Headline Results"), html.Div(make_metric_cards(), className="metrics-grid")], className="section-block"),
        ]
    )


def eda_tab() -> html.Div:
    return html.Div(
        [
            html.Section(
                [
                    html.H2("EDA Explorer"),
                    html.P(
                        "Use the controls to see how the same text-derived patterns from the notebooks change by platform and length range.",
                        className="section-intro",
                    ),
                    html.Div(
                        [
                            html.Label("Platforms"),
                            dcc.Checklist(
                                id="eda-platform-filter",
                                options=platform_options(),
                                value=PLATFORMS,
                                inline=True,
                                className="checklist",
                            ),
                            html.Label("Maximum text length shown"),
                            dcc.Slider(
                                id="eda-length-slider",
                                min=100,
                                max=2500,
                                step=100,
                                value=1000,
                                marks={100: "100", 500: "500", 1000: "1k", 2000: "2k", 2500: "2.5k"},
                            ),
                            html.Label("Style metric"),
                            dcc.Dropdown(
                                id="style-metric",
                                options=[{"label": label, "value": key} for key, label in STYLE_METRICS.items()],
                                value="num_hashtags",
                                clearable=False,
                            ),
                        ],
                        className="control-panel",
                    ),
                    html.Div(id="eda-summary"),
                    html.Div(
                        [
                            html.Div(dcc.Graph(id="length-hist", config={"displayModeBar": False}), className="chart-panel"),
                            html.Div(dcc.Graph(id="style-bar", config={"displayModeBar": False}), className="chart-panel"),
                        ],
                        className="two-col",
                    ),
                    html.Div(dcc.Graph(id="monthly-line", config={"displayModeBar": False}), className="chart-panel"),
                    html.Div(
                        "Twitter is much shorter and much richer in hashtags, mentions, and URLs. "
                        "Hacker News and Reddit contain longer conversational posts, but Reddit tends to show more questions and community-style language. "
                        "Time features were inspected for coverage but excluded from models to avoid learning collection artifacts.",
                        className="callout",
                    ),
                    html.Div(
                        "The engineered text statistics are informative, but the classes still overlap. That is why the project combines simple style features with richer TF-IDF and MiniLM embedding representations instead of relying on one chart or one feature.",
                        className="callout",
                    ),
                ],
                className="section-block",
            )
        ]
    )


def modeling_tab() -> html.Div:
    logreg_val = METRICS_DF.loc[METRICS_DF["model"] == "Logistic Regression", "validation_accuracy"].iloc[0]
    nn_best_epoch = read_json(NN_OUTPUT_DIR / "nn_embedding_metrics.json").get("best_epoch", 11)
    return html.Div(
        [
            html.Section(
                [
                    html.H2("Modeling Results"),
                    html.P(
                        "The dashboard compares a transparent TF-IDF logistic baseline, an embedding-based XGBoost model, and the final MiniLM plus neural network.",
                        className="section-intro",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Logistic Regression"),
                                    html.P(
                                        "A strong baseline for sparse TF-IDF text features. Its main value is interpretability because coefficients map back to readable words and text statistics."
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H3("XGBoost"),
                                    html.P(
                                        "Uses MiniLM sentence embeddings so the model can learn nonlinear boundaries in semantic space instead of only counting explicit words."
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H3("Neural Network"),
                                    html.P(
                                        "Combines MiniLM embeddings with engineered text statistics in a small neural network, using dropout, batch normalization, and early stopping to reduce overfitting."
                                    ),
                                ]
                            ),
                        ],
                        className="three-col model-rationale-grid",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Model comparison metric"),
                                    dcc.Dropdown(
                                        id="model-metric",
                                        options=[
                                            {"label": "Validation accuracy", "value": "validation_accuracy"},
                                            {"label": "Test accuracy", "value": "test_accuracy"},
                                            {"label": "Macro F1", "value": "macro_f1"},
                                        ],
                                        value="test_accuracy",
                                        clearable=False,
                                    ),
                                ],
                                className="control-inline",
                            ),
                            html.Div(
                                [
                                    html.Label("Per-class metric"),
                                    dcc.Dropdown(
                                        id="class-metric",
                                        options=[
                                            {"label": "Precision", "value": "precision"},
                                            {"label": "Recall", "value": "recall"},
                                            {"label": "F1 score", "value": "f1_score"},
                                        ],
                                        value="f1_score",
                                        clearable=False,
                                    ),
                                ],
                                className="control-inline",
                            ),
                        ],
                        className="control-row",
                    ),
                    html.Div(id="model-takeaway"),
                    html.Div(
                        [
                            html.Div(dcc.Graph(id="model-comparison", config={"displayModeBar": False}), className="chart-panel"),
                            html.Div(dcc.Graph(id="class-heatmap", config={"displayModeBar": False}), className="chart-panel"),
                        ],
                        className="two-col",
                    ),
                    html.Div(
                        f"For logistic regression, the engineered text statistics lifted validation accuracy from 0.827 with TF-IDF alone to {logreg_val:.3f} with TF-IDF plus text stats. "
                        f"For our other models, MiniLM embeddings added semantic signal, and the neural network ended up reaching the best test accuracy after early stopping at epoch {nn_best_epoch}.",
                        className="callout",
                    ),
                    html.Div(
                        "Each model uses a validation-based tuning pass before final test evaluation. Macro F1 is tracked alongside accuracy so performance is judged across all three platform classes, not just by one headline number.",
                        className="callout",
                    ),
                ],
                className="section-block",
            )
        ]
    )

def features_tab() -> html.Div:
    return html.Div(
        [
            html.Section(
                [
                    html.H2("Feature Lens"),
                    html.P(
                        "Switch between direct Logistic Regression coefficients and XGBoost's semantic embedding dimensions. Both views help connect the models' signals back to real posts from our dataset.",
                        className="section-intro",
                    ),
                    html.Div(
                        "For Logistic Regression, a feature is a readable word or text statistic with a positive coefficient for a platform. For XGBoost, each MiniLM dimension is a latent semantic coordinate, so the dashboard interprets it through high-value and low-value example posts rather than pretending the dimension has one exact label.",
                        className="callout",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Explanation model"),
                                    dcc.Dropdown(
                                        id="feature-model",
                                        options=[
                                            {"label": "Logistic Regression coefficients", "value": "logreg"},
                                            {"label": "XGBoost embedding dimensions", "value": "xgboost"},
                                        ],
                                        value="xgboost",
                                        clearable=False,
                                    ),
                                ],
                                className="control-inline",
                            ),
                            html.Div(
                                [
                                    html.Label("Platform class"),
                                    dcc.Dropdown(id="feature-platform", options=platform_options(), value="reddit", clearable=False),
                                ],
                                className="control-inline",
                                id="feature-platform-wrap",
                            ),
                            html.Div(
                                [
                                    html.Label("Coefficient feature", id="feature-select-label"),
                                    dcc.Dropdown(id="feature-select", clearable=False),
                                ],
                                className="control-inline",
                            ),
                        ],
                        className="control-row",
                    ),
                    html.Div(
                        [
                            html.Div(dcc.Graph(id="feature-bar", config={"displayModeBar": False}), className="chart-panel"),
                            html.Div(
                                [
                                    html.Div(id="feature-note", className="callout"),
                                    dash_table.DataTable(
                                        id="feature-posts",
                                        columns=table_columns(),
                                        data=[],
                                        page_size=5,
                                        style_table={"overflowX": "auto"},
                                        style_cell={"textAlign": "left", "padding": "10px", "whiteSpace": "normal", "height": "auto"},
                                        style_header={"fontWeight": "700"},
                                    ),
                                ],
                                className="chart-panel",
                            ),
                        ],
                        className="two-col",
                    ),
                ],
                className="section-block",
            )
        ]
    )

if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = build_app()
    app.run(debug=False, port=8040)
