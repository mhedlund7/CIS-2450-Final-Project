# Final Project: Detailed Rubric

**CIS 2450: Big Data Analytics**

## Types of scoring Label

- ➕ **Positive scoring (select multiple):** the section starts with 0, points stack if multiple aspects of the rubric are met.
- ✖️ **Negative scoring (select 1):** the section starts with the max points available for each section, and points are deducted if elements are missing from each section.
- ✖️✖️✖️ **Negative scoring (select multiple):** the section starts with the max points available, and points stack if multiple aspects of the rubric are missing.

---

## 1. Project Proposal (+5 points) — ✖️

| Criteria | Points |
|---|---:|
| **All Good:** Project Proposal submitted on time | 0 |
| **Late Submission:** Project Proposal submitted late. | -5 |
| **Failure to add all team members:** Every time you **re-submit**, you need to **re-add** your teammate. Failure to do so will result in a 5 points penalty for the entire team. | -5 |

---

## 2. Intermediate Check-In (+5 points) — ✖️✖️✖️

### Check-In Questions

- **Get to know your data:** Have you recognized issues / assumptions with your data and how do you plan to handle them?
- **EDA:** Do you have at least 3 meaningful EDA visuals?
- **Modeling:** How does the baseline model perform? Comment on the performance. What are the 2-3 models that you will further implement? Why have you chosen these models over other options?
- **Project Management:** What’s the plan of action? By when do you plan to complete various stages of this project?

| Criteria | Points |
|---|---:|
| **All Good:** intermediate check-in completed on time with sufficient and clear responses to the above questions. | 0 |
| **Insufficient work completed:** intermediate check-in completed on time, but insufficient work completed or responses to the questions above by the Intermediate Check-In, such as incomplete EDA or missing baseline model. One point deduction for insufficient answer to each of the questions above. | -1, stackable |
| **No work completed:** Notebook not created, no data loaded, insufficient responses for all questions above. | -5 |
| **Late:** intermediate check-in not completed on time | -5 |
| **Individual Penalty:** Team Members did not show up or show up very late without valid reason. | -5 |

---

## 3. Difficulty (+13 points) — ➕

In this section, points will be awarded on the **depth** of application of these topics. For each of the below, award partial credit if the usage of the feature is not fully justified OR if the feature is not well implemented. If there are more than 3 concepts used, choose the best 3 concepts. We prioritize depth, applying the concepts well, rather than breadth, touching on many concepts.

Possible concepts include:

- **Feature importance:** using feature importance method in tree models OR using coefficient in regression model to identify most important features
- **Feature selection:** explore using forward/backward feature selection, PCA, regularization to narrow down feature space
- **Other visualization packages:** using plotly, folium
- **Feature engineering:** use of other categorical encoding methods or engineering interaction terms to help improve model performance
- **Ensemble models:** implementing ensemble models
- **Imbalance data:** using upsampling/downsampling/SMOTE-derivative methods when dealing with an imbalanced data set
- **Hyperparameter tuning:** using smarter hyperparameter tuning methods such as Randomized Search or Bayesian Optimization
- **Entity Linking:** using entity linking to enrich data sets and/or combine data sets that are not obviously compatible

| Criteria | Points |
|---|---:|
| **Concept 1:** Award points based on the following metric:<br>- Concept is implemented correctly (+1)<br>- Use of concept is fully justified (+1)<br>- Results are used and reflected in the conclusion (+1)<br>- Correctly identify where concept is used (+1) | +4 |
| **Concept 2:** Award points based on the following metric:<br>- Concept is implemented correctly (+1)<br>- Use of concept is fully justified (+1)<br>- Results are used and reflected in the conclusion (+1)<br>- Correctly identify where concept is used (+1) | +4 |
| **Concept 3:** Award points based on the following metric:<br>- Concept is implemented correctly (+1)<br>- Use of concept is fully justified (+1)<br>- Results are used and reflected in the conclusion (+1)<br>- Correctly identify where concept is used (+1) | +4 |
| **Bonus:** Team went above and beyond on one of the concept implementations. | +1 |
| **Standard Project:** this project follows the standard outline/difficulty of our homeworks. | +0 |

---

## 4a. EDA (+10 points) — ✖️

EDA steps may include, but are not limited to, the following:

- ☑️ Provide necessary context on attributes in relation to overall data.
- ☑️ Provide background on the issue statement and high-level definitions of critical variables.
- ☑️ Understand key variables’ data types, summary statistics, and distribution, using histogram/scatter plot/pair-plots.
- ☑️ Identify outliers and understand how it may be handled.

Each step should include a brief summary of findings in a markdown cell, and inform the roadmap for the project as a whole.

| Criteria | Points |
|---|---:|
| **Excellent:** Incorporated all necessary EDA aspects proficiently with each step providing a further understanding of the underlying data. EDA is informative and succinct, quality over quantity. Furthermore, EDA is used to tell a story based on business sense and/or domain knowledge. | 0 |
| **Good:** Incorporated all necessary EDA aspects proficiently with each step providing a further understanding of the underlying data. | -3 |
| **Average:** Incorporated most necessary EDA aspects proficiently. Some EDA steps may not fit in the larger scope of the project. Most steps come with justification. | -5 |
| **Fair:** Incorporated a few EDA aspects proficiently. Many EDA steps are performed without justifications and detract from the overall objective. | -8 |
| **Poor:** simply imported data without any EDA | -10 |

---

## 4b. Data Pre-processing & Feature Engineering (+10 points) — ✖️✖️✖️

Feature engineering / data pre-processing steps may include, but are not limited to, the following:

- ☑️ efficient scraping or collection of data from data sources
- ☑️ effective combination of data from distinct data sources
- ☑️ handle null values appropriately based on variable distribution
- ☑️ handle outliers appropriately
- ☑️ engineer appropriate new features
- ☑️ one-hot encode categorical variable, if applicable
- ☑️ check variable correlation/remove highly correlated variables
- ☑️ address imbalanced data
- ☑️ scale data

| Criteria | Points |
|---|---:|
| **Excellent:** incorporated all necessary aspects proficiently with sufficient justification for choices. Feature engineering and data pre-processing steps are informed decisions following insights drawn from EDA. | 0 |
| **Good:** incorporated most necessary aspects proficiently with sufficient justification for choices. Feature engineering and data pre-processing steps are informed decisions following insights drawn from EDA. | -3 |
| **Average:** incorporated a few necessary aspects with lack of justification for choices | -5 |
| **Poor:** incorporated a few necessary aspects with lack of justification for choices and conceptual errors were made | -8 |
| **None:** The group simply loaded the data and proceeded to modeling. | -10 |

---

## 5a. Model Implementation (+12 points) — ✖️✖️

### Example Justification Error

These include, but are not limited to:

- ➡️ failing to address model limitations and potential biases when choosing model
- ➡️ introduce new model but it has the same potential shortcomings of the original model
- ➡️ use NN models when the goal is having an interpretable/explainable results

### Example Conceptual Error

These include, but are not limited to:

- ➡️ increased regularization for already underfit models
- ➡️ increased complexity for already overfit models, etc.

| Criteria | Points |
|---|---:|
| **Pipeline**, if any is applicable. -1 points for each incorrect implementation, up to 5 points:<br>- failure to train-test-split data<br>- applied `.fit` on test data for either standardization, PCA, or modeling<br>- applied standardization/PCA before train-test split<br>- applied up/downsampling before train-test split<br>- Impute missing values based on statistics, such as mean, median, mode, calculated from the entire dataset<br>- Incorporate test set for hyper-parameter tuning, instead of using validation set | -3 |
| **Model 1: Baseline**<br>- Model is not implemented (-1)<br>- Model is implemented with conceptual error (-1)<br>- Model is implemented without appropriate justification (-1) | -3 |
| **Model 2**<br>- Model is not implemented (-1)<br>- Model is implemented with conceptual error (-1)<br>- Model is implemented without appropriate justification (-1) | -3 |
| **Model 3**<br>- Model is not implemented (-1)<br>- Model is implemented with conceptual error (-1)<br>- Model is implemented without appropriate justification (-1) | -3 |

---

## 5b. Model Assessment and Hyperparameter Tuning (+8 points) — ✖️✖️

| Criteria | Points |
|---|---:|
| **Flawless:** uses multiple assessment metrics correctly in understanding model performance, performs methodical approach to hyperparameter and parameter tuning, offers sufficient rationale for new design implementations and changes | 0 |
| **Hyper-parameter Tuning Missing:** missing hyper-parameter tuning in any one of the models. This penalty stacks with others. | -2 |
| **Missing/Inappropriate Evaluation Metrics:** does not use any form of evaluation metric, or inappropriate choice of evaluation metric. This penalty stacks with others.<br><br>Common mistakes include, but are not limited to:<br>- ➡️ using “accuracy” for a classification of severely imbalanced dataset with no attempts to address the imbalance<br>- ➡️ using a regression evaluation metric for a classification problem, or vice versa | -2 |
| **Conceptual Error:** lacks methodical and iterative approach in implementing changes in training any one of the models. This penalty stacks with others.<br><br>Common mistakes include, but are not limited to:<br>- ➡️ deciding to increase model complexity despite training and validation curves clearly indicating the presence of overfitting<br>- ➡️ increased regularization for already underfit models<br>- ➡️ stating model is performing fine by solely focusing on one accuracy metric and neglecting others | -2 |
| **Evaluation/Justification Missing:** fails to communicate the rationale of the model design decisions made in relation to the project objectives. This penalty stacks with others.<br><br>Common mistakes include, but are not limited to:<br>- ➡️ using grid search cross-validation to tune hyperparameters without entirely understanding their roles in the model<br>- ➡️ only reporting the performance results of best models but not comparing them with previous models | -2 |
| **Not at all:** no model assessment or hyperparameter tuning attempted | -8 |

---

## 7. Code Quality/Readability (+10 points) — ✖️

| Criteria | Points |
|---|---:|
| **Excellent:** The codebase is well-organized modularly and functionally. Code is totally readable, broken into logical sections, easy to follow, and appropriate in-line code comments are provided when helpful. | 0 |
| **Good:** Codebase is organized not quite modularly, is slightly difficult to follow, e.g. not broken down into logical blocks, or other inconveniences. For the most part, a third person can look at your code and tell the story of what’s going on with only minor difficulties. | -3 |
| **Needs Improvement:** Codebase is completely disorganized, all code contained within one file, for example. Code is very difficult to follow, e.g. code is difficult to understand because comments are absent when they are necessary for a third-person to understand what's going on, or other major obstacles. | -6 |
| **Poor:** no code or large portions of code are missing i.e. an incomplete project | -10 |

---

## 8. Application of Course Topics (+10 points) — ✖️

Course topics include:

- ☑️ Polars
- ☑️ SQL
- ☑️ Text representations, embeddings, and LLMs
- ☑️ Record Linking
- ☑️ Hypothesis testing
- ☑️ Unsupervised Learning models
- ☑️ Supervised Learning models
- ☑️ Spark
- ☑️ Graphs
- ☑️ Time Series
- ☑️ Streaming
- ☑️ Deep learning (CNN, RNN, Transformers)
- ☑️ Agents
- ☑️ Joins
- ☑️ Relational database
- ☑️ Different methods of hyperparameter tuning

If any of these is used without logical conditioning on how the concept is relevant to the overall goal of the project, use of concept will not be counted.

For example:

1. In a classification project, a student runs a K-Means algorithm as a stand-alone model without incorporating cluster results as a feature.
2. A dataset only has 1 table, and the team provided an ER diagram.

| Criteria | Points |
|---|---:|
| **Excellent:** incorporated at least 6 of the above. | 0 |
| **Good:** incorporated 3-4 of the above. | -3 |
| **Mediocre:** incorporated only 2 of the above. | -5 |
| **Poor:** incorporated only 1 of the above. | -8 |
| **None:** no course topics included in the project | -10 |

---

## 9. Quality of Dashboard Demo (+20 points) — ✖️

| Criteria | Points |
|---|---:|
| **Excellent:** A highly polished, visually appealing, and bug-free dashboard. Presents information clearly and professionally, using effective and potentially complex visualizations. Thoroughly covers all aspects of the project, including both exploratory data analysis (EDA) and modeling work, with clear interpretation and insights throughout. | 0 |
| **Good:** A solid and mostly well-constructed dashboard that covers the majority of the project work. Uses primarily standard visualization techniques and is generally clear, though minor bugs or inconsistencies may be present. Some areas may lack depth or refinement. | -5 |
| **Average:** A dashboard with meaningful content but lacking clarity or cohesion. Visualizations may be poorly explained, insufficiently justified, or difficult to interpret. Organization and presentation make it harder for the audience to follow the narrative. | -10 |
| **Unacceptable:** A severely incomplete or non-functional deliverable that is neither clean nor engaging. Lacks clarity, structure, and basic effort, and fails to meet minimum project requirements. | -20 |

---

## 11. Quality of Presentation (+10 points) — ✖️✖️

Along with your annotated notebook, you will also be required to give a presentation to the TAs grading your project. Presentation can either be done live over Zoom or recorded and submitted on Gradescope and should cover the following topics:

- ☑️ Objective, value proposition, and dataset used
- ☑️ Major learnings from EDA, with top 3-5 well-formatted charts that deepened your understanding of the dataset and informed your model
- ☑️ Modeling results, models used, performance, etc.
- ☑️ Implications and insights
- ☑️ Challenges/limitations/potential future work

Additionally, we require that all group members speak and have relatively equal roles in the presentation, each group member’s camera be on during the presentation, and no text-to-speech or virtual avatars should be used in place of the above.

| Criteria | Points |
|---|---:|
| **Excellent:** Presentation addressed all requested sections, and delivered a clean and engaging presentation. All visuals used are informative. Every project member presented. | 0 |
| **Missing section:** presentation missing any of the requested sections OR the content of any section is poorly discussed. This penalty stacks with others. | -2 |
| **Missing/inappropriate visuals:** did not use visuals or visuals are irrelevant to topics discussed. This penalty stacks with others. | -2 |
| **Under/Over-time:** presentation is less than 8 or over 10 mins OR evident post-editing to speed up the video to fit the time limit. This penalty stacks with others. | -2 |
| **Unpolished presentation:** contains various typos, codes, or inconsistent formats that make the presentation look unprofessional. This penalty stacks with others. | -2 |
| **Not all members presented:** This penalty stacks with others. | -2 |
| **No Submission:** missing recorded/live presentation | -10 |

---

## 12. Other Penalties — ✖️✖️

| Criteria | Points |
|---|---:|
| **All Good** | 0 |
| **Late Final Deliverable** | Vary |
| **Project Contribution Issue:** Teammates will have the opportunity to provide feedback. Individual grades will be a direct reflection of contribution. | Vary |
| **Others** | Vary |

---

## Note

The provided screenshots appear to skip section 6 and section 10, or those sections were not included in the images. This markdown file only includes the content visible in the screenshots.
