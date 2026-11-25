# AAI 501 Group 01 Project

## RAGTruth dataset download

**This project uses the RAGTruth hallucination benchmark from ParticleMedia.** [RAGTruth README](https://github.com/ParticleMedia/RAGTruth/blob/main/README.md)

The dataset files are included in this repository under `data/raw/` so team members can pull the project and work with the same data immediately.

1. If you are starting from scratch, clone the project repository to your computer:

```bash
git clone https://github.com/alexafshar/aai501-grp1-project.git
```

Then move into the project folder:

```bash
cd aai501-grp1-project
```

2. To update your local repository with the latest project files from GitHub, pull the repository:

```bash
git pull origin main
```

3. Load the files with pandas and verify the expected row counts. You can do this either in a Python session or by opening the notebook at `notebooks/01_load_ragtruth_data.ipynb`.

```bash
python
```

```python
import pandas as pd

response_df = pd.read_json("data/raw/response.jsonl", lines=True)
source_df = pd.read_json("data/raw/source_info.jsonl", lines=True)

print(len(response_df))
print(len(source_df))
```

Expected output:

```text
17790
2965
```

Side note: if a teammate wants to refresh the dataset from the upstream source instead of using the repository copy, they can follow the original RAGTruth instructions and copy the files into `data/raw/` manually.

## Token counts (task T05)

`notebooks/T05_count_tokens.ipynb` counts tokens for every response in `data/responses_clean.csv` (the T04 output):

- `input_tokens`: tokens in the `prompt` sent to the model. The prompt already includes the source passages, so the source is not counted twice.
- `output_tokens`: tokens in the model's `response`.

It writes the two columns back into `data/responses_clean.csv` (local only, since that file is over GitHub's 100 MB limit) and saves a small shareable file, `data/processed/T05_token_counts.csv`, with `id`, `model`, `input_tokens`, and `output_tokens`. Join it to `data/processed/features.csv` on `id`.

To rerun it, open the notebook and run all cells (first `pip install tiktoken pandas` if needed).

Counts use OpenAI's `cl100k_base` encoding, so they are exact for GPT-3.5 and GPT-4 and approximate for Llama-2 and Mistral, which use their own tokenizers.

## Price assumptions (task T06)

`T06_config.yaml` holds every price and cost assumption the cost analysis uses (T14, T17, T19): input and output prices for the six RAGTruth models, the LLM-judge prices for the checking-cost comparison, and the guardrail and sensitivity settings. Each price lists its source and the date it was checked (2026-11-14). Read it in a notebook with:

```python
import yaml
config = yaml.safe_load(open("T06_config.yaml"))
```

Prices are cited assumptions, not data. The four open-weight models (Llama-2 7B, 13B, 70B; Mistral-7B) use hosted-API prices as a stand-in for the cost of running them locally. The cost of a missed hallucination ($1.00) has no published source and is varied in T19.

References (APA 7):

- LLM Reference. (n.d.). *Together AI API: Available models and pricing*. Retrieved November 14, 2026, from https://www.llmreference.com/provider/together-ai-api
- OpenAI. (n.d.-a). *API pricing*. Retrieved November 14, 2026, from https://developers.openai.com/api/docs/pricing
- OpenAI. (n.d.-b). *GPT-3.5 Turbo*. Retrieved November 14, 2026, from https://developers.openai.com/api/docs/models/gpt-3.5-turbo
- OpenAI. (n.d.-c). *GPT-4*. Retrieved November 14, 2026, from https://developers.openai.com/api/docs/models/gpt-4

## Train/test leakage check (task T08)

`notebooks/T08_train_test_leakage_check.ipynb` checks that the train and test sets in `data/processed/features.csv` are truly separate. Each source passage has about six responses (one per model), so if the same `source_id` landed in both sets, the model could learn source-specific patterns and the test scores would be inflated.

The predefined `split` column passes the check: 14,942 train rows and 2,675 test rows, with no `source_id` in both. Every later model should use this column instead of a random row-level split.

Classes are moderately imbalanced (43% hallucinated overall), and the hallucination rate differs between the sets (45% train vs. 35% test), which lowers precision on the test set.

## Baseline models (task T09)

`notebooks/T09_baseline_models.ipynb` trains three baseline classifiers on the 14 engineered features, using the T08 split: logistic regression, a decision tree (max depth 5), and a random forest (200 trees). All use `class_weight="balanced"` with median imputation and standard scaling. It saves the comparison to `data/processed/baseline_model_results.csv`.

Test-set results (hallucinated = positive class):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|
| Logistic regression | 0.779 | 0.657 | 0.779 | **0.713** | **0.872** | **0.802** |
| Random forest | 0.784 | 0.672 | 0.758 | 0.713 | 0.862 | 0.792 |
| Decision tree | 0.773 | 0.646 | **0.792** | 0.711 | 0.860 | 0.761 |

Logistic regression is the benchmark for T12: it has the best F1, ROC-AUC, and average precision. The three models are within 0.002 F1 of each other, so the margin is small; if recall is the main criterion, the decision tree is slightly ahead. The best model was chosen on the test set, which the report should state.

To rerun it, open the notebook and run all cells (requires `pandas`, `scikit-learn`, `matplotlib`, `seaborn`).

## Random forest and XGBoost (task T12)

`notebooks/T12_random_forest_xgboost.ipynb` tunes a random forest and XGBoost on the 14 engineered features, using the T08 split and the Assignment 4.2 workflow. Each model gets 8 settings searched with `GridSearchCV(cv=StratifiedKFold(5), scoring='f1')` on the training split only. Imbalance is handled with `class_weight="balanced"` (random forest) and `scale_pos_weight` = negatives / positives = 1.231 (XGBoost). Each tuned model is scored once on the test split at the default 0.5 threshold, next to the T09 logistic regression refit on the same split.

Test-set results (hallucinated = positive class):

| Model | Best settings | CV F1 | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|---|
| Random forest | 200 trees, max depth 10, min leaf 1 | 0.759 | **0.708** | 0.755 | **0.731** | **0.875** | 0.812 |
| XGBoost | 200 rounds, max depth 6, learning rate 0.05 | 0.758 | 0.688 | 0.762 | 0.723 | 0.874 | **0.815** |
| Logistic regression (T09) | — | — | 0.657 | **0.779** | 0.713 | 0.872 | 0.802 |

Both ensembles beat the benchmark on F1 (random forest +0.017, XGBoost +0.010) mainly through higher precision; logistic regression keeps the highest recall. The ROC-AUC gains are small (+0.002 to +0.003), so the features, not the model, limit performance. Fully grown random-forest trees overfit (CV train F1 1.00 vs. validation 0.75), so the depth limit matters.

Outputs (in `results/`):

- `predictions.csv`: one row per test answer (2,675) with `id`, `source_id`, `llm`, `task_type`, the true label `hallucinated`, and the predicted probabilities `prob_logreg`, `prob_rf`, and `prob_xgb`. Join to `data/processed/T05_token_counts.csv` on `id` for T17.
- `T12_model_comparison.csv`: the test-set metrics above.
- `T12_cv_results.csv`: every grid setting with its CV train and validation F1.

CV folds are split by row, so answers to the same source can fall in different folds and the CV scores are slightly optimistic; the test score is unaffected.

To rerun it, open the notebook and run all cells (requires `xgboost` in addition to the T09 packages).

## Feature importance (task T15)

`notebooks/T15_feature_importance.ipynb` explains the two tuned T12 models (rebuilt from `results/T12_cv_results.csv`; the notebook checks that they reproduce `results/predictions.csv`). It compares three methods: built-in importance (impurity for the random forest, gain for XGBoost), permutation importance on the test split (drop in ROC-AUC and F1, 20 repeats), and SHAP values, which also give the direction of each effect. The six LLM one-hot columns and the three task-type columns are also shuffled as groups, so "which LLM" and "which task" can be compared with the numeric features as single factors.

Grouped permutation importance (drop in test ROC-AUC when the feature is shuffled):

| Feature (group) | XGBoost | Random forest | Direction (SHAP) |
|---|---|---|---|
| LLM that wrote the answer (6 columns) | **0.140** | **0.141** | GPT-4 and GPT-3.5 lower the risk; Mistral-7B and Llama-2-7B raise it |
| `word_overlap` | 0.093 | 0.054 | Higher overlap with the source → less hallucination |
| Task type (3 columns) | 0.025 | 0.052 | Data-to-text raises the risk; QA lowers it |
| `answer_length` | 0.014 | 0.008 | Longer answers → more hallucination |
| `missing_numbers` | 0.007 | 0.007 | More numbers not found in the source → more hallucination |
| `missing_names` | 0.002 | 0.005 | More names not found in the source → more hallucination |
| `length_ratio` | 0.001 | 0.001 | No clear effect |

Which LLM wrote the answer is the strongest factor, followed by `word_overlap`, the strongest content feature. All methods and both models agree on the top five single columns (`word_overlap`, the GPT-3.5, GPT-4, and Mistral-7B indicators, and Data-to-text). `length_ratio` adds almost nothing on unseen data and could be dropped. Built-in importance overstates the continuous features (for example, `missing_names` and `length_ratio` in the random forest) compared with test-set permutation importance, so the report should cite the permutation and SHAP results.

Outputs (in `results/`): `T15_feature_importance.csv` (every score, its rank per method, the mean rank, and the direction) and `T15_grouped_permutation_importance.csv`.

To rerun it, run T12 first, then open the notebook and run all cells (requires `shap` in addition to the T12 packages).

## Task status

*After finishing each task, add 2–3 sentences to its Comments cell in the table below: what you did, what you found, and any decisions you made (for example, "input tokens = prompt only"). These notes become most of the report's Data and Methods sections, so writing them as you go saves time at the end.*

Legend: ✅ Done · 🟨 In progress · ⬜ Not started · ⛔ Blocked

To update, edit this README on GitHub (pencil icon) and change the emoji in the Status column. The full task descriptions are in the Group 01 project plan.

| # | Task | Owner | Due | Status | Depends on | Comments |
|---|---|---|---|---|---|---|
| T01 | Finalize and submit the proposal | Javier | Nov 9 | ✅ Done | — |  |
| T02 | Create the GitHub repo, README, folder structure | Alex | Nov 9 | ✅ Done | — |  |
| T03 | Download RAGTruth and confirm both files load | Alex | Nov 9 | ✅ Done | T02 |  |
| T04 | Join files, build the label, apply exclusions | Angelica | Nov 13 | ✅ Done | T03 |  |
| T05 | Count input and output tokens | Alex | Nov 13 | ✅ Done | T04 | Input tokens = prompt only, because the prompt already includes the source. Counts use cl100k_base (exact for GPT, approximate for Llama-2 and Mistral). Output: data/processed/T05_token_counts.csv, joinable to features.csv on id. |
| T06 | Record price assumptions with sources | Alex | Nov 16 | ✅ Done | — | All prices are in T06_config.yaml with sources, checked 2026-11-14. Both GPT models are retired, so last published prices are used; local models use hosted-API prices as a proxy. The $1.00 miss cost is an assumption varied in T19. |
| T07 | Engineer the features | Angelica | Nov 16 | ✅ Done | T04 |  |
| T08 | Check the train/test split for leakage | Javier | Nov 16 | ✅ Done | T04 | The predefined `split` column has no `source_id` in both train (14,942 rows) and test (2,675 rows), so it is leakage-safe and all models use it. Hallucination rate is 45% in train vs. 35% in test. |
| T09 | Build baseline models | Javier | Nov 16 | ✅ Done | T07, T08 | Trained logistic regression, decision tree, and random forest (balanced class weights) on the T08 split. Logistic regression is the benchmark (F1 0.713, ROC-AUC 0.872), but all three are within 0.002 F1. Output: data/processed/baseline_model_results.csv. |
| T10 | Submit the status update; remind check-ins | Alex | Nov 16 | ⬜ Not started | — |  |
| T11 | Make exploratory charts | Angelica | Nov 20 | ⬜ Not started | T04, T07 |  |
| T12 | Train random forest and XGBoost | Javier | Nov 23 | ✅ Done | T09 | Tuned 8 settings per model with 5-fold stratified CV on F1, with balanced class weights (RF) and scale_pos_weight (XGBoost). Random forest is best on test F1 (0.731 vs. 0.713 for the benchmark), XGBoost is close (0.723); ROC-AUC barely moves (0.872 → 0.875). Test probabilities for all three models are in results/predictions.csv for T17 and T18. |
| T13 | Compute hallucination rates by model | Angelica | Nov 23 | ⬜ Not started | T04 |  |
| T14 | Compute cost per answer and per trustworthy answer | Alex | Nov 23 | ⬜ Not started | T05, T06 |  |
| T15 | Find the most important features | Javier | Nov 27 | ✅ Done | T12 | Compared built-in importance, test-set permutation importance (single and grouped one-hots), and SHAP for the tuned RF and XGBoost. The LLM that wrote the answer is the strongest factor (ROC-AUC drop 0.14), then word_overlap (0.05–0.09; higher overlap → less hallucination); length_ratio adds almost nothing. Outputs: results/T15_feature_importance.csv and results/T15_grouped_permutation_importance.csv. |
| T16 | Cluster the hallucinated answers | Angelica | Nov 27 | ⬜ Not started | T07 |  |
| T17 | Find the best threshold; simulate local-first fallback | Alex | Nov 30 | ⬜ Not started | T12, T14 |  |
| T18 | Build the learning curve | Alex | Nov 30 | ⬜ Not started | T12 |  |
| T19 | Run the sensitivity analysis | Alex | Nov 30 | ⬜ Not started | T17 |  |
| T20 | Draft your own report sections | All | Nov 30 | ⬜ Not started | T11, T13, T15, T16, T18, T19 |  |
| T21 | Clean up the code (PEP 8, README run steps) | Alex | Dec 3 | ⬜ Not started | T20 |  |
| T22 | Assemble the final report | Alex | Dec 4 | ⬜ Not started | T20 |  |
| T23 | Build the slides | All (Alex merges) | Dec 4 | ⬜ Not started | T20 |  |
| T24 | Rehearse, record, and upload the presentation | All | Dec 6 | ⬜ Not started | T23 |  |
| T25 | Final submission | Alex | Dec 7 | ⬜ Not started | T21, T22, T24 |  |
| T26 | Peer evaluation | All | Dec 7 | ⬜ Not started | — |  |

## Final deliverables (due Dec 7)

Everything below is required by Assignment 7.2. One team member submits; no extensions are given.

| Deliverable | What's required |
|---|---|
| 1. Final report | About 10 pages (appendices excluded), APA 7 style, Word or PDF. Must cover: purpose, goals, and scope, with references; whether we adapted a published algorithm or built a new one, and how the work relates to course topics; a clear explanation of each algorithm with analysis, evaluation, and critique (algorithm comparisons shown as charts preferred); an appendix listing each member's contributions. |
| 2. Code | Complete, clean, well-documented code following PEP 8, submitted to Canvas or linked as this public GitHub repository in the report. A README is required. |
| 3. Slides and video | A 20–30 minute recorded presentation with every member presenting, and the video link on the title slide. |
| 4. Peer evaluation | Assignment 7.3, completed by each member individually (45 points). |
| 5. AI-use disclosure | Any AI tool use must be disclosed, cited, and explained in the report and in code comments. |

The report goes through Turnitin (a plagiarism checker), so cite every source we draw on.
