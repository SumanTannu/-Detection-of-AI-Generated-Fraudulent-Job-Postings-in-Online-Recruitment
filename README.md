# An Intelligent Framework for Detection of AI-Generated Fraudulent Job Postings in Online Recruitment

## Research Objective

This project supports a dissertation pipeline for detecting fraudulent job postings, with an eventual three-class task:

- `0` = Legitimate
- `1` = Human-written Fraud
- `2` = AI-generated Fraud

The completed work currently covers Phase 1 (dataset analysis), Phase 2A (cleaning and preparing the original EMSCAD dataset), and Phase 2B-A (a 20-source controlled generation-pipeline pilot). No full synthetic dataset or later dissertation phase has been implemented.

## Current Implementation Phase

Phase 1 covers EMSCAD dataset inspection and exploratory data analysis. Phase 2A creates a deterministic, duplicate-preserving, text-only processed dataset from the original EMSCAD source. Phase 2B-A tests generation, provenance, validation, and human-review workflow on only 20 legitimate source advertisements.

The completed phases do not include:

- Full-scale AI scam generation
- Transformer training
- Model evaluation
- Robustness testing
- Adaptive retraining
- Web application development

## Final Dataset Construction

The leakage-safe three-class construction stage creates `data/processed/final_3class_dataset.csv` and deterministic group-aware `train`, `validation`, and `test` files in `data/splits/` using seed `42`. Model input is `model_input_text`; labels, source IDs, generation provenance, validation, and review fields are audit metadata and must never be model features. Only explicitly accepted synthetic records are included. No model training, rebalancing, or transformer tokenization is performed in this stage.

## Project Structure

```text
AI_Job_Fraud_Detection/
├── data/
│   ├── raw/
│   │   └── emscad.csv
│   ├── processed/
│   └── splits/
├── notebooks/
│   └── 01_dataset_analysis.ipynb
├── src/
├── models/
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── confusion_matrices/
│   └── tables/
├── prompts/
├── requirements.txt
└── README.md
```

## Dataset Location

The verified EMSCAD candidate dataset is located at:

```text
data/raw/emscad.csv
```

The original workspace file `DataSet.csv` was inspected and matched the expected EMSCAD schema, including `title`, `company_profile`, `description`, `requirements`, `benefits`, and `fraudulent`. A copy was placed at the expected Phase 1 path. The raw dataset contents were not modified.

## How to Run Phase 1

From the `AI_Job_Fraud_Detection/` directory:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

To regenerate notebook outputs non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_dataset_preparation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_ai_generation_pilot.ipynb
```

Generated tables are saved under `results/tables/`. Generated plots are saved under `results/plots/`.

## Phase 1 Findings

The actual dataset dimensions are:

- Rows: `17,880`
- Columns: `18`

The detected target column is `fraudulent`. Its observed values are `f` and `t`, interpreted as legitimate and fraudulent after inspection.

Class distribution:

| Class | Count | Percentage |
|---|---:|---:|
| Legitimate | 17,014 | 95.1566% |
| Fraudulent | 866 | 4.8434% |

The observed EMSCAD counts match the dissertation proposal exactly:

- Expected legitimate records: `17,014`; observed: `17,014`
- Expected fraudulent records: `866`; observed: `866`
- Expected total records: `17,880`; observed: `17,880`

The class imbalance ratio is approximately `19.65:1` legitimate to fraudulent. No rebalancing was performed in Phase 1.

## Missing-Value Findings

Columns with substantial missingness include:

| Column | Missing Count | Missing % |
|---|---:|---:|
| salary_range | 15,012 | 83.9597% |
| department | 11,547 | 64.5805% |
| required_education | 8,105 | 45.3300% |
| benefits | 7,196 | 40.2461% |
| required_experience | 7,050 | 39.4295% |
| function | 6,455 | 36.1018% |
| industry | 4,903 | 27.4217% |
| employment_type | 3,471 | 19.4128% |
| company_profile | 3,308 | 18.5011% |
| requirements | 2,689 | 15.0391% |

No columns were dropped.

## Duplicate Findings

Duplicate checks found:

- Exact duplicate rows: `235`
- Duplicate `description` records after first occurrence: `2,785`
- Duplicate `title` records after first occurrence, after whitespace stripping: `6,965`
- No obvious ID column was detected.

Duplicates were documented only. No records were removed.

## Text-Field Findings

Major recruitment text fields identified:

- `title`
- `company_profile`
- `description`
- `requirements`
- `benefits`

Other relevant short text or descriptor fields:

- `location`
- `department`
- `employment_type`
- `required_experience`
- `required_education`
- `industry`
- `function`
- `salary_range`

Selected text-length findings:

| Column | Non-empty Records | Mean Words | Median Words |
|---|---:|---:|---:|
| title | 17,880 | 3.73 | 3 |
| company_profile | 14,572 | 131.39 | 109 |
| description | 17,880 | 217.91 | 180 |
| requirements | 15,191 | 130.40 | 104 |
| benefits | 10,684 | 70.36 | 46 |

Raw text contains HTML markup in most records, so HTML cleaning should be planned explicitly in Phase 2.

## Data-Quality Issues

Important issues documented in Phase 1:

- `17,863` records contain HTML tags in major text fields.
- `4,304` records contain the replacement character `�`, suggesting encoding artifacts.
- `6` whitespace-only object cells were found.
- `137` description values are repeated five or more times.
- `54` title values are repeated twenty or more times.
- No unexpected target values were found.
- No row has all major text fields empty.

## Methodological Decisions

The dissertation is primarily focused on textual recruitment-fraud detection. Phase 1 therefore separates core text fields from metadata and preprocessing artifacts.

Potential metadata leakage or shortcut-learning concerns:

- `fraudulent` is the target and must never be included as input.
- `in_balanced_dataset` appears to be a preprocessing or subsetting artifact and should be excluded from model inputs.
- Binary metadata such as `telecommuting`, `has_company_logo`, and `has_questions` may be predictive but could shift the project away from text-only detection.
- `salary_range`, `location`, `industry`, and similar descriptors may help classification but should be reviewed for bias, sparsity, and shortcut learning before Phase 2.

## Preliminary Recommendation - Requires Confirmation Before Phase 2

Candidate future transformer inputs:

Option A:

```text
title + description
```

Option B:

```text
title + company_profile + description + requirements + benefits
```

Option C:

```text
all relevant textual recruitment fields, including short descriptors where justified
```

Preliminary recommendation: use Option B as the primary text-only input candidate, with Option A as a minimal baseline. Exclude `fraudulent`, `in_balanced_dataset`, and binary metadata from the first transformer experiments unless a later metadata-enhanced comparison is explicitly approved.

## Phase 2A - Dataset Preparation

The raw EMSCAD source remains at `data/raw/emscad.csv` and is never modified. Phase 2A writes a separate, reproducible dataset to:

```text
data/processed/emscad_clean.csv
```

The processed dataset has `17,880` rows and these columns:

```text
source_row_id, title, company_profile, description, requirements, benefits,
text, fraudulent, text_group_id
```

`source_row_id` is a generated 0-based position in the raw file, not an original EMSCAD identifier. `text_group_id` is a deterministic SHA-256 hash of exact cleaned canonical text. It is retained solely to support future group-aware splitting, so identical cleaned advertisements can be kept out of different train/validation/test splits. It is not a model feature.

### Text-Only Definition

The primary `text` field contains only non-empty cleaned versions of the five core recruitment fields, joined with explicit labels:

```text
Title: ...

Company Profile: ...

Description: ...

Requirements: ...

Benefits: ...
```

Missing core text becomes an empty string only in the processed representation. No content is invented, and raw metadata missing values are preserved in the raw file.

### Cleaning Rules

- Decode HTML entities and remove HTML markup while preserving readable word boundaries.
- Normalize line breaks, non-breaking spaces, and repeated whitespace.
- Replace only actual Unicode replacement placeholders and their UTF-8 mojibake form with spaces; surrounding content is retained.
- Preserve case, normal punctuation, URLs, email-like strings, phone numbers, salary values, and other potentially meaningful signals.
- Do not stem, lemmatize, remove stopwords, aggressively remove punctuation, or perform traditional tokenization.

The raw core fields had HTML markup in `17,863` records; the cleaned core fields have `0` records with detected HTML. Exact Unicode revalidation found replacement placeholders in `24` records across the core text fields and `0` after cleaning. The earlier Phase 1 figure of `4,304` was encoding-sensitive and should not be treated as the verified placeholder count.

### Duplicate Investigation and Policy

- Exact duplicate rows after the first occurrence: `235`, across `204` duplicate groups.
- Exact full-row duplicates have identical labels, and no feature-identical group with different labels was found when `fraudulent` was excluded.
- Duplicate description groups: `1,052`; duplicate descriptions after the first occurrence: `2,785`.
- One duplicate-description group (three records) has conflicting labels, so automated duplicate deletion is not justified.
- Repeated titles are reported but are not treated as repeated advertisements.

No records are removed in `emscad_clean.csv`. A separate deduplicated dataset has not been created. Any later removal policy requires an explicit methodological decision and must preserve a traceable dataset version.

### Metadata Excluded from the Primary Text Input

`fraudulent` is the target and `in_balanced_dataset` is a likely prior balancing/subsetting artifact; neither may be used as model input. The latter contains `16,980` `f` values and `900` `t` values; its `t` subset has an even `450` legitimate / `450` fraudulent split, which is strong evidence of an artifact.

Binary metadata (`telecommuting`, `has_company_logo`, `has_questions`) and contextual metadata (`salary_range`, `location`, `industry`, `employment_type`, `required_experience`, `required_education`, `function`, `department`) are also excluded from the primary text-only representation. They remain available in raw EMSCAD for later, explicitly approved ablation work.

### Phase 2A Outputs

- `notebooks/02_dataset_preparation.ipynb`
- `src/preprocessing.py`
- `data/processed/emscad_clean.csv`
- `results/tables/duplicate_analysis.csv`
- `results/tables/in_balanced_dataset_analysis.csv`
- `results/tables/data_dictionary.csv`
- `results/tables/replacement_character_analysis.csv`
- `results/tables/preprocessing_before_after.csv`
- `results/tables/cleaning_examples.csv`

### Unresolved Methodological Questions

- Should future model splits be group-aware using `text_group_id` only, or should a separately approved near-duplicate rule be added?
- How should the one conflicting-label duplicate-description group be handled in later experiments?
- Should `title + description` be retained as a smaller text-only baseline alongside the five-field canonical text?
- Should metadata-enhanced experiments be permitted as a clearly separated secondary comparison after the text-only baseline is established?

At completion of the mock pilot, no real AI-generated record had been created. The separate real Groq pilot is documented below. No transformer has been trained, no data split has been created, and no model evaluation or robustness testing has been performed.

## Phase 2B-A - Controlled Generation Pilot

Phase 2B-A is a deliberately small, 20-source readiness test. It does not create the final AI-generated fraud class, rebalance EMSCAD, create a three-class dataset, create train/test splits, or train/evaluate a classifier.

### Source Selection and Provenance

The pilot selects exactly `20` records using a fixed seed (`42`) from the verified legitimate representation (`fraudulent == 'f'`). It selects only one record from each exact `text_group_id`, then uses industry/function context and text-length buckets solely to diversify the pilot. Human-written fraudulent EMSCAD records are never used as sources.

Each raw pilot audit record records a stable `synthetic_id`, `source_row_id`, `source_text_group_id`, source dataset/label, generated label, provider/model, generation mode, timestamp, prompt version, structured source fields, structured generated fields, and validation audit data.

### Provider and Secret Handling

The generation interface includes mock, OpenAI, and Groq adapters. The real Groq runner adds durable checkpoints, model checks and separate outputs while reusing source selection and automatic validation.

Credentials are read only from environment variables. `.env` is ignored by Git and `.env.example` contains placeholders only. API calls require an explicit opt-in configuration:

```text
GENERATION_MODE=api
LLM_PROVIDER=openai
LLM_MODEL=your_selected_model
OPENAI_API_KEY=your_api_key_here
```

The original pilot used `mock / mock-v1`; its preserved records carry `generation_mode=mock` and the marker `MOCK - NOT REAL GENERATED DATA`. They are test artifacts, not synthetic fraud data.

### Prompting and Safety

Versioned prompts are stored in:

- `prompts/ai_fraud_generation_v1.txt`
- `prompts/ai_fraud_validation_v1.txt`

The generation prompt requests strict JSON with `title`, `company_profile`, `description`, `requirements`, and `benefits`. It requires preservation of broad job context while prohibiting real-company impersonation, real contacts, personal data, credentials, payment instructions, and generation commentary. Contact-like source details are redacted before prompt construction; EMSCAD source files are never altered.

### Validation and Human Review

Automatic validation checks structure, title/context continuity, source-relative length, multiple categories of fraud signals, generation commentary, placeholders, contact channels, repeated boilerplate, and trivial copying. Fraud keywords alone are insufficient: automatic passage requires multiple signal categories plus context and quality checks.

Automatic validation is only a triage stage. Passing records are written to `ai_fraud_pilot_validated.csv` with `needs_human_review`; no automatic pass is treated as final human acceptance. `human_review_template.csv` contains blank reviewer fields and does not claim that review occurred.

### Current Pilot Result

The executed mock run selected `20` legitimate, unique-text-group sources. It created `20` structurally valid mock JSON records, but all `20` were correctly rejected by quality/fraud validation because mock output is deliberately marked non-real. The validated CSV is therefore empty. This tests parsing, file writing, provenance, validation, error handling, and the safeguard that mock outputs never become research examples.

Pilot outputs:

- `notebooks/03_ai_generation_pilot.ipynb`
- `src/generation.py`
- `src/validation.py`
- `data/synthetic/pilot/ai_fraud_pilot_raw.jsonl`
- `data/synthetic/pilot/ai_fraud_pilot_validated.csv`
- `data/synthetic/pilot/human_review_template.csv`
- `results/tables/ai_fraud_pilot_source_selection.csv`
- `results/tables/ai_fraud_pilot_summary.csv`

### Methodological Limitations

- A 20-source pilot cannot establish the validity or diversity of a final AI-generated class.
- Generated content can be biased by the chosen LLM, prompt, safety configuration, and provider/model version.
- Rule-based validation can miss subtle scams and can reject realistic wording; an LLM validator can add its own bias.
- Human review remains necessary for context preservation, fraud plausibility, coherence, safety, and acceptance.
- Future work should assess cross-model generation and distribution shift between human-written and AI-generated fraud.
- Duplicate source text can leak across future experiments, so `source_text_group_id` and `text_group_id` must remain part of the data-governance process.

### Decisions Required Before Any Scale-Up

- Approve a real provider/model, budget, and secure local API configuration.
- Define the required human-review protocol and acceptance threshold.
- Approve the duplicate and near-duplicate policy for generation sources and future split groups.
- Review whether prompt v1 and rule thresholds preserve context while creating sufficiently realistic, non-operational research examples.

The project stops here for Phase 2B-A. No full-scale generation, final dataset construction, classifier training, or evaluation has been performed.

## Real Groq GPT Pilot

The updated `notebooks/03_ai_generation_pilot.ipynb` displays the historical mock summary read-only, then runs the separate real Groq workflow. Its default is `openai/gpt-oss-120b`, verified against the account's available models before generation. Groq documents strict JSON-schema output for this model: [models](https://console.groq.com/docs/models), [structured outputs](https://console.groq.com/docs/structured-outputs).

Configure the environment or the ignored project `.env`:

```dotenv
GENERATION_PROVIDER=groq
GENERATION_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your_groq_api_key_here
```

The notebook is explicitly API mode. Environment variables take precedence over `.env`; `.env.example` is never loaded. Actual credentials must never be placed in the example file. `python-dotenv` is the only additional dependency for this step; the existing OpenAI client calls Groq's compatible endpoint.

Run from the project directory:

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace notebooks/03_ai_generation_pilot.ipynb --ExecutePreprocessor.timeout=600
```

Real outputs are isolated at `data/synthetic/pilot/groq_gpt/openai_gpt-oss-120b/`:

- `ai_fraud_pilot_raw.jsonl`: generated candidates and failed attempts, exact request prompts, source links, numeric labels 0 and 2, response IDs, actual model, timestamps, usage, prompt and validation versions.
- `manifest.json`, `prompt.txt`, `model_availability.json`: frozen inputs, hashes, settings, prompt v2 and account model check.
- `validation_results.csv` and `ai_fraud_pilot_validated.csv`: all attempted candidates and the automatically passing subset, respectively.
- `human_review_template.csv`: full source/generated text for all 20 selected sources with blank reviewer fields.
- `summary.csv`, `errors.json`, `state.json`: actual counts, error audit and resumable checkpoints.

The source selector and seed 42 are unchanged, and source IDs must match the saved mock selection. Raw EMSCAD, the processed dataset, mock JSONL/CSVs and mock summary files are protected by SHA-256 checks. No source or rejected candidate is silently dropped or regenerated.

The first generation has exactly one attempt, including zero SDK retries. Any failed, refused, truncated or malformed first response stops the entire run. Later sources run sequentially with a 4,096-token completion cap and at most one retry for a transient HTTP error. The maximum is 20 candidates and 39 generation requests if every later source needs a retry. There are no LLM-validator calls. Finished runs reuse saved candidates; interrupted requests with unknown server outcomes require investigation before reuse. A frozen manifest prevents mixing changed prompts, source inputs, code or settings into an existing run.

Prompt `v2` preserves role/context and requires a generic unnamed employer, excluding real contacts, identities, credentials and operational payment details. Prompt `v1` remains the historical mock version. Every candidate records the exact prompt and its hash. Sanitization is conservative and does not guarantee removal of every named entity; manual privacy and impersonation checks are required.

Validation version `v2-groq-pilot` reuses the prior structural, context, multi-category fraud, copy and quality checks. Low lexical overlap or insufficient keyword evidence alone now triggers human review rather than automatic rejection. Lexical overlap is not semantic proof. Structural and quality failures are `auto_reject`; full automatic passage is `auto_pass` with `validation_status=needs_human_review`. Candidate label 2 records generation intent, not a verified research label. All 20 sources are available for review, including rejected or failed cases, and existing reviewer notes survive reruns.

Before a Llama comparison, complete human review and approve the prompt, acceptance protocol and equivalent 20-source comparison. Changing only `GENERATION_MODEL` then selects a separate output directory with identical source selection, generation procedure, validation and provenance. Models without strict schema support use JSON-object mode and the same local schema checks; record that API-format difference as a comparison limitation. Do not change prompt or validation thresholds mid-run. This step does not authorize full-scale generation, an expanded dataset, splits, training or Phase 3.

### Observed Real-API Result

The executed notebook used Groq `openai/gpt-oss-120b` in API mode, prompt `v2`, validation `v2-groq-pilot`. Account model availability and the one-request smoke test succeeded. Exactly 20 original legitimate source groups were selected and attempted.

| Measure | Observed |
|---|---:|
| Successfully generated / structurally valid | 13 |
| Generation failures | 7 |
| Automatic passes | 3 |
| Automatic rejections (all generation failures) | 7 |
| Generated candidates awaiting human review | 13 |
| Human-review template rows (includes failed sources) | 20 |
| Completed human reviews | 0 |
| Generation requests including bounded retries | 30 |
| Average source length, characters | 2970.65 |
| Average generated length, successful candidates only | 1764.15 |

All seven final failures were HTTP 429 token-per-minute rate limits. There were 17 rate-limited request attempts in total; ten sources received their one permitted retry. No failed source was regenerated after the run. Model responses that arrived successfully recorded 25,821 total tokens; this is response usage, not an independently verified billing total.

Twelve generated candidates passed the lexical-context heuristic and four passed the multi-category fraud rules; three passed all automatic checks. The other ten generated candidates were routed for manual review rather than declared non-fraudulent based only on wording. Every generated candidate still requires assessment of semantic context and credible fraud intent. The rule-based pass rate must not be presented as dataset accuracy or evidence of research validity.

The notebook completed without execution errors. Four offline tests passed, covering smoke-test stopping, malformed responses, 20-record/resume behavior, preserved reviewer notes, schema and review triage. Original raw/processed EMSCAD and preserved mock output hashes matched their pre-run values. All 20 attempted records have provenance and validation; credentials were absent from source, notebooks, prompts and exported artifacts.

Before any further generation, review the available candidates and approve a revised pacing policy that honors Groq retry timing/token limits. Keep this attempted pilot intact; filling its failed cases or launching a Llama comparison needs a separately identified follow-up run. Do not silently change the prompt or acceptance rules based on observed outcomes. The reusable source code is frozen by hash in the manifest for reproducible resumption.

## GPT Pilot Regeneration: Prompt v3

The existing `notebooks/03_ai_generation_pilot.ipynb` now starts with a separate v3 section. Historical v2 code cells are disabled to prevent writes to frozen artifacts; the original v2 summary is loaded read-only. Reusable code is in `src/regeneration_v3.py` and `src/validation_v3.py`. Run the notebook from the project/notebooks directory, or run `python -m src.regeneration_v3` from the project root. Use the existing ignored `.env` and `GROQ_API_KEY`; no new dependency is needed.

The requested source IDs are 11662, 12066, 13604, 14350, 14484, 15885, 3203, 4581, 5581, 6199, 7192 and 9208. Source 10308 is excluded. These are the original legitimate EMSCAD sources, not v2 generated text. `prompts/gpt_regeneration_v3_protocol.json` freezes source-specific factual anchors and two assignments per mechanism: sensitive information, unusual financial arrangements, suspicious application requirements, urgency/pressure, payment-related recruitment, and implausible recruitment conditions. Assigned categories are not verified labels.

`prompts/ai_fraud_generation_v3.txt` requires strict five-field JSON, source fact preservation, empty-field preservation, integrated recruitment-related indicators, and removal of unnecessary distinctive identities and generation artifacts. Changes to compensation, work arrangements or other unsupported employment facts are prohibited. Actual credentials, payment destinations and personal values are prohibited. Prompt content, hashes, parent synthetic ID, source group, numeric source label 0, intended generated label 2, provider/model and validation/review statuses are recorded for each attempted candidate.

Validation v3 checks structure, source-specific anchors, supported fact patterns, artifact terms in context, inappropriate identity/financial requests or other assigned mechanisms, integration across sentences, lexical overlap and copying. These are conservative screening rules, not exhaustive semantic or privacy verification. An automatic pass would still require independent human review. No LLM validator or claim of human approval is used.

Requests run sequentially, at least 65 seconds apart, with at most three attempts per source. HTTP 429 respects Retry-After (seconds or HTTP date), otherwise exponential backoff starts at 65 seconds. Three consecutive 429 responses stop the run; server-requested waits over 300 seconds stop without retrying early. State is saved before requests; completed or failed attempts are not automatically regenerated. Unknown interrupted outcomes stop for investigation. See [Groq rate-limit documentation](https://console.groq.com/docs/rate-limits) and [structured-output support](https://console.groq.com/docs/structured-outputs).

### Observed v3 Smoke Result

Groq `openai/gpt-oss-120b` was checked for availability and actually called in API mode, with prompt v3 and validation v3. Twelve sources were selected; only source **11662** was attempted. The API returned one structurally valid candidate after one request. There were no API failures or HTTP 429 responses. Full validation produced **0 automatic passes and 1 automatic rejection**, so the remaining **11 were not attempted**. No human approval occurred.

The assigned mechanism was `sensitive_information`. The generated advertisement preserved the traineeship/customer-service context and introduced pre-interview requests for identity and financial information without actual sensitive values or payment destinations. Role, responsibility, qualification, location, artifact, integration and similarity screens passed. These signals do not establish final realism or research validity.

**Known validator false positive:** the sole rejection reason was `Unsupported experience duration: 16-18 years`. Inspection of the original source shows the same 16-18 age eligibility, written as `16-18 year olds`. The experience-duration rule failed to distinguish age eligibility from professional experience and to normalize the singular/plural wording. This is not evidence that the candidate invented an experience requirement. The executed v3 validator and rejection remain frozen; they were not relaxed after observing this result. A separately versioned, regression-tested correction distinguishing age from experience requires approval before another generation attempt. Human review should also examine the redistribution/expansion of source-supported training and support into the benefits field and whether source institutional affiliations were generalized appropriately.

All artifacts are separate under `data/synthetic/pilot/groq_gpt_v3/`: `regenerated_records.jsonl`, `regenerated_records.csv`, `validation_results.csv`, `review_template.csv`, and `generation_summary.json`. Additional manifest, checkpoint state and model-availability files support audit/resumption. The review template includes twelve sources, explicitly identifying the eleven unattempted rows; reviewer fields are blank. Failed-smoke state prevents further requests on rerun. Raw EMSCAD, clean EMSCAD, mock artifacts and the original v2 output files are protected by SHA-256 checks.

This stopped smoke test does not establish readiness for scaling or a Llama comparison. Small purposive sampling, generator bias, rule-based validator errors, possible distribution shift and source-related leakage remain unresolved. Review the frozen candidate and approve a corrected validation version and separately identified follow-up run first. No final dataset, Llama output, split, transformer training or Phase 3 work was performed.

### GPT v3.1 Follow-up

`data/synthetic/pilot/groq_gpt_v3/` remains frozen. The follow-up is isolated at `data/synthetic/pilot/groq_gpt_v3_1/`, implemented by `src/regeneration_v3_1.py` and `src/validation_v3_1.py`, with prompt `prompts/ai_fraud_generation_v3_1.txt`. It uses the same twelve approved legitimate EMSCAD source rows, excludes source 10308, retains the same deterministic mechanism assignments, and records source label `0`, intended generated label `2`, Groq provider, `openai/gpt-oss-120b`, prompt version `v3.1`, and validation version `v3.1-groq-pilot` (lineage: `v3-groq-pilot`).

The only validation logic change is intentionally narrow: an explicit generated age range is not considered an unsupported professional-experience duration only when the same explicit age range exists in the source as age eligibility. New experience durations, benefits, compensation, employment arrangements, remote-work claims, real identifiers, origin/meta artifacts and other unsupported facts remain rejected. Offline regression tests cover both the exemption and continued rejection of a new `5 years` experience claim.

The follow-up repeats the real API smoke-test gate before any remaining sources can run. Requests remain sequential with saved state, bounded retries, Retry-After handling and exponential backoff. Every successful candidate remains `pending_human_review`; the v3.1 review template includes context preservation, fraud intent, realism, unsupported-fact, research/meta-artifact and AI-artifact questions using `Yes/Partial/No`, plus `Accept/Revise/Reject` overall suitability. No generated candidate is human-approved by this process.

#### Observed v3.1 Smoke Result

The v3.1 smoke test called Groq `openai/gpt-oss-120b` once for source `11662`. It returned valid JSON and passed structural, source-context, age-eligibility, unsupported-fact, research/meta-artifact, AI-artifact, sensitive-value, fraud-intent and source-similarity checks. It was automatically rejected because both sensitive-information indicators occurred in a single application sentence; v3.1 requires the assigned mechanism to be integrated across at least two recruitment-context sentences. This criterion was retained. The remaining eleven sources were not requested. There were no generation failures or HTTP 429 responses, and all source data plus frozen v2/v3 artifacts matched their recorded hashes.

The candidate is retained for human review as `pending_human_review`; automatic rejection is not a finding that it contains no fraud intent. Any subsequent retry needs a separately versioned prompt/protocol or a methodologically justified validation change. No Llama work, dataset merge, split or model training is authorized by this follow-up.

### GPT v3.2 Follow-up

`data/synthetic/pilot/groq_gpt_v3_2/` is a separate, immutable follow-up to v3.1. It retains the same twelve legitimate EMSCAD sources, excludes source 10308, uses Groq `openai/gpt-oss-120b`, and records prompt version `v3.2` with validation version `v3.2-groq-pilot`. The sole methodological change is the integration screen: a fraud mechanism may appear in one naturally written sentence when it is clearly part of the application or recruitment process. The validator rejects warning-style or disconnected appendices and excessive mechanism repetition. Structural, context, fraud-intent, artifact, unsupported-fact, sensitive-value and similarity checks remain unchanged.

#### Observed v3.2 Smoke Result

The v3.2 smoke test called Groq once for source `11662`. The generated candidate was structurally valid, context-preserving, similarity-consistent, free of research/meta and AI artifacts, free of sensitive values, and contained a clear sensitive-information mechanism integrated into the pre-interview process. It passed the new contextual-integration screen with one linked recruitment sentence and no disconnected appendix or excessive repetition. It was automatically rejected only because it introduced an unsupported `full-time` customer-service apprenticeship. The original source describes a traineeship leading to an apprenticeship but does not state a full-time arrangement.

The strict unsupported-employment check was retained. The remaining eleven records were not requested. There were no generation failures, no HTTP 429 responses and no human approvals. All source data and frozen v2, v3 and v3.1 artifacts matched their recorded hashes. The v3.2 candidate remains `pending_human_review`. A further generation attempt requires a separate versioned protocol that prevents unsupported employment-arrangement additions; no Llama work, merging, splitting or model training is authorized.

## Phase 1 Output Files

Key Phase 1 outputs include:

- `results/tables/dataset_summary.csv`
- `results/tables/class_distribution.csv`
- `results/tables/missing_values.csv`
- `results/tables/duplicate_analysis.csv`
- `results/tables/text_statistics.csv`
- `results/tables/text_statistics_by_class.csv`
- `results/tables/data_quality_checks.csv`
- `results/tables/column_methodology_assessment.csv`
- `results/plots/class_distribution.png`
- `results/plots/missing_values_by_column.png`
- `results/plots/major_text_missingness.png`
- `results/plots/major_text_word_counts_by_class.png`
- `results/plots/description_word_count_distribution_by_class.png`

## Unresolved Questions for Phase 2

- Which text input option should be used first: Option A, Option B, or Option C?
- Should HTML tags be stripped, converted to readable separators, or preserved in any ablation?
- How should duplicate postings be handled before train/validation/test splitting?
- Should near-duplicate advertisements be detected, not just exact duplicate text?
- Should metadata-only and text-plus-metadata baselines be allowed as secondary experiments?
- How should missing text fields be represented when concatenating inputs?
- Should encoding artifacts such as `�` be repaired or normalized during preprocessing?
