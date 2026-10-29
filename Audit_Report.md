# Methodological Audit Report, Reproducibility, and System Verification — HateGuard AI
This document records the technical and methodological audit results for the training pipeline (`HateGuard_Training_Final.ipynb`) and the inference application (`HateGuard_App_Final.ipynb`) to guarantee the integrity of experimental evidence in the publication manuscript.
---
## 1. Summary of Integrity Verification & Cross-Device Parity
| Audit Component | Verification Method | Actual Result on Final Execution | Status |
| :--- | :--- | :--- | :---: |
| **Model Weights Integrity** | SHA-256 streaming check on `model.safetensors` against `trusted_model_sha256` in the manifest. | `3ec22197cb5248be993782d490cc080efb9f2d0f341ba7884f1b7d99ff31e0df` (100% Match). | **PASS** |
| **Core Module Integrity (`hateguard_core.py`)** | SHA-256 check on shared module prior to import by Training, App, and Smoke Test. | `e07ad4c7524b1ed3e30fa0fa5a91a9163c40ab4a0547663afc15ef7f861a91b6` (v1.0.0, 100% Match). | **PASS** |
| **Preprocessing Parity (`clean_text`)** | SHA-256 hash testing on test sample vector across 4 languages (`PREPROCESSING_PARITY_SAMPLES`). | `0ebe0f02665f2f7b49c5965ceb4ce35f17ba932da93ff528b4e691d6cc5152f8` (Identical in Training & App). | **PASS** |
| **GPU vs CPU Inference Parity (200 Stratified Samples)** | Comparing FP32 probabilities on App (CPU) against Training reference (GPU) on `inference_parity_reference.csv`. | Maximum absolute difference = `1.27e-05` (`0.0000127`), mean = `1.26e-06` (tolerance threshold `1e-3`), `tier_flips_total = 0`. | **PASS** |
| **Standalone Smoke Test (Separate Python Process)** | Execution of `smoke_test_standalone.py` outside Jupyter Notebook memory using CPU. | `status: PASS`, maximum difference = `1.55e-05`, `tier_flips_within_tolerance_zone = 0`. | **PASS** |
| **Truncation Audit (`max_length = 128`)** | Token length distribution check across all 40,934 rows of the Internal Test set. | `frac_truncated = 0.0` (0% truncated; median = 24 tokens, p99 = 75 tokens, maximum = 118 tokens). | **PASS** |
| **Moderator State Machine & Database Self-Test** | Headless testing on isolated SQLite database to validate HITL atomic transaction rules. | Rejects non-STRONG items, rejects duplicate final decisions, and validates audit column completeness. | **PASS** |
---

## 2. Key Methodological Improvements Applied to Final Version
1. **Unification of FP32 Inference Path (`hateguard_core.logits_fp32`):**
   Evaluation in previous training notebooks used `Trainer.predict` with mixed-precision (`fp16=True`), which could produce subtle numerical shifts compared to CPU inference (`float32`) near threshold boundaries (noted a maximum difference of 0.0188 shifting 1 binary label and 5 tiers out of 40,934 rows). In this final version, all evaluation and application inference uses a single unified FP32 function without autocast.
2. **Dual-Reporting & Group-Based Resampling (`group_id`):**
   To prevent optimism bias due to the high proportion of augmented text (90.5% of test rows), metrics are reported side-by-side for the original text subset (`ORIGINAL_ONLY`, N = 3,907), all rows (`ALL_TEST row-weighted`), and equal weight per group (`ALL_TEST group-weighted`), with 95% Confidence Intervals from 1,000 grouped bootstrap resamples.
3. **Classic Baselines & Logit-Based LLM Evaluation (N = 400):**
   Added `TF-IDF + Logistic Regression` baseline (C = 2.0) on identical splits and expanded pairwise evaluation against `Qwen/Qwen2.5-1.5B-Instruct` to 400 independent groups using first-token logit difference scoring and threshold calibration on a separate validation set (N = 200).
4. **Probability Calibration Analysis & Guardrail Policy Ablation:**
   Included Expected Calibration Error (ECE), Brier Score, Reliability Diagram, Temperature Scaling fitting (T = 2.8091 on validation set), comparison of 6 moderation policies (Policies A through F), and moderator workload projections under low prevalence scenarios (1%, 5%, 10%).
5. **Deserialization Security & Application Log Accountability:**
   The application verifies the SHA-256 hash of `preprocessing_bundle.joblib` prior to invoking `joblib.load()`, detects long comments exceeding 128 tokens (`was_truncated`), and enforces logging of moderator identity (`decided_by`) on every final decision.
---

## 3. Provenance Notes & Transparently Documented Limitations
- Evaluation was run in `verify_frozen` mode on frozen trained checkpoint weights (`TRAINED_IN_THIS_RUN = False`). Text sets and labels on the Internal Test set (40,934 rows) and Official Held-Out Test set (1,995 rows) were verified to be 100% identical to the prediction CSVs from the original training run, and training parameters were verified to match via `config.json`.
- The manifest from the initial training session did not store the `training_config` object separately and exhibited differences in string fingerprint sorting schema for train/validation partitions; this fact is explicitly noted in `experiment_manifest_addendum.json` and in the *Limitations* section of the scientific manuscript.
- Normalization ablation (V0–V3) was conducted at inference time using frozen model weights, and training was performed using a single random seed (`seed = 42`).